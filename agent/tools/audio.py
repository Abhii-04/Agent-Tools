import queue
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_MS = 800

audio_queue = queue.Queue()

def load_silero():
    torch.set_num_threads(1)

    model = load_silero_vad()
    vad = VADIterator(
        model,
        sampling_rate = SAMPLE_RATE,
        threshold=0.5,
        min_silence_duration_ms=SILENCE_MS,
    )
    return vad


def load_whisper_model():
    whisper = WhisperModel(
        "base.en",
        device = "cpu",
        compute_type = "int8",
    )
    return whisper

vad = load_silero()
whisper = load_whisper_model()

def listen_and_transcribe():
    
    audio_queue = queue.Queue()
    pre_buffer = deque(maxlen=10)
    recorded = []
    speaking = False

    vad.reset_states()

    #capture audio
    def callback(indata,frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata[:,0].copy())
    
    print("\nListening...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels = 1,
        dtpye = "float32",
        blocksize = CHUNK_SIZE,
        callback = callback
    ):
        while True:
            chunk = audio_queue.get() #get audio directly from microphone
            event = vad(torch.from_numpy(chunk))  #silero checks for spech here

            if not speaking:
                pre_buffer.append(chunk)

                if event and "start" in event:
                    speaking = True
                    recorded.extend(pre_buffer)
                    print("speech detected")

            else:
                recorded.append(chunk)
                if event and "end" in event:
                    print("Transcribing...")
                    break
    
    #combine audio chunks into 1 np array
    audio = np.concatenate(recorded)  

    #faster-whisper converts audio into text
    segments,_ = whisper.transcribe(
        audio,
        language="en",
        vad_filter=False,
    )

    text = " ".join( segment.text for segment in segments).strip()

    return text
    
