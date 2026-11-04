import sounddevice as sd
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device = "cpu", compute_type = "int8")

def record():
    audio = sd.rec(
        int(5* 16000),
        samplerate = 16000,
        channels = 1,
        dtype = "float32"
    )

    sd .wait()

    #transcribing
    segments,info = model.transcribe(audio.flatten())

    text = " ".join(segments.text for segment in segments)
    print(text)
    return text