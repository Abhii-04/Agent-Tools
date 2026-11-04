from langgraph.types import Command
from agent.agent import Agent
import asyncio
from agent.tools.audio import listen_and_transcribe

async def main():
    agent = Agent()
    try:
        await agent.setup()
        await agent.build_graph()
        print("Agent is running.")
        next_input = None
        while True:
            if isinstance(next_input,Command):
                result = await agent.run_superstep(next_input,history = None)
            else:
                user_input = await asyncio.to_thread(listen_and_transcribe)
                if not user_input:
                    continue
                if user_input.lower() in {"exit","quit"}:
                    break
                if not user_input:
                    continue
                result = await agent.run_superstep(user_input,history = None)
            next_input=( result if isinstance(result,Command) else None)
            if next_input is None and result is not None:
                print("Agent:",result)
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
    