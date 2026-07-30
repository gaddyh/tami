import os
import uuid

from dotenv import load_dotenv

from app.agent.agent import run_agent

load_dotenv()

DEFAULT_MESSAGE = "Remind me tomorrow at 9 to call Dana"


def foo() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )

    thread_id = str(uuid.uuid4())

    print(f"Running default example:\n{DEFAULT_MESSAGE}\n")

    result = run_agent(
        DEFAULT_MESSAGE,
        thread_id=thread_id,
    )

    print(f"Agent: {result.response}")
    print(f"(latency: {result.latency_seconds:.2f}s)")

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key.")

    print("Reminder Agent — type a message to create a reminder, or 'quit' to exit.\n")

    thread_id = str(uuid.uuid4())
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        result = run_agent(user_input, thread_id=thread_id)
        print(f"Agent: {result.response}")
        print(f"(latency: {result.latency_seconds:.2f}s)\n")


if __name__ == "__main__":
    foo()
