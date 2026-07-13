import os
import uuid

from dotenv import load_dotenv

from agent import run_agent

load_dotenv()


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

        response, latency, _ = run_agent(user_input, thread_id=thread_id)
        print(f"Agent: {response}")
        print(f"(latency: {latency:.2f}s)\n")


if __name__ == "__main__":
    main()
