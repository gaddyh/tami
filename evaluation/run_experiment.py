from __future__ import annotations

import os

from dotenv import load_dotenv
from langsmith import Client

from evaluation.evaluators import (
    behavior_correct,
    forbidden_tool_correct,
    reminder_policy_correct,
    required_tool_correct,
)
from evaluation.target import run_reminder_target

load_dotenv()

DATASET_NAME = "tami-reminder-baseline"


def main() -> None:
    client = Client()

    results = client.evaluate(
        run_reminder_target,
        data=DATASET_NAME,
        evaluators=[
            behavior_correct,
            required_tool_correct,
            forbidden_tool_correct,
            reminder_policy_correct,
        ],
        experiment_prefix="tami-reminder-baseline",
        description=(
            "Initial offline behavioral baseline for "
            "the Tami reminder agent."
        ),
        metadata={
            "agent": "tami",
            "framework": "langgraph",
            "evaluation_type": "offline",
            "dataset": DATASET_NAME,
            "model": os.getenv(
                "OPENAI_MODEL",
                "unknown",
            ),
        },
        max_concurrency=1,
    )

    print(results)


if __name__ == "__main__":
    main()