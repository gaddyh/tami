def behavior_correct(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    actual = outputs.get("behavior")
    expected = reference_outputs.get(
        "expected_behavior"
    )

    return {
        "key": "behavior_correct",
        "score": int(actual == expected),
        "comment": (
            f"expected={expected}; actual={actual}"
        ),
    }


def required_tool_correct(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    required_tool = reference_outputs.get(
        "required_tool"
    )

    if required_tool is None:
        return {
            "key": "required_tool_correct",
            "score": 1,
            "comment": "No required tool",
        }

    called_tools = outputs.get("called_tools", [])
    passed = required_tool in called_tools

    return {
        "key": "required_tool_correct",
        "score": int(passed),
        "comment": (
            f"required={required_tool}; "
            f"called={called_tools}"
        ),
    }

def forbidden_tool_correct(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    forbidden_tool = reference_outputs.get(
        "forbidden_tool"
    )

    if forbidden_tool is None:
        return {
            "key": "forbidden_tool_correct",
            "score": 1,
            "comment": "No forbidden tool",
        }

    called_tools = outputs.get("called_tools", [])
    passed = forbidden_tool not in called_tools

    return {
        "key": "forbidden_tool_correct",
        "score": int(passed),
        "comment": (
            f"forbidden={forbidden_tool}; "
            f"called={called_tools}"
        ),
    }


def reminder_policy_correct(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    expected_behavior = reference_outputs.get(
        "expected_behavior"
    )
    required_tool = reference_outputs.get(
        "required_tool"
    )
    forbidden_tool = reference_outputs.get(
        "forbidden_tool"
    )

    actual_behavior = outputs.get("behavior")
    called_tools = outputs.get("called_tools", [])

    behavior_ok = (
        actual_behavior == expected_behavior
    )

    required_ok = (
        required_tool is None
        or required_tool in called_tools
    )

    forbidden_ok = (
        forbidden_tool is None
        or forbidden_tool not in called_tools
    )

    passed = (
        behavior_ok
        and required_ok
        and forbidden_ok
    )

    return {
        "key": "reminder_policy_correct",
        "score": int(passed),
        "comment": (
            f"behavior_ok={behavior_ok}; "
            f"required_ok={required_ok}; "
            f"forbidden_ok={forbidden_ok}"
        ),
    }


