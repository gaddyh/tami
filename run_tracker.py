import json
import os
import time
from datetime import datetime
from statistics import quantiles

from rich.console import Console
from rich.table import Table

_run_stats: list[dict] = []
_current_test: dict | None = None


def record_test_start(test_name: str) -> None:
    global _current_test
    _current_test = {"name": test_name, "start": time.perf_counter()}


def record_test_latency(latency: float) -> None:
    if _current_test is not None:
        _current_test["agent_latency"] = latency


def record_test_language(lang: str) -> None:
    if _current_test is not None:
        _current_test["lang"] = lang


def record_test_end(outcome: str, error: str | None = None) -> None:
    global _current_test
    if _current_test is None:
        return
    _current_test["total_time"] = time.perf_counter() - _current_test["start"]
    _current_test["outcome"] = outcome
    _current_test["error"] = error
    _run_stats.append(_current_test)
    _current_test = None


def _get_run_dir() -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_report() -> str:
    if not _run_stats:
        return ""

    run_dir = _get_run_dir()

    passed = sum(1 for s in _run_stats if s["outcome"] == "passed")
    failed = sum(1 for s in _run_stats if s["outcome"] == "failed")
    total = len(_run_stats)
    total_time = sum(s.get("total_time", 0) for s in _run_stats)
    total_latency = sum(s.get("agent_latency", 0) for s in _run_stats)
    latencies = sorted(s.get("agent_latency", 0) for s in _run_stats)
    if latencies:
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p95 = latencies[p95_idx]
    else:
        p95 = 0

    table = Table(title=f"Agent Test Run — {os.path.basename(run_dir)}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Test", style="bold")
    table.add_column("Lang", style="blue", width=4)
    table.add_column("Type", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Agent Latency", justify="right", style="magenta")
    table.add_column("Total Time", justify="right", style="yellow")
    table.add_column("Error", style="red")

    for i, s in enumerate(_run_stats, 1):
        test_name = s["name"]
        test_lang = s.get("lang", "en")
        test_type = "single" if "single" in test_name else "two-turn"
        status = "[green]PASS[/green]" if s["outcome"] == "passed" else "[red]FAIL[/red]"
        agent_lat = f"{s.get('agent_latency', 0):.2f}s"
        total_t = f"{s.get('total_time', 0):.2f}s"
        error = (s.get("error") or "")[:80]
        table.add_row(str(i), test_name, test_lang, test_type, status, agent_lat, total_t, error)

    summary = Table(title="Summary", show_header=False, box=None)
    summary.add_column("key", style="bold")
    summary.add_column("value")
    summary.add_row("Total tests", str(total))
    summary.add_row("Passed", f"[green]{passed}[/green]")
    summary.add_row("Failed", f"[red]{failed}[/red]")
    summary.add_row("Pass rate", f"{passed / total * 100:.0f}%" if total else "N/A")
    summary.add_row("Total wall time", f"{total_time:.2f}s")
    summary.add_row("Total agent latency", f"{total_latency:.2f}s")
    summary.add_row("Avg agent latency", f"{total_latency / total:.2f}s" if total else "N/A")
    summary.add_row("95p agent latency", f"{p95:.2f}s")

    console = Console(record=True, width=120)
    console.print()
    console.print(table)
    console.print()
    console.print(summary)

    html_path = os.path.join(run_dir, "report.html")
    console.save_html(html_path)

    txt_console = Console(record=True, width=120, no_color=True, highlight=False)
    txt_console.print()
    txt_console.print(table)
    txt_console.print()
    txt_console.print(summary)
    txt_path = os.path.join(run_dir, "report.txt")
    txt_console.save_text(txt_path)

    json_path = os.path.join(run_dir, "stats.json")
    with open(json_path, "w") as f:
        json.dump(_run_stats, f, indent=2)

    try:
        from agent import SYSTEM_PROMPT, _MODEL
    except ImportError:
        _MODEL = "unknown"
        SYSTEM_PROMPT = ""

    prompt_path = os.path.join(run_dir, "system_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(f"Model: {_MODEL}\n\n{SYSTEM_PROMPT}\n")

    md_lines = [
        f"# Agent Test Run — {os.path.basename(run_dir)}",
        "",
        f"**Model:** `{_MODEL}`",
        "",
        "## Test Results",
        "",
        "| # | Test | Lang | Type | Status | Agent Latency | Total Time | Error |",
        "|---|------|------|------|--------|--------------|------------|-------|",
    ]
    for i, s in enumerate(_run_stats, 1):
        test_name = s["name"]
        test_lang = s.get("lang", "en")
        test_type = "single" if "single" in test_name else "two-turn"
        status = "PASS" if s["outcome"] == "passed" else "FAIL"
        agent_lat = f"{s.get('agent_latency', 0):.2f}s"
        total_t = f"{s.get('total_time', 0):.2f}s"
        error = (s.get("error") or "").replace("|", "\\|").replace("\n", " ")[:80]
        md_lines.append(f"| {i} | {test_name} | {test_lang} | {test_type} | {status} | {agent_lat} | {total_t} | {error} |")

    pass_rate = f"{passed / total * 100:.0f}%" if total else "N/A"
    avg_lat = f"{total_latency / total:.2f}s" if total else "N/A"

    md_lines.extend([
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total tests | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Pass rate | {pass_rate} |",
        f"| Total wall time | {total_time:.2f}s |",
        f"| Total agent latency | {total_latency:.2f}s |",
        f"| Avg agent latency | {avg_lat} |",
        f"| 95p agent latency | {p95:.2f}s |",
        "",
    ])

    md_path = os.path.join(run_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[runs] Report written to {run_dir}/")
    return run_dir
