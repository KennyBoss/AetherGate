#!/usr/bin/env python3
"""Generate a portable Markdown report from TextPy/SoA release gate artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReleaseReportError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if required:
            raise ReleaseReportError(f"Missing release report artifact: {path}") from exc
        return None
    except json.JSONDecodeError as exc:
        raise ReleaseReportError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseReportError(f"Expected a JSON object: {path}")
    return payload


def save_text(path: str, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def cell(value: Any) -> str:
    text = "-" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return lines


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) < 1.0e-12:
        return f"{numeric:.{digits}f}"
    return f"{numeric:+.{digits}f}"


def gate_rows(gates: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for gate in gates:
        metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
        key_metrics = []
        for key in [
            "benchmark_loss",
            "loss_delta",
            "best_throughput_tokens_s",
            "best_throughput_tokens_s_delta",
            "mean_pair_overlap",
            "memory_mean_pair_overlap_delta",
            "audit_check_count",
            "audit_failed_count_delta",
            "artifact_checks",
            "artifact_count",
        ]:
            if key in metrics and metrics[key] is not None:
                value = signed(metrics[key]) if "delta" in key else fmt(metrics[key])
                key_metrics.append(f"{key}={value}")
        rows.append(
            [
                gate.get("name"),
                "PASS" if gate.get("valid") is True else "FAIL",
                gate.get("failure_count", 0),
                "; ".join(key_metrics) or "-",
                gate.get("path"),
            ]
        )
    return rows


def render_failures(failures: list[Any]) -> list[str]:
    if not failures:
        return ["- No release gate failures."]
    lines: list[str] = []
    for item in failures:
        if isinstance(item, dict):
            details = item.get("failures")
            if isinstance(details, list):
                details_text = ", ".join(str(detail) for detail in details)
            else:
                details_text = str(details)
            lines.append(f"- `{item.get('gate', 'gate')}`: {details_text}")
        else:
            lines.append(f"- {item}")
    return lines


def render_audit(dashboard: dict[str, Any] | None) -> list[str]:
    if not dashboard:
        return ["- Dashboard JSON was not provided."]
    audit = dashboard.get("audit") if isinstance(dashboard.get("audit"), dict) else {}
    checks = audit.get("checks") if isinstance(audit.get("checks"), list) else []
    if not checks:
        return ["- Dashboard audit contains no checks."]
    rows = [
        [row.get("name"), "PASS" if row.get("ok") is True else "FAIL", row.get("message")]
        for row in checks
        if isinstance(row, dict)
    ]
    return table(["Check", "Status", "Message"], rows)


def render_history(history_analysis: dict[str, Any] | None) -> list[str]:
    if not history_analysis:
        return ["- History analysis JSON was not provided."]
    comparison = history_analysis.get("comparison")
    if not isinstance(comparison, dict):
        return ["- History analysis contains no comparison block."]
    rows = [
        ["baseline_mode", history_analysis.get("baseline_mode")],
        ["release_count", history_analysis.get("release_count")],
        ["winner_by_loss", comparison.get("winner_by_loss")],
        ["loss_delta", signed(comparison.get("loss_delta"))],
        ["accuracy_delta", signed(comparison.get("accuracy_delta"))],
        ["best_throughput_tokens_s_delta", signed(comparison.get("best_throughput_tokens_s_delta"), 0)],
        ["memory_mean_pair_overlap_delta", signed(comparison.get("memory_mean_pair_overlap_delta"))],
        ["audit_failed_count_delta", signed(comparison.get("audit_failed_count_delta"), 0)],
    ]
    return table(["Metric", "Value"], rows)


def render_report(
    gates_payload: dict[str, Any],
    *,
    gates_path: Path,
    dashboard: dict[str, Any] | None,
    dashboard_path: Path | None,
    history_analysis: dict[str, Any] | None,
    history_analysis_path: Path | None,
) -> str:
    if gates_payload.get("kind") != "text_release_gate_summary":
        raise ReleaseReportError("Gate summary kind must be text_release_gate_summary.")
    summary = gates_payload.get("summary")
    gates = gates_payload.get("gates")
    failures = gates_payload.get("failures")
    if not isinstance(summary, dict):
        raise ReleaseReportError("Gate summary must contain a summary object.")
    if not isinstance(gates, list) or not all(isinstance(item, dict) for item in gates):
        raise ReleaseReportError("Gate summary must contain a gates array of objects.")
    if not isinstance(failures, list):
        failures = []

    verdict = "PASS" if gates_payload.get("valid") is True else "FAIL"
    lines: list[str] = [
        "# TextPy/SoA Release Report",
        "",
        "## Verdict",
        "",
        *table(
            ["Field", "Value"],
            [
                ["release_name", summary.get("release_name")],
                ["verdict", verdict],
                ["failed_gate_count", summary.get("failed_gate_count")],
                ["release_count", summary.get("release_count")],
                ["created_at", gates_payload.get("created_at")],
                ["report_created_at", timestamp()],
            ],
        ),
        "",
        "## Release Metrics",
        "",
        *table(
            ["Metric", "Value"],
            [
                ["benchmark_loss", fmt(summary.get("benchmark_loss"))],
                ["best_throughput_tokens_s", fmt(summary.get("best_throughput_tokens_s"), 0)],
                ["memory_mean_pair_overlap", fmt(summary.get("memory_mean_pair_overlap"))],
                ["history_baseline_mode", summary.get("history_baseline_mode")],
                ["history_loss_delta", signed(summary.get("history_loss_delta"))],
                ["history_memory_overlap_delta", signed(summary.get("history_memory_overlap_delta"))],
            ],
        ),
        "",
        "## Gates",
        "",
        *table(["Gate", "Status", "Failures", "Key Metrics", "Source"], gate_rows(gates)),
        "",
        "## Failures",
        "",
        *render_failures(failures),
        "",
        "## History Trend",
        "",
        *render_history(history_analysis),
        "",
        "## Dashboard Audit",
        "",
        *render_audit(dashboard),
        "",
        "## Source Artifacts",
        "",
        *table(
            ["Artifact", "Path"],
            [
                ["gate_summary", gates_path],
                ["dashboard", dashboard_path or "-"],
                ["history_analysis", history_analysis_path or "-"],
            ],
        ),
        "",
        "## Rebuild",
        "",
        "```bash",
        "make summarize-release-gates-demo",
        "python3 generate_text_release_report.py \\",
        f"  --gates {gates_path} \\",
        f"  --dashboard {dashboard_path or '-'} \\",
        f"  --history-analysis {history_analysis_path or '-'} \\",
        "  --output-md artifacts/releases/text_release_report.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    gates_path = Path(args.gates)
    dashboard_path = Path(args.dashboard) if args.dashboard else None
    history_analysis_path = Path(args.history_analysis) if args.history_analysis else None
    gates_payload = load_json(gates_path)
    if gates_payload is None:
        raise ReleaseReportError(f"Missing gate summary: {gates_path}")
    dashboard = load_json(dashboard_path, required=False) if dashboard_path else None
    history_analysis = load_json(history_analysis_path, required=False) if history_analysis_path else None
    markdown = render_report(
        gates_payload,
        gates_path=gates_path,
        dashboard=dashboard,
        dashboard_path=dashboard_path,
        history_analysis=history_analysis,
        history_analysis_path=history_analysis_path,
    )
    save_text(args.output_md, markdown)
    summary = gates_payload["summary"]
    print("TextPy/SoA release report")
    print(f"release_name: {summary.get('release_name')}")
    print(f"valid: {gates_payload.get('valid')}")
    print(f"gates: {len(gates_payload.get('gates') or [])}")
    print(f"failed_gates: {summary.get('failed_gate_count')}")
    print(f"output_md: {args.output_md}")
    print("")
    print("RELEASE REPORT OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown release report from gate artifacts.")
    parser.add_argument("--gates", default="artifacts/releases/text_release_gates.json")
    parser.add_argument("--dashboard", default="artifacts/releases/text_release_dashboard.json")
    parser.add_argument("--history-analysis", default="artifacts/releases/text_release_history_analysis.json")
    parser.add_argument("--output-md", default="artifacts/releases/text_release_report.md")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseReportError as exc:
        raise SystemExit(str(exc)) from exc
