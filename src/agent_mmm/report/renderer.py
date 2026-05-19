"""Markdown rendering helpers: HDI formatting, currency, report writing."""
from __future__ import annotations

from pathlib import Path

from agent_mmm.workspace.paths import report_path, reports_dir


def fmt_currency(value: float, currency: str = "ILS", decimals: int = 0) -> str:
    """Format a number with currency label."""
    sign = "₪" if currency == "ILS" else currency + " "
    formatted = f"{value:,.{decimals}f}"
    return f"{sign}{formatted}"


def fmt_hdi(mean: float, low: float, high: float, currency: str | None = None, decimals: int = 2) -> str:
    """Format a value with 89% HDI bracket."""
    if currency:
        return (
            f"{fmt_currency(mean, currency)} "
            f"(89% CI: {fmt_currency(low, currency)}–{fmt_currency(high, currency)})"
        )
    return f"{mean:.{decimals}f} (89% CI: {low:.{decimals}f}–{high:.{decimals}f})"


def fmt_roas(mean: float, low: float, high: float) -> str:
    return f"{mean:.2f}x (89% CI: {low:.2f}x–{high:.2f}x)"


def save_report(role: str, content: str) -> Path:
    """Write report markdown to mmm-workspace/reports/<role>.md."""
    reports_dir().mkdir(parents=True, exist_ok=True)
    path = report_path(role)
    path.write_text(content, encoding="utf-8")
    return path
