"""Data quality audit for MMM readiness (11 checks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Tier = Literal["PASS", "WARN", "FAIL"]

_TIER_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class AuditFinding:
    check: str
    tier: Tier
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    findings: list[AuditFinding] = field(default_factory=list)
    tier: Tier = "PASS"

    def add(self, finding: AuditFinding) -> None:
        self.findings.append(finding)
        if _TIER_RANK[finding.tier] > _TIER_RANK[self.tier]:
            self.tier = finding.tier

    def to_dict(self) -> dict:
        return {
            "overall_tier": self.tier,
            "findings": [
                {
                    "check": f.check,
                    "tier": f.tier,
                    "message": f.message,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def audit_data(df: pd.DataFrame, spec) -> AuditResult:
    """Run data quality checks. Returns AuditResult with PASS/WARN/FAIL findings."""
    result = AuditResult()
    date_col = spec.data.date_col
    target_col = spec.target.column
    channel_cols = [ch.spend_col for ch in spec.channels]
    n = len(df)

    # 1. Row count
    if n < 52:
        result.add(AuditFinding("row_count", "FAIL", f"{n} rows — need >= 52 weeks.", {"n_rows": n}))
    elif n < 104:
        result.add(AuditFinding("row_count", "WARN", f"{n} rows — 104+ recommended for CV.", {"n_rows": n}))
    else:
        result.add(AuditFinding("row_count", "PASS", f"{n} rows.", {"n_rows": n}))

    # 2. Date column presence
    if date_col not in df.columns:
        result.add(AuditFinding("date_column", "FAIL", f"Date column '{date_col}' missing."))
        return result  # cannot continue

    # 3. Date continuity (no gaps)
    dates = pd.to_datetime(df[date_col]).sort_values()
    diffs = dates.diff().dropna()
    if len(diffs):
        expected = diffs.mode().iloc[0]
        gaps = diffs[diffs > expected * 1.5]
        if len(gaps):
            result.add(AuditFinding("date_gaps", "WARN", f"{len(gaps)} gap(s) in date sequence.", {"n_gaps": len(gaps)}))
        else:
            result.add(AuditFinding("date_gaps", "PASS", "No date gaps."))

    # 4. Target present
    if target_col not in df.columns:
        result.add(AuditFinding("target_column", "FAIL", f"Target column '{target_col}' missing."))
    else:
        y = df[target_col]
        missing = int(y.isna().sum())
        tier_miss = "FAIL" if missing / n > 0.05 else ("WARN" if missing > 0 else "PASS")
        if missing:
            result.add(AuditFinding("target_missing", tier_miss, f"{missing} missing values in target."))

        # 5. Target variation
        cv = float(y.std() / y.mean()) if y.mean() != 0 else 0.0
        if cv < 0.10:
            result.add(AuditFinding("target_variation", "WARN", f"Target CV={cv:.3f} < 0.10 — low variation.", {"cv": cv}))
        else:
            result.add(AuditFinding("target_variation", "PASS", f"Target CV={cv:.3f}.", {"cv": cv}))

        # 6. Target non-negative
        if (y < 0).any():
            result.add(AuditFinding("target_negative", "FAIL", "Target contains negative values."))

    # 7. Channel columns present
    missing_ch = [c for c in channel_cols if c not in df.columns]
    if missing_ch:
        result.add(AuditFinding("channel_columns", "FAIL", f"Missing spend columns: {missing_ch}."))
    else:
        result.add(AuditFinding("channel_columns", "PASS", f"All {len(channel_cols)} spend columns present."))

    # 8. Channel zeros and negatives (per channel)
    for ch in spec.channels:
        col = ch.spend_col
        if col not in df.columns:
            continue
        s = df[col]
        if (s < 0).any():
            result.add(AuditFinding(f"channel_neg_{ch.name}", "FAIL", f"'{ch.name}' has negative spend."))
            continue
        zero_pct = float((s == 0).mean())
        if zero_pct > 0.80:
            result.add(AuditFinding(f"channel_zeros_{ch.name}", "FAIL",
                                    f"'{ch.name}': {zero_pct:.0%} zeros — cannot model.", {"zero_pct": zero_pct}))
        elif zero_pct > 0.40:
            result.add(AuditFinding(f"channel_zeros_{ch.name}", "WARN",
                                    f"'{ch.name}': {zero_pct:.0%} zeros — sparse.", {"zero_pct": zero_pct}))

    return result


def print_audit_summary(result: AuditResult) -> None:
    """Print a human-readable audit summary."""
    print(f"\nData Audit: {result.tier}")
    for f in result.findings:
        mark = {"PASS": "[OK]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[f.tier]
        print(f"  {mark} {f.message}")
