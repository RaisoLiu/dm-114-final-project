#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Artifact:
    path: str
    role: str
    produced_by: str
    consumed_by: str


ARTIFACTS = [
    Artifact(
        "submissions/submission_phd_below075_20260522.csv",
        "Uploaded final Kaggle submission; public MAE 0.7628",
        "make phd-below075 cached re-blend",
        "Kaggle upload and make verify-submission",
    ),
    Artifact(
        "reports/training_menu_v1.json",
        "Training menu controlling the final cached blend",
        "scripts/analyze_data_distribution.py --force-synthesis --emit-menu",
        "scripts/multi_blend_grid.py --fixed",
    ),
    Artifact(
        "reports/oof_tensor.csv",
        "Aligned OOF predictions for retained GBDT legs",
        "scripts/build_oof_tensor.py / retained experiment cache",
        "Table III and Fig. 3",
    ),
    Artifact(
        "reports/_local_eval_gate_report.csv",
        "32-row public upload ledger and calibration-gate source",
        "scripts/local_eval_gate.py / Kaggle history audit",
        "Table IV and Fig. 2",
    ),
    Artifact(
        "reports/_track2_fft_peaks.csv",
        "Per-region FFT peak summary for periodicity analysis",
        "scripts/track2_phase_eda.py",
        "Fig. 1",
    ),
    Artifact(
        "reports/data_characteristics_v1.md",
        "Dataset statistics and retained 5-fold rho summary",
        "scripts/analyze_data_distribution.py --emit-menu",
        "Table I and orthogonality wording",
    ),
    Artifact(
        "report/figures/generate_figures.py",
        "Report figure generator",
        "manual report code",
        "cd report && make",
    ),
    Artifact(
        "report/figures/fig1_periodicity.pdf",
        "Periodicity figure",
        "report/figures/generate_figures.py",
        "report/DM_project_Group_3.tex",
    ),
    Artifact(
        "report/figures/fig2_slope.pdf",
        "Public-ledger slope trend figure",
        "report/figures/generate_figures.py",
        "report/DM_project_Group_3.tex",
    ),
    Artifact(
        "report/figures/fig3_orthogonality.pdf",
        "OOF residual correlation figure",
        "report/figures/generate_figures.py",
        "report/DM_project_Group_3.tex",
    ),
    Artifact(
        "report/figures/fig4_cv_generalization.pdf",
        "5-fold CV diagnostic figure",
        "reports/plots/cv*.png via report/figures/generate_figures.py",
        "report/DM_project_Group_3.tex",
    ),
    Artifact(
        "report/figures/fig5_trajectory.pdf",
        "Public submission trajectory figure",
        "report/figures/generate_figures.py",
        "report/DM_project_Group_3.tex",
    ),
    Artifact(
        "report/DM_project_Group_3.pdf",
        "Canonical final report PDF",
        "cd report && make",
        "course submission",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt_cell(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    lines = [
        "# DM 114 Final Project Artifact Manifest",
        "",
        "This manifest records the artefacts needed to audit the reported final submission and PDF.",
        "The final submission path is a cached-prediction re-blend, not a full retraining of every historical model.",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "| Path | Role | Size bytes | SHA256 | Produced by | Consumed by |",
        "|---|---|---:|---|---|---|",
    ]

    for artifact in ARTIFACTS:
        path = ROOT / artifact.path
        if path.exists():
            size = str(path.stat().st_size)
            digest = sha256(path)
        else:
            size = "MISSING"
            digest = "MISSING"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{artifact.path}`",
                    fmt_cell(artifact.role),
                    size,
                    f"`{digest}`",
                    fmt_cell(artifact.produced_by),
                    fmt_cell(artifact.consumed_by),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- `make verify-submission` rebuilds the cached final CSV to `/tmp/dm114_verify_submission.csv`, validates its schema, and compares it with `submissions/submission_phd_below075_20260522.csv`.",
            "- `cd report && make && make check` regenerates figures and the canonical A4 report PDF.",
            "- Kaggle API audit on 2026-05-25 showed `submission_phd_below075_20260522.csv` as COMPLETE with public MAE `0.7628`; `submission_v17_real_match.csv` was `SubmissionStatus.ERROR`.",
        ]
    )

    (ROOT / "ARTIFACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote ARTIFACTS.md")


if __name__ == "__main__":
    main()
