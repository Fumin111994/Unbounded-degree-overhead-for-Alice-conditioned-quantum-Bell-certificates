"""Stable JSON/NPZ export for M2 control runs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path, names: list[str]) -> Path:
    checksum_path = directory / "checksums.sha256"
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return checksum_path


def write_solution(
    output_root: Path,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    run_name = "__".join(
        [
            summary["functional"],
            summary["hierarchy"],
            f"level_{summary['level']}",
            summary["solver"].lower(),
        ]
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "raw_primal_dual.npz", **arrays)

    certificate_manifest = {
        "functional": summary["functional"],
        "hierarchy": summary["hierarchy"],
        "level": summary["level"],
        "solver": summary["solver"],
        "psd_blocks": [],
    }
    for block_name, stats in summary["psd_dual_stats"].items():
        entry: dict[str, Any] = {
            "block": block_name,
            "gram_array": f"dual_psd__{block_name}",
            "factor_array": f"dual_factor__{block_name}",
            "word_labels": summary["word_labels"],
            "stats": stats,
        }
        if summary["hierarchy"] == "one_sided_pvm":
            pieces = block_name.split("_")
            entry["alice_question"] = int(pieces[1][1:])
            entry["alice_outcome"] = int(pieces[2][1:])
            entry["cfnz_nice_check"] = "one fixed Alice question in this Gram block"
            entry["kpr_sparse_check"] = (
                "projectivized Bob-word Gram block; a POVM/localizer convention "
                "requires an explicit finite-level comparison map"
            )
        certificate_manifest["psd_blocks"].append(entry)
    (run_dir / "certificate_manifest.json").write_text(
        json.dumps(certificate_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_checksums(
        run_dir,
        ["summary.json", "raw_primal_dual.npz", "certificate_manifest.json"],
    )
    return run_dir
