"""Fetch the GGUF models used by the measurements in RESULTS.md.

Optional helper - nothing else in this repository calls it. The benchmark scripts take a
`-Model` path, so any copy of these files works; this just saves hunting for the right one.

    uv run scripts/download_models.py --dest ./models
    uv run scripts/download_models.py --dest E:/big-disk/models --only qwen
    uv run scripts/download_models.py --dest ./models --check-only

Sources were chosen by matching the exact byte size of the files that produced the published
numbers. For the Nemotron model that is ggml-org's repository: bartowski also publishes a
file with the same name, but it is a different quantisation run (19063900128 bytes) and would
not be the same weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Xet is huggingface_hub's fast transfer path; it has to be enabled before the import.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


@dataclass(frozen=True)
class Model:
    key: str
    repo: str
    filename: str
    size: int
    note: str


MODELS = (
    Model(
        key="nemotron",
        repo="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        filename="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf",
        size=18_898_091_584,  # 17.6 GiB on disk; llama-bench reports 17.59 GiB. Larger than a
                             # 16 GB card either way, so part of it always lands on the CPU.
        note="the main measurement model; must not fit in VRAM",
    ),
    Model(
        key="qwen",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size=491_400_032,  # 469 MiB - the "fits entirely in VRAM" control
        note="control: fits entirely in VRAM, where the gap disappears",
    ),
)


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GiB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MiB"


def remote_size(model: Model) -> int | None:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        return get_hf_file_metadata(hf_hub_url(model.repo, model.filename)).size
    except Exception as exc:  # network, auth, renamed file - all reported the same way
        print(f"  could not read remote metadata: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path,
                    help="directory to download into (no default - models are large)")
    ap.add_argument("--only", choices=[m.key for m in MODELS], action="append",
                    help="download just this model; repeatable. Default: all")
    ap.add_argument("--check-only", action="store_true",
                    help="report remote sizes and what is already present, download nothing")
    args = ap.parse_args()

    wanted = [m for m in MODELS if not args.only or m.key in args.only]
    args.dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    failures = 0
    for m in wanted:
        target = args.dest / m.filename
        print(f"{m.key}: {m.filename}")
        print(f"  {m.note}")
        print(f"  repo     {m.repo}")
        print(f"  expected {human(m.size)} ({m.size} bytes)")

        if target.exists():
            actual = target.stat().st_size
            if actual == m.size:
                print(f"  present  {target} - size matches, skipping\n")
                continue
            print(f"  present  {target} but {actual} bytes, expected {m.size} - re-downloading")

        if args.check_only:
            size = remote_size(m)
            if size is None:
                failures += 1
            elif size == m.size:
                print(f"  remote   {size} bytes - matches\n")
            else:
                print(f"  remote   {size} bytes - DIFFERS from the published run\n")
                failures += 1
            continue

        try:
            path = hf_hub_download(
                repo_id=m.repo,
                filename=m.filename,
                local_dir=args.dest,
            )
        except KeyboardInterrupt:
            print("\n  interrupted - hf_hub_download resumes from where it stopped\n")
            return 130
        except Exception as exc:
            print(f"  FAILED: {exc}\n", file=sys.stderr)
            failures += 1
            continue

        actual = Path(path).stat().st_size
        if actual == m.size:
            print(f"  done     {path}\n")
        else:
            print(f"  WARNING  {path} is {actual} bytes, expected {m.size}\n", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} model(s) not in the expected state", file=sys.stderr)
        return 1
    if args.check_only:
        print("every source matches the size that produced the published numbers")
    else:
        print("all requested models are present with the expected size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
