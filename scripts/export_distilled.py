"""Export a distillation checkpoint as a few-step sampling directory.

Usage:
    python scripts/export_distilled.py output/turbo/stage2_4step/checkpoints/step_00000500.pth \
        exported/iris-3b-turbo --dtype bf16

Writes what ``scripts/export_checkpoint.py`` writes, ``OUT_DIR/model.safetensors``
(the student's EMA weights; no fake network, teacher, optimizer or RNG state)
and ``OUT_DIR/config.yaml``, plus ``OUT_DIR/distill.yaml``: the few-step
schedule (``dmd``) and the training size. ``scripts/sample.py --checkpoint
OUT_DIR`` samples it with that schedule and no CFG. The released Iris-3B Turbo
is the stage-2 EMA at step 500 exported with ``--dtype bf16``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from omegaconf import OmegaConf

from export_checkpoint import DTYPES
from export_checkpoint import main as export_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="checkpoint written by scripts/distill.py")
    parser.add_argument("out_dir", help="output directory")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if "distill_config" not in payload:
        parser.error(
            f"{args.checkpoint} was not written by scripts/distill.py; use scripts/export_checkpoint.py"
        )
    export_checkpoint([args.checkpoint, args.out_dir, "--dtype", args.dtype])
    distill = Path(args.out_dir) / "distill.yaml"
    sections = {key: payload["distill_config"][key] for key in ("dmd", "height", "width")}
    OmegaConf.save(OmegaConf.create(sections), str(distill))
    print(f"wrote {distill}")


if __name__ == "__main__":
    main()
