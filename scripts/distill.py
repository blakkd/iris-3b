"""Distill Iris-3B into a few-step student with Trajectory Distribution Matching.

Single node, one process per GPU (CUDA with bf16 support is required):

    torchrun --nproc_per_node=8 scripts/distill.py --config configs/iris3b/turbo_stage1_6step.yaml \
        prompts_file=/data/prompts.txt

Trailing ``key=value`` pairs are dotlist overrides merged onto the YAML config.

Training is data-free: it reads only text prompts. ``prompts_file`` is a UTF-8
text file with one prompt per line; blank lines are skipped, and the file is
read in order and cycled, each rank taking its own slice of every global batch.
The teacher is an exported directory (``model.safetensors`` + ``config.yaml``,
e.g. ``hf download speridlabs/iris-3b --exclude "depth/*" "upscaler/*" "turbo/*"
--local-dir iris-3b``) or a training ``.pth``.

Every ``save_every`` student updates the run writes
``work_dir/checkpoints/step_XXXXXXXX.pth`` (student, EMA, fake network, both
optimizers and the RNG state) and points ``latest.pth`` at it. Continue an
interrupted run with ``resume_from=<work_dir>/checkpoints/latest.pth``, same
config and GPU count. ``scripts/export_distilled.py`` turns a checkpoint into a
directory that ``scripts/sample.py`` samples with the few-step schedule.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris3b.distill.config import load_distill_config
from iris3b.distill.trainer import DistillationTrainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    DistillationTrainer(load_distill_config(args.config, args.overrides)).run()


if __name__ == "__main__":
    main()
