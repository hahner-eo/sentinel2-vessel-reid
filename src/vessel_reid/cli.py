from __future__ import annotations

import argparse
import json

from . import __version__
from .data import read_manifest
from .evaluate import evaluate_archive
from .inference import embed_manifest
from .training import train


def _data_arguments(parser):
    parser.add_argument("manifest", help="JSONL or JSONL.gz observation manifest")
    parser.add_argument("--crop-root", help="root prepended to relative image member paths")
    parser.add_argument("--geometry", help="JSONL or JSONL.gz scene-geometry sidecar")


def _rows(args):
    return read_manifest(args.manifest, crop_root=args.crop_root, geometry_path=args.geometry)


def build_parser():
    parser = argparse.ArgumentParser(prog="vessel-reid")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-manifest", help="validate metadata and partitions")
    _data_arguments(validate)

    inference = commands.add_parser("infer", help="create gallery and query embeddings")
    _data_arguments(inference)
    inference.add_argument("checkpoint")
    inference.add_argument("output")
    inference.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="test"
    )
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--batch-size", type=int, default=64, help="maximum augmented images")

    evaluation = commands.add_parser("evaluate", help="run chronological retrieval")
    evaluation.add_argument("archive")
    evaluation.add_argument("--cutoff", required=True)
    evaluation.add_argument("--gap-days", type=int, default=7)
    evaluation.add_argument("--depth", type=int, default=20)
    evaluation.add_argument("--bootstrap-replicates", type=int, default=2000)
    evaluation.add_argument("--seed", type=int, default=20260919)

    training = commands.add_parser("train", help="train the reported encoder")
    _data_arguments(training)
    training.add_argument("initialization")
    training.add_argument("output")
    training.add_argument("--device", default="cpu")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--seed", type=int, default=20260911)
    training.add_argument("--resume", help="epoch checkpoint from this package")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        rows = _rows(args)
        counts = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        }
        print(
            json.dumps(
                {
                    "status": "valid",
                    "rows": len(rows),
                    "identities": len({row["imo"] for row in rows}),
                    "partition_rows": counts,
                }
            )
        )
    elif args.command == "infer":
        embed_manifest(
            _rows(args),
            args.checkpoint,
            args.output,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate":
        result = evaluate_archive(
            args.archive,
            args.cutoff,
            gap_days=args.gap_days,
            depth=args.depth,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
    else:
        train(
            _rows(args),
            args.initialization,
            args.output,
            device=args.device,
            epochs=args.epochs,
            seed=args.seed,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
