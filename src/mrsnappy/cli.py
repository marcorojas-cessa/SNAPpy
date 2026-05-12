from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import detect, init_config, optimize, optimize_dry_run


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="mrsnappy", description="SNAPpy 3D fluorescent puncta detection")
    sub = parser.add_subparsers(dest="command", required=True)

    init_config = sub.add_parser("init-config", help="Write an editable default optimizer config.")
    init_config.add_argument("--output", type=Path, required=True)

    optimize_cmd = sub.add_parser("optimize", help="Optimize a SNAPpy model from labeled train/val images.")
    optimize_cmd.add_argument("--train-dir", dest="train_dir", type=Path, default=None, help="Labeled training folder containing train/ and val/ subfolders.")
    optimize_cmd.add_argument("--out-dir", dest="out_dir", type=Path, default=None, help="Output folder for the optimized model and optimization records.")
    optimize_cmd.add_argument("--config", type=str, default="default", help="`default` or a JSON/YAML optimizer config path.")
    optimize_cmd.add_argument("--dataset-name", type=str, default=None)
    optimize_cmd.add_argument("--match-distance", type=float, default=None, help="Candidate-to-ground-truth matching distance in voxel units.")
    optimize_cmd.add_argument("--dry-run", action="store_true", help="Print optimizer plan and safety checks without training.")
    optimize_cmd.add_argument("--export-optimize-report", action="store_true", default=None, help="Export optimizer audit CSV/JSON files into export_optimize_report/.")
    optimize_cmd.add_argument("--export-candidate-features", action="store_true", default=None, help="Export validation candidate features for the winning pipeline into export_candidate_features/.")

    detect_cmd = sub.add_parser("detect", help="Detect puncta in one image, an image folder, or an image list.")
    detect_cmd.add_argument("--model", type=str, required=True, help="Path to a native model.joblib created by `mrsnappy optimize`.")
    detect_cmd.add_argument("--input", type=Path, default=None, help="Input TIFF image or folder of TIFF images.")
    detect_cmd.add_argument("--input-list", type=Path, default=None, help="Text file containing one TIFF path per line.")
    detect_cmd.add_argument("--output", type=Path, required=True, help="Output CSV for one image, or output folder such as /path/to/detections for multiple images.")
    detect_cmd.add_argument("--config", type=Path, default=None, help="Optional pipeline override. Optimized models embed their recipe.")
    detect_cmd.add_argument("--score-threshold", type=float, default=None)
    detect_cmd.add_argument("--export-candidate-features", action="store_true", help="Export all model-scored candidate features into export_candidate_features/.")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "init-config":
        print(init_config(args.output))
        return
    if args.command == "optimize":
        if args.out_dir is None:
            raise SystemExit("mrsnappy optimize requires --out-dir.")
        if args.dry_run:
            plan = optimize_dry_run(
                config=args.config,
                out_dir=args.out_dir,
                train_dir=args.train_dir,
                dataset_name=args.dataset_name,
                match_distance=args.match_distance,
            )
            print(json.dumps(plan, indent=2))
            return
        summary = optimize(
            config=args.config,
            out_dir=args.out_dir,
            train_dir=args.train_dir,
            dataset_name=args.dataset_name,
            match_distance=args.match_distance,
            export_optimize_report=args.export_optimize_report,
            export_candidate_features=args.export_candidate_features,
        )
        print(summary)
        return
    if args.command == "detect":
        outputs = detect(
            model=args.model,
            input_path=args.input,
            input_list=args.input_list,
            output=args.output,
            config=args.config,
            score_threshold=args.score_threshold,
            export_candidate_features=args.export_candidate_features,
        )
        print(json.dumps({key: [str(path) for path in value] for key, value in outputs.items()}, indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
