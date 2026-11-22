"""Command line entry point, with independently repeatable stages."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import cv2

from . import pipeline


def size_value(value):
    try:
        w, h = map(int, value.lower().split("x"))
        if w < 16 or h < 16 or w % 2 or h % 2:
            raise ValueError
        return w, h
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Size must be even dimensions, e.g. 1080x1920 (minimum 16x16)"
        ) from exc


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pose-aligned montage and morph video from images"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for stage in [
        "run",
        "detect",
        "relevance",
        "select",
        "align",
        "order",
        "render",
        "report",
        "evaluate",
    ]:
        p = sub.add_parser(stage)
        if stage in ["run", "detect"]:
            p.add_argument("images", type=Path, nargs="+")
            p.add_argument(
                "--poses",
                type=Path,
                help="Import offline COCO pose JSON instead of running models",
            )
            p.add_argument(
                "--person-overrides",
                type=Path,
                help="JSON mapping filename to zero-based person index",
            )
            p.add_argument("--cache", type=Path)
            p.add_argument(
                "--pose-backend",
                choices=["body7", "body7-rgb", "humanart"],
                default="body7",
                help="humanart is experimental: training-data/weight commercial scope unresolved",
            )
        p.add_argument("--out", type=Path, required=True)
        if stage == "evaluate":
            p.add_argument("--baseline-out", type=Path, action="append")
            p.add_argument("--sample-size", type=positive, default=30)
            p.add_argument("--seed", type=nonnegative, default=20261108)
            p.add_argument("--target-duration", type=float, default=35.2)
            p.add_argument("--seconds-per-cut", type=float, default=0.2)
        if stage in ["run", "relevance"]:
            p.add_argument(
                "--relevance-prompts", type=Path, required=stage == "relevance"
            )
            p.add_argument("--relevance-python", type=Path)
        if stage in ["run", "select"]:
            p.add_argument("--confidence", type=float, default=0.35)
            p.add_argument("--cluster", choices=["all", "largest"], default="all")
            p.add_argument("--cluster-threshold", type=float, default=0.55)
            p.add_argument("--max-images", type=positive)
            p.add_argument("--min-relevance", type=float)
            p.add_argument("--min-pose-conf", type=float)
            p.add_argument("--min-person-area", type=float, default=0.0)
            p.add_argument("--max-head-error", type=float)
            p.add_argument(
                "--max-upscale",
                type=float,
                help="Maximum source enlargement in percent (100 permits 2x)",
            )
        if stage in ["run", "relevance", "select", "align"]:
            p.add_argument("--size", type=size_value, default=(1080, 1920))
            p.add_argument("--alignment", choices=["similarity", "head-torso"])
        if stage in ["run", "align"]:
            p.add_argument("--mirror", choices=["off", "auto"], default="off")
        if stage in ["run", "order"]:
            p.add_argument("--start", help="Filename to put first (must be selected)")
            p.add_argument("--hook", type=nonnegative, default=0)
            p.add_argument(
                "--order",
                dest="order_method",
                choices=["path", "phase"],
                default="path",
            )
            p.add_argument("--cycles", type=positive, default=1)
            p.add_argument("--phase-bins", type=positive, default=3)
            p.add_argument("--group-by-material", action="store_true")
        if stage in ["run", "report"]:
            p.add_argument("--baseline-order", type=Path, action="append")
        if stage == "select":
            p.add_argument("--duration", type=float)
            p.add_argument("--speed-curve")
            p.add_argument("--fps", type=positive, default=30)
            p.add_argument("--hold", type=positive, default=6)
        if stage in ["run", "render"]:
            p.add_argument("--mode", choices=["cut", "morph"], default="cut")
            p.add_argument("--fps", type=positive, default=30)
            p.add_argument("--hold", type=positive, default=6)
            p.add_argument("--transition", type=nonnegative, default=4)
            p.add_argument("--audio", type=Path)
            p.add_argument("--audio-start", type=float, default=0)
            p.add_argument("--speed-curve")
            p.add_argument(
                "--duration", type=float, help="Scale cut timing to this many seconds"
            )
            p.add_argument("--beats", type=Path)
            p.add_argument(
                "--snap-beats",
                action="store_true",
                help="Detect audio onsets (also automatic with audio + speed curve)",
            )
            p.add_argument("--beat-tolerance", type=float, default=0.06)
            p.add_argument("--zoom", default="1:1")
            p.add_argument("--normalize-tone", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run" and args.relevance_prompts and args.min_relevance is None:
        args.min_relevance = 0.5
    if hasattr(args, "confidence") and not (0 < args.confidence <= 1):
        parser.error("--confidence must be in (0,1]")
    if hasattr(args, "cluster_threshold") and args.cluster_threshold <= 0:
        parser.error("--cluster-threshold must be positive")
    if hasattr(args, "audio_start") and not args.audio_start >= 0:
        parser.error("--audio-start must be nonnegative")
    if args.command in ["run", "render"]:
        if not shutil.which("ffmpeg"):
            parser.error("ffmpeg not found; install it and add it to PATH")
        if args.audio and not args.audio.is_file():
            parser.error("--audio file does not exist")
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        result = None
        for stage in (
            [
                "detect",
                *(["relevance"] if args.relevance_prompts else []),
                "select",
                "align",
                "order",
                "render",
                "report",
            ]
            if args.command == "run"
            else [args.command]
        ):
            if stage == "detect":
                result = pipeline.detect(
                    args.images,
                    args.out,
                    args.poses,
                    args.person_overrides,
                    args.cache,
                    args.pose_backend,
                )
            elif stage == "select":
                result = pipeline.select(
                    args.out,
                    args.confidence,
                    args.cluster,
                    args.cluster_threshold,
                    args.max_images,
                    args.min_relevance,
                    args.min_pose_conf,
                    args.min_person_area,
                    args.max_head_error,
                    args.max_upscale,
                    args.size,
                    args.alignment or "similarity",
                    args.duration,
                    args.speed_curve,
                    args.fps,
                    args.hold,
                )
            elif stage == "relevance":
                result = pipeline.relevance(
                    args.out,
                    args.relevance_prompts,
                    args.relevance_python,
                    args.size,
                    args.alignment or "similarity",
                )
            elif stage == "align":
                result = pipeline.align(
                    args.out, args.size, args.mirror, args.alignment
                )
            elif stage == "order":
                result = pipeline.order(
                    args.out,
                    args.start,
                    args.hook,
                    args.order_method,
                    args.cycles,
                    args.group_by_material,
                    args.phase_bins,
                )
            elif stage == "render":
                result = pipeline.render(
                    args.out,
                    args.mode,
                    args.fps,
                    args.hold,
                    args.transition,
                    args.audio,
                    args.audio_start,
                    args.speed_curve,
                    args.beats,
                    args.snap_beats,
                    args.beat_tolerance,
                    args.zoom,
                    args.normalize_tone,
                    args.duration,
                )
            elif stage == "evaluate":
                result = pipeline.evaluate(
                    args.out,
                    args.baseline_out,
                    args.sample_size,
                    args.seed,
                    args.target_duration,
                    args.seconds_per_cut,
                )
            else:
                result = pipeline.report(args.out, args.baseline_order)
            print(f"{stage}: {args.out / (stage + '.json')}", flush=True)
        if args.command in ["report", "run"]:
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k.endswith("count")
                        or k
                        in [
                            "drop_reasons",
                            "filter_drop_counts",
                            "continuity",
                            "alignment_consistency",
                            "rendered_alignment_consistency",
                        ]
                    },
                    indent=2,
                )
            )
            continuity = result.get("continuity", {})
            comparisons = [
                ("random -> shown", continuity.get("random"), continuity.get("shown")),
                *[
                    (
                        f"{name} -> current (shared images)",
                        data["before"],
                        data["after"],
                    )
                    for name, data in continuity.get("baselines", {}).items()
                ],
            ]
            for label, before, after in comparisons:
                if before and after:
                    print(
                        f"continuity {label}: mean {before['mean']:.4f} -> {after['mean']:.4f}; p90 {before['p90']:.4f} -> {after['p90']:.4f}"
                    )
        return 0
    except (ValueError, TypeError, OSError, cv2.error) as exc:
        print(f"posemorph: {exc}", file=sys.stderr)
        return 1
