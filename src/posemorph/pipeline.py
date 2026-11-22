"""Disk-backed stages. Source images are only opened for reading."""

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .continuity import (
    SWAP,
    continuity_metrics,
    material_features,
    material_labels,
    mirror_plan,
    phase_path,
    reflect_pose,
)
from .finishing import (
    ToneNormalizer,
    audio_onsets,
    camera_frame,
    cut_timing,
    parse_zoom,
    read_beats,
)
from .geometry import (
    EDGES,
    distance_matrix,
    morph_frames,
    order_path,
    path_length,
    target_torso,
)
from .quality import (
    alignment_geometry,
    alignment_summary,
    automatic_count,
    person_area,
    pose_confidence,
    rendered_alignment_summary,
)
from .relevance import read_prompts

SCHEMA = 1
DETECTOR_REVISION = 1


def save(out, stage, data):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        **data,
        "schema": SCHEMA,
        "stage": stage,
        "created": datetime.now(UTC).isoformat(),
    }
    target = out / f"{stage}.json"
    partial = target.with_suffix(".json.tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(target)
    return data


def load(out, stage):
    path = Path(out) / f"{stage}.json"
    if not path.exists():
        raise ValueError(f"Missing {path.name}; run {stage} first")
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or data.get("stage") != stage:
        raise ValueError(f"Unsupported manifest: {path}")
    # Detect an upstream rerun rather than silently using stale transforms/order.
    for name, expected in data.get("inputs", {}).items():
        if digest(Path(out) / f"{name}.json") != expected:
            raise ValueError(f"{stage}.json is stale after {name}; rerun {stage}")
        load(out, name)
    return data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(out, *stages):
    return {s: digest(Path(out) / f"{s}.json") for s in stages}


def read_image(path):
    with Image.open(path) as im:
        return cv2.cvtColor(
            np.array(ImageOps.exif_transpose(im).convert("RGB")), cv2.COLOR_RGB2BGR
        )


def source_path(record, detected):
    if "source_path" in record:
        return Path(record["source_path"])
    return Path(detected["source"]) / record["file"]


def validate_person(person):
    if (
        not isinstance(person, dict)
        or not {"keypoints", "scores", "box"} <= person.keys()
    ):
        raise ValueError("Each person must contain keypoints, scores, and box")
    p, s = np.asarray(person["keypoints"], float), np.asarray(person["scores"], float)
    box = np.asarray(person["box"], float)
    if p.shape != (17, 2) or s.shape != (17,) or box.shape != (4,):
        raise ValueError("Expected 17 COCO xy keypoints, 17 scores, and xyxy box")
    if not (np.isfinite(p).all() and np.isfinite(s).all() and np.isfinite(box).all()):
        raise ValueError("Pose data must be finite")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Scores must be in [0,1]")
    return person


def detect(images, out, poses=None, overrides=None, cache=None, pose_backend="body7"):
    start = time.monotonic()
    roots = [
        Path(p).resolve()
        for p in (images if isinstance(images, (list, tuple)) else [images])
    ]
    out = Path(out)
    files = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Image directory does not exist: {root}")
        files.extend(
            sorted(
                p
                for p in root.iterdir()
                if not p.name.startswith(".")
                and p.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
            )
        )
    source_id = str(roots[0]) if len(roots) == 1 else list(map(str, roots))
    if not files:
        raise ValueError("No supported images found")
    supplied = json.loads(Path(poses).read_text()) if poses else None
    choices = json.loads(Path(overrides).read_text()) if overrides else {}
    if supplied is not None and (
        not isinstance(supplied, dict)
        or any(not isinstance(people, list) for people in supplied.values())
    ):
        raise ValueError("--poses must map filenames to lists of people")
    if not isinstance(choices, dict):
        raise TypeError("--person-overrides must map filenames to person indices")
    old = {}
    if (out / "detect.json").exists():
        previous = json.loads((out / "detect.json").read_text())
        if (
            previous.get("source") == source_id
            and previous.get("pose_input") == supplied
            and previous.get("detector_revision") == DETECTOR_REVISION
            and previous.get("pose_backend", "body7") == pose_backend
        ):
            old = {r["file"]: r for r in previous["records"] if not r.get("error")}
    detector = None
    model_info = None
    records = []
    for i, path in enumerate(files):
        name = (
            path.name
            if len(roots) == 1
            else f"{roots.index(path.parent) + 1:02d}/{path.name}"
        )
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        cached = old.get(name)
        if cached and cached.get("stamp") == stamp:
            record = dict(cached)
        else:
            # A model/download failure is fatal, not an unreadable-image rejection.
            if supplied is None and detector is None:
                from .detector import Detector

                detector = Detector(cache, backend=pose_backend)
            record = {"file": name, "source_path": str(path), "stamp": stamp}
            try:
                image = read_image(path)
                record["size"] = [image.shape[1], image.shape[0]]
                if supplied is not None:
                    people = supplied.get(name, [])
                else:
                    people = detector(image)
                    model_info = detector.models
                record["people"] = [validate_person(p) for p in people]
            except (OSError, ValueError, cv2.error) as exc:
                record["error"] = str(exc)
                record["people"] = []
        index = choices.get(name, 0)
        if (
            not isinstance(index, int)
            or index < 0
            or (record["people"] and index >= len(record["people"]))
        ):
            raise ValueError(f"Invalid person override for {path.name}: {index}")
        record["chosen"] = index
        if record["people"]:
            record.update(record["people"][index])
        records.append(record)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"detect {i + 1}/{len(files)}", flush=True)
            # Incremental checkpoint: interrupted inference resumes at the next image.
            save(
                out,
                "detect",
                {
                    "source": source_id,
                    "detector_revision": DETECTOR_REVISION,
                    "pose_backend": pose_backend,
                    "pose_input": supplied,
                    "models": model_info or (previous.get("models") if old else None),
                    "records": records,
                    "complete": i + 1 == len(files),
                    "seconds": time.monotonic() - start,
                },
            )
    return load(out, "detect")


def relevance(
    out, prompts_path, python=None, size=(1080, 1920), alignment="similarity"
):
    start = time.monotonic()
    detected = load(out, "detect")
    if not detected.get("complete"):
        raise ValueError("Detection was interrupted; rerun detect")
    prompts = read_prompts(prompts_path)
    request = {"prompts": prompts, "records": []}
    for r in detected["records"]:
        if r.get("error"):
            continue
        item = {
            "file": r["file"],
            "path": str(source_path(r, detected)),
            "stamp": r["stamp"],
            "view": "full",
        }
        request["records"].append(item)
        if r.get("people"):
            try:
                head_index = int(np.argmax(r["scores"][:5]))
                matrix, _, _ = alignment_geometry(
                    r["keypoints"], r["scores"], head_index, size, alignment
                )
            except ValueError:
                continue
            view_size = [max(16, round(384 * size[0] / size[1])), 384]
            matrix = np.diag(np.array(view_size) / np.array(size)) @ matrix
            request["records"].append(
                {
                    **item,
                    "view": "aligned",
                    "aligned_matrix": matrix.tolist(),
                    "view_size": view_size,
                }
            )
    request_path, response_path = (
        Path(out) / "relevance-request.json",
        Path(out) / "relevance-worker.json",
    )
    request_path.write_text(json.dumps(request))
    try:
        subprocess.run(
            [
                str(python or sys.executable),
                str(Path(__file__).with_name("relevance.py")),
                str(request_path),
                str(response_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "SigLIP2 worker failed; install .[relevance] or choose --relevance-python with torch/transformers"
        ) from exc
    response = json.loads(response_path.read_text())
    by_file = {}
    for r in response["records"]:
        by_file.setdefault(r["file"], []).append(r)
    response["records"] = [
        {
            "file": name,
            "stamp": views[0]["stamp"],
            "score": min(v["score"] for v in views),
            "views": views,
        }
        for name, views in by_file.items()
    ]
    response["view_method"] = (
        "minimum of full image and aligned viewport scores (before mirroring, tone and zoom)"
    )
    response["alignment_method"] = alignment
    response["size"] = list(size)
    return save(
        out,
        "relevance",
        {
            **response,
            "inputs": inputs(out, "detect"),
            "seconds": time.monotonic() - start,
        },
    )


def select(
    out,
    confidence=0.35,
    cluster="all",
    cluster_threshold=0.55,
    max_images=None,
    min_relevance=None,
    min_pose_conf=None,
    min_person_area=0.0,
    max_head_error=None,
    max_upscale=None,
    size=(1080, 1920),
    alignment="similarity",
    duration=None,
    speed_curve=None,
    fps=30,
    hold=6,
):
    start = time.monotonic()
    detected = load(out, "detect")
    if not detected.get("complete"):
        raise ValueError("Detection was interrupted; rerun detect before select")
    for name, value in [
        ("min-relevance", min_relevance),
        ("min-pose-conf", min_pose_conf),
        ("min-person-area", min_person_area),
    ]:
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"--{name} must be in [0,1]")
    for name, value in [
        ("max-head-error", max_head_error),
        ("max-upscale", max_upscale),
    ]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"--{name} must be finite and nonnegative")
    use_relevance = min_relevance is not None
    relevance_by_file = (
        {r["file"]: r for r in load(out, "relevance")["records"]}
        if use_relevance
        else {}
    )
    if use_relevance:
        relevance_data = load(out, "relevance")
        if relevance_data.get(
            "alignment_method", alignment
        ) != alignment or relevance_data.get("size", list(size)) != list(size):
            raise ValueError(
                "Relevance viewport settings differ; rerun relevance with matching --size and --alignment"
            )
    quality_enabled = (
        use_relevance
        or min_pose_conf is not None
        or min_person_area > 0
        or max_head_error is not None
        or max_upscale is not None
        or alignment != "similarity"
    )
    inferred_cap = automatic_count(duration, speed_curve, fps, hold)
    cap = max_images if max_images is not None else inferred_cap
    usable, dropped = [], []
    for record in detected["records"]:
        reason = None
        if record.get("error"):
            reason = "read_error"
        elif not record.get("people"):
            reason = "no_pose"
        else:
            p, s = np.asarray(record["keypoints"]), np.asarray(record["scores"])
            if s[:5].max() < confidence:
                reason = "low_head_confidence"
            elif s[[5, 6]].min() < confidence:
                reason = "low_shoulder_confidence"
            elif s[[11, 12]].min() < confidence:
                reason = "low_hip_confidence"
            else:
                # The highest-confidence facial point serves as a head anchor.
                head = int(np.argmax(s[:5]))
                record = dict(record, head_index=head)
                shoulder = p[[5, 6]].mean(0)
                hip = p[[11, 12]].mean(0)
                w, h = record["size"]
                if np.linalg.norm(shoulder - hip) < 0.04 * min(w, h):
                    reason = "degenerate_torso"
                elif np.any(p[[head, 5, 6, 11, 12]] < 0) or np.any(
                    p[[head, 5, 6, 11, 12]] >= [w, h]
                ):
                    reason = "anchor_outside_image"
            if reason is None and quality_enabled:
                record = dict(record)
                _, _, geometry = alignment_geometry(
                    p, s, record["head_index"], size, alignment
                )
                quality = {
                    **geometry,
                    "pose_confidence": pose_confidence(record),
                    "person_area": person_area(record),
                    "relevance": 1.0,
                }
                if use_relevance:
                    scored = relevance_by_file.get(record["file"])
                    if scored is None or scored["stamp"] != record["stamp"]:
                        raise ValueError(
                            "Missing/stale relevance score; rerun relevance"
                        )
                    quality["relevance"] = scored["score"]
                quality["rank_score"] = (
                    quality["relevance"]
                    * quality["pose_confidence"]
                    / (1 + quality["alignment_error_px"] / (0.02 * size[1]))
                )
                record["quality"] = quality
                if quality["relevance"] < (min_relevance or 0):
                    reason = "low_relevance"
                elif quality["person_area"] < min_person_area:
                    reason = "small_person_box"
                elif quality["pose_confidence"] < (min_pose_conf or 0):
                    reason = "low_pose_confidence"
                elif (
                    max_head_error is not None
                    and quality["head_error_px"] > max_head_error
                ):
                    reason = "head_alignment_error"
                elif (
                    max_upscale is not None
                    and (quality["scale"] - 1) * 100 > max_upscale
                ):
                    reason = "excessive_upscale"
        if reason:
            dropped.append(
                {
                    "file": record["file"],
                    "reason": reason,
                    **({"quality": record["quality"]} if "quality" in record else {}),
                }
            )
        else:
            usable.append(record)
    usable_count = len(usable)
    cluster_counts = {}
    if cluster == "largest" and len(usable) > 1:
        distances = distance_matrix(usable)
        labels = fcluster(
            linkage(squareform(distances, checks=False), method="average"),
            cluster_threshold,
            criterion="distance",
        )
        cluster_counts = dict(Counter(map(int, labels)))
        largest = max(cluster_counts, key=lambda k: (cluster_counts[k], -k))
        selected = []
        for r, label in zip(usable, labels):
            if label == largest:
                selected.append(r)
            else:
                dropped.append({"file": r["file"], "reason": "outside_largest_cluster"})
        usable = selected
    if cap is not None and len(usable) > cap:
        # Quality first, then restore stable filename order before path search.
        ranked = sorted(
            usable,
            key=lambda r: (
                -r["quality"]["rank_score"]
                if quality_enabled
                else -float(
                    np.mean(np.asarray(r["scores"])[[r["head_index"], 5, 6, 11, 12]])
                )
            ),
        )
        dropped.extend(
            {"file": r["file"], "reason": "max_images"} for r in ranked[cap:]
        )
        usable = sorted(ranked[:cap], key=lambda r: r["file"])
    if not usable:
        raise ValueError("No usable poses. Inspect detect.json or lower --confidence")
    return save(
        out,
        "select",
        {
            "inputs": inputs(out, "detect", *(["relevance"] if use_relevance else [])),
            "records": usable,
            "dropped": dropped,
            "usable_before_cluster_and_limit": usable_count,
            "cluster_counts": cluster_counts,
            "confidence": confidence,
            "cluster": cluster,
            "cluster_threshold": cluster_threshold,
            "quality_enabled": quality_enabled,
            "quality_filters": {
                "min_relevance": min_relevance,
                "min_pose_conf": min_pose_conf,
                "min_person_area": min_person_area,
                "max_head_error": max_head_error,
                "max_upscale_percent": max_upscale,
            },
            "alignment_method": alignment,
            "quality_size": list(size),
            "max_images": cap,
            "max_images_source": "explicit"
            if max_images is not None
            else "duration_and_curve"
            if inferred_cap
            else None,
            "seconds": time.monotonic() - start,
        },
    )


def align(out, size=(1080, 1920), mirror="off", alignment=None):
    start = time.monotonic()
    selected, detected = load(out, "select"), load(out, "detect")
    alignment = alignment or selected.get("alignment_method", "similarity")
    if selected.get("quality_enabled") and (
        selected.get("quality_size") != list(size)
        or alignment != selected.get("alignment_method")
    ):
        raise ValueError(
            "Alignment settings differ from quality gates; rerun select with matching --size and --alignment"
        )
    frames = Path(out) / "aligned"
    frames.mkdir(exist_ok=True)
    records = []
    target = target_torso(size)
    majority, cues = mirror_plan(selected["records"], mirror)
    for i, (r, cue) in enumerate(zip(selected["records"], cues)):
        source = source_path(r, detected)
        stat = source.stat()
        if [stat.st_size, stat.st_mtime_ns] != r["stamp"]:
            raise ValueError(f"Source changed: {r['file']}; rerun detect")
        image = read_image(source)
        p, scores = np.asarray(r["keypoints"]), np.asarray(r["scores"])
        head_index = r["head_index"]
        reflection = np.array([[-1.0, 0, image.shape[1] - 1], [0, 1.0, 0]])
        if cue["flipped"]:
            p, scores = reflect_pose(p, scores, image.shape[1])
            image = cv2.flip(image, 1)
            head_index = int(SWAP[head_index])
        matrix, points, geometry = alignment_geometry(
            p, scores, head_index, size, alignment
        )
        # Fill only pixels outside the transformed source; preserve exact alignment.
        foreground = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LANCZOS4)
        mask = cv2.warpAffine(np.full(image.shape[:2], 255, np.uint8), matrix, size)
        missing = mask < 254
        if missing.any():
            background = cv2.warpAffine(
                image, matrix, size, borderMode=cv2.BORDER_REPLICATE
            )
            sigma = max(size) / 60
            background = cv2.GaussianBlur(background, (0, 0), sigma)
            alpha = mask.astype(np.float32)[:, :, None] / 255
            foreground = (foreground * alpha + background * (1 - alpha)).astype(
                np.uint8
            )
        name = f"aligned/{i:05d}.jpg"
        if not cv2.imwrite(
            str(Path(out) / name), foreground, [cv2.IMWRITE_JPEG_QUALITY, 94]
        ):
            raise OSError(f"Could not write {name}")
        records.append(
            {
                **geometry,
                "file": r["file"],
                "frame": name,
                "keypoints": points.tolist(),
                "scores": scores.tolist(),
                "matrix": (
                    np.vstack([matrix, [0, 0, 1]]) @ np.vstack([reflection, [0, 0, 1]])
                )[:2].tolist()
                if cue["flipped"]
                else matrix.tolist(),
                "landmark_permutation": SWAP.tolist()
                if cue["flipped"]
                else list(range(17)),
                "mirror": cue,
                "flipped": cue["flipped"],
                "material_features": material_features(foreground).tolist(),
                "scale": float(np.linalg.norm(matrix[:, 0])),
                "rotation_degrees": float(
                    np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
                ),
                "fill_fraction": float(missing.mean()),
                "head_index": head_index,
                "quality": r.get("quality"),
            }
        )
    return save(
        out,
        "align",
        {
            "inputs": inputs(out, "select"),
            "size": list(size),
            "target": target.tolist(),
            "alignment_method": alignment,
            "mirror": mirror,
            "majority_handedness": majority,
            "flip_count": sum(c["flipped"] for c in cues),
            "records": records,
            "fill_count": sum(r["fill_fraction"] > 0 for r in records),
            "mean_alignment_error_px": float(
                np.mean([r["alignment_error_px"] for r in records])
            ),
            "seconds": time.monotonic() - start,
        },
    )


def order(
    out,
    start_image=None,
    hook=0,
    method="path",
    cycles=1,
    group_by_material=False,
    phase_bins=3,
):
    start = time.monotonic()
    selected = load(out, "select")
    uses_align = (Path(out) / "align.json").exists()
    aligned = load(out, "align") if uses_align else None
    records = aligned["records"] if uses_align else selected["records"]
    start_index = None
    if start_image:
        matches = [
            i
            for i, r in enumerate(records)
            if r["file"] == start_image or r["file"] == Path(start_image).name
        ]
        if not matches:
            raise ValueError("--start image is not in the selection")
        start_index = matches[0]
    distances = distance_matrix(records)
    if cycles != 1 and method != "phase":
        raise ValueError("--cycles requires --order phase")
    groups = None
    if group_by_material:
        if not uses_align:
            raise ValueError("--group-by-material requires align first")
        groups = material_labels([r["material_features"] for r in records])
    details = None
    if method == "phase":
        wrist = 9 if not aligned or aligned.get("majority_handedness", 1) == 1 else 10
        path, details = phase_path(
            records, distances, wrist, cycles, hook, start_index, groups, phase_bins
        )
    elif groups is not None:
        path = []
        group_order = sorted(set(groups))
        if start_index is not None:
            group_order.remove(groups[start_index])
            group_order.insert(0, groups[start_index])
        for group in group_order:
            members = np.flatnonzero(groups == group).tolist()
            local_start = members.index(start_index) if start_index in members else None
            if path:
                local_start = int(np.argmin(distances[path[-1], members]))
            path.extend(
                members[i]
                for i in order_path(
                    distances[np.ix_(members, members)], local_start, hook
                )
            )
    else:
        path = order_path(distances, start_index, hook)
    random_path = np.random.default_rng(0).permutation(len(records)).tolist()
    return save(
        out,
        "order",
        {
            "inputs": inputs(out, "select", *(["align"] if uses_align else [])),
            "files": [records[i]["file"] for i in path],
            "method": method,
            "cycles": cycles,
            "phase_bins": phase_bins if method == "phase" else None,
            "group_by_material": group_by_material,
            "poses": [
                {
                    "file": records[i]["file"],
                    "flipped": records[i].get("flipped", False),
                    **(details[i] if details else {}),
                }
                for i in path
            ],
            "continuity": {
                "shown": continuity_metrics(path, distances),
                "random": continuity_metrics(random_path, distances),
            },
            "path_length": path_length(path, distances),
            "random_path_length": path_length(random_path, distances),
            "start": start_image,
            "hook": hook,
            "seconds": time.monotonic() - start,
        },
    )


def ordered_frames(out):
    aligned, ordered = load(out, "align"), load(out, "order")
    by_name = {r["file"]: r for r in aligned["records"]}
    if set(by_name) != set(ordered["files"]):
        raise ValueError(
            "Alignment and ordering selections differ; rerun align and order"
        )
    return aligned, [by_name[name] for name in ordered["files"]]


def render(
    out,
    mode="cut",
    fps=30,
    hold=6,
    transition=4,
    audio=None,
    audio_start=0,
    speed_curve=None,
    beats=None,
    snap_beats=False,
    beat_tolerance=0.06,
    zoom="1:1",
    normalize_tone=False,
    duration=None,
):
    start = time.monotonic()
    aligned, records = ordered_frames(out)
    w, h = aligned["size"]
    if mode != "cut" and (speed_curve or beats or snap_beats or duration is not None):
        raise ValueError(
            "Variable timing and beat snapping currently require --mode cut"
        )
    zoom_values = parse_zoom(zoom)
    holds, timing = cut_timing(
        len(records), fps, hold, speed_curve, duration, tolerance=beat_tolerance
    )
    beat_values = None
    beat_source = None
    if beats:
        beat_values = read_beats(Path(beats), audio_start)
        beat_source = "json"
    elif snap_beats or (audio and speed_curve):
        if not audio:
            raise ValueError("--snap-beats requires --audio or --beats")
        beat_values = audio_onsets(audio, audio_start, sum(holds) / fps)
        beat_source = "spectral_flux_onsets"
    if beat_values is not None:
        holds, timing = cut_timing(
            len(records), fps, hold, speed_curve, duration, beat_values, beat_tolerance
        )
    total = sum(holds) + ((len(records) - 1) * transition if mode == "morph" else 0)
    tone = ToneNormalizer()
    center = np.mean(np.asarray(aligned["target"])[1:], axis=0)
    video = Path(out) / f"{mode}.mp4"
    partial = video.with_name(f".{mode}.partial.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "2",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio:
        command += [
            "-ss",
            str(audio_start),
            "-i",
            str(Path(audio).resolve()),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-af",
            "apad",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    command += [
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        (
            "scale=in_range=full:out_range=tv:out_color_matrix=bt709,"
            "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
        ),
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-t",
        str(total / fps),
        "-movflags",
        "+faststart",
        str(partial),
    ]
    stderr_file = Path(out) / f"{mode}-ffmpeg.log"
    written = 0
    try:
        with stderr_file.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                image = tone.apply(
                    read_image(Path(out) / records[0]["frame"]), normalize_tone
                )
                for i, record in enumerate(records):
                    for _ in range(holds[i]):
                        process.stdin.write(
                            camera_frame(
                                image, written / max(total - 1, 1), zoom_values, center
                            ).tobytes()
                        )
                        written += 1
                    if i + 1 < len(records):
                        next_record = records[i + 1]
                        nxt = tone.apply(
                            read_image(Path(out) / next_record["frame"]), normalize_tone
                        )
                        if mode == "morph":
                            for frame in morph_frames(
                                image,
                                nxt,
                                np.array(record["keypoints"]),
                                np.array(next_record["keypoints"]),
                                np.array(record["scores"]),
                                np.array(next_record["scores"]),
                                transition,
                            ):
                                process.stdin.write(
                                    camera_frame(
                                        frame,
                                        written / max(total - 1, 1),
                                        zoom_values,
                                        center,
                                    ).tobytes()
                                )
                                written += 1
                        image = nxt
                    if (i + 1) % 10 == 0 or i + 1 == len(records):
                        print(
                            f"render {mode} {i + 1}/{len(records)} ({written}/{total} frames)",
                            flush=True,
                        )
                process.stdin.close()
                code = process.wait()
            except BaseException:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                process.terminate()
                process.wait()
                raise
        if code:
            raise ValueError(f"ffmpeg failed ({code}); see {stderr_file}")
        partial.replace(video)
    except BrokenPipeError as exc:
        raise ValueError(f"ffmpeg closed input; see {stderr_file}") from exc
    finally:
        partial.unlink(missing_ok=True)
    data = {
        "inputs": inputs(out, "align", "order"),
        "video": video.name,
        "mode": mode,
        "fps": fps,
        "hold": hold,
        "transition": transition,
        "frames": written,
        "duration_seconds": written / fps,
        "encoding": {
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "color_space": "bt709",
            "color_range": "tv",
            "threads": 2,
            "crf": 19,
        },
        "audio": str(audio) if audio else None,
        "audio_start": audio_start,
        "timing": {
            **timing,
            "holds": holds,
            "beat_source": beat_source,
            "beats": beat_values,
        },
        "zoom": list(zoom_values),
        "zoom_center": center.tolist(),
        "normalize_tone": normalize_tone,
        "tone": tone.report(),
        "seconds": time.monotonic() - start,
    }
    save(out, f"render-{mode}", data)
    return save(out, "render", data)


def compare_baseline(out, records, size, baseline_order):
    baseline_path = Path(baseline_order)
    baseline = json.loads(baseline_path.read_text())
    detected = load(out, "detect")
    selected = load(out, "select")
    by_original = {r["file"]: r for r in selected["records"]}
    aliases = {}
    old_select_path = baseline_path.parent / "select.json"
    old_detect_path = baseline_path.parent / "detect.json"
    if old_select_path.exists() and old_detect_path.exists():
        old_selected = json.loads(old_select_path.read_text())
        old_detect = json.loads(old_detect_path.read_text())
        by_source = {
            str(source_path(r, detected).resolve()): r["file"]
            for r in selected["records"]
        }
        for r in old_selected["records"]:
            path = str(source_path(r, old_detect).resolve())
            if path in by_source:
                aliases[r["file"]] = by_source[path]
    old_files = [aliases.get(name, name) for name in baseline["files"]]
    current = {r["file"]: r for r in records}
    common = [name for name in old_files if name in current]
    if len(common) < 2:
        raise ValueError("Baseline order has fewer than two shared selected images")
    old_align_path = baseline_path.parent / "align.json"
    old_aligned = (
        json.loads(old_align_path.read_text()) if old_align_path.exists() else None
    )
    if old_aligned:
        old_by_name = {
            aliases.get(r["file"], r["file"]): r for r in old_aligned["records"]
        }
        before_records = [old_by_name[name] for name in common]
    else:
        before_records = [by_original[name] for name in common]
    after_records = [current[name] for name in common]
    index = {name: i for i, name in enumerate(common)}
    after_path = [index[r["file"]] for r in records if r["file"] in index]
    result = {
        "shared_images": len(common),
        "baseline_images": len(old_files),
        "baseline_order": str(baseline_path.resolve()),
        "scope": "shared selected images; actual canonical aligned poses when baseline alignment exists; intervening new-only images skipped",
        "before": continuity_metrics(
            list(range(len(common))), distance_matrix(before_records)
        ),
        "after": continuity_metrics(after_path, distance_matrix(after_records)),
    }
    if old_aligned:
        original_by_name = {r["file"]: r for r in old_aligned["records"]}
        old_shown = [original_by_name[name] for name in baseline["files"]]
        result["full_before_alignment"] = alignment_summary(
            old_shown, old_aligned["size"], size
        )
        result["full_after_alignment"] = alignment_summary(records, size)
        result["full_before_continuity"] = continuity_metrics(
            list(range(len(old_shown))), distance_matrix(old_shown)
        )
    return result


def report(out, baseline_order=None):
    start = time.monotonic()
    aligned, records = ordered_frames(out)
    detected, selected, ordered = (
        load(out, "detect"),
        load(out, "select"),
        load(out, "order"),
    )

    continuity = dict(ordered.get("continuity", {}))
    if baseline_order:
        paths = (
            baseline_order
            if isinstance(baseline_order, (list, tuple))
            else [baseline_order]
        )
        comparisons = {
            Path(path).parent.name: compare_baseline(
                out, records, aligned["size"], path
            )
            for path in paths
        }
        continuity["baselines"] = comparisons
        continuity["v0_comparison"] = next(iter(comparisons.values()))
    tw, th, columns = 144, 256, 8
    sheet = Image.new(
        "RGB",
        (columns * tw, ((len(records) + columns - 1) // columns) * (th + 30)),
        "#151515",
    )
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        image = read_image(Path(out) / record["frame"])
        image = cv2.resize(image, (tw, th))
        p = np.asarray(record["keypoints"]) * [
            tw / aligned["size"][0],
            th / aligned["size"][1],
        ]
        scores = np.asarray(record["scores"])
        for a, b in EDGES:
            if min(scores[a], scores[b]) >= 0.3:
                cv2.line(
                    image,
                    tuple(np.round(p[a]).astype(int)),
                    tuple(np.round(p[b]).astype(int)),
                    (30, 255, 100),
                    1,
                    cv2.LINE_AA,
                )
        for point, score in zip(p, scores):
            if score >= 0.3:
                cv2.circle(
                    image, tuple(np.round(point).astype(int)), 2, (40, 80, 255), -1
                )
        x, y = i % columns * tw, i // columns * (th + 30)
        sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, y))
        draw.text(
            (x + 4, y + th + 2), f"{i + 1:03d} {record['file'][:19]}", fill="white"
        )
    sheet.save(Path(out) / "contact-sheet.jpg", quality=92)
    renders = {}
    for mode in ["cut", "morph"]:
        if (Path(out) / f"render-{mode}.json").exists():
            try:
                renders[mode] = load(out, f"render-{mode}")
            except ValueError:
                pass
    return save(
        out,
        "report",
        {
            "inputs": inputs(out, "detect", "select", "align", "order"),
            "input_count": len(detected["records"]),
            "usable_pose_count": selected["usable_before_cluster_and_limit"],
            "selected_count": len(records),
            "dropped_count": len(selected["dropped"]),
            "drop_reasons": dict(Counter(r["reason"] for r in selected["dropped"])),
            "full_image_fallback_count": sum(
                r.get("box_source") == "full_image_fallback"
                for r in detected["records"]
            ),
            "fill_count": aligned["fill_count"],
            "mean_fill_fraction": float(np.mean([r["fill_fraction"] for r in records])),
            "mean_alignment_error_px": aligned["mean_alignment_error_px"],
            "path_length": ordered["path_length"],
            "random_path_length": ordered["random_path_length"],
            "continuity": continuity,
            "alignment_consistency": alignment_summary(records, aligned["size"]),
            "rendered_alignment_consistency": rendered_alignment_summary(
                records, renders["cut"]
            )
            if "cut" in renders
            else None,
            "quality_filters": selected.get("quality_filters", {}),
            "filter_drop_counts": {
                reason: sum(r["reason"] == reason for r in selected["dropped"])
                for reason in [
                    "read_error",
                    "no_pose",
                    "low_head_confidence",
                    "low_shoulder_confidence",
                    "low_hip_confidence",
                    "degenerate_torso",
                    "anchor_outside_image",
                    "low_relevance",
                    "small_person_box",
                    "low_pose_confidence",
                    "head_alignment_error",
                    "excessive_upscale",
                    "outside_largest_cluster",
                    "max_images",
                ]
            },
            "flip_count": aligned.get("flip_count", 0),
            "videos": renders,
            "contact_sheet": "contact-sheet.jpg",
            "stage_seconds": {
                s: load(out, s)["seconds"]
                for s in ["detect", "select", "align", "order"]
            },
            "seconds": time.monotonic() - start,
        },
    )


def evaluate(
    out,
    baseline_out=None,
    sample_size=30,
    seed=20261108,
    target_duration=35.2,
    seconds_per_cut=0.2,
):
    from .evaluation import audit_sheets, detection_summary

    start = time.monotonic()
    if (
        not np.isfinite(target_duration)
        or target_duration <= 0
        or not np.isfinite(seconds_per_cut)
        or seconds_per_cut <= 0
    ):
        raise ValueError(
            "Evaluation target duration and seconds per cut must be finite and positive"
        )
    required = int(np.ceil(target_duration / seconds_per_cut - 1e-9))
    detected = load(out, "detect")
    if not detected.get("complete"):
        raise ValueError("Detection was interrupted; rerun detect before evaluate")
    baselines = [
        (Path(path).name, load(path, "detect")) for path in (baseline_out or [])
    ]
    reference_hashes = {
        Path(path).name: digest(Path(path) / "detect.json")
        for path in (baseline_out or [])
    }
    if any(not d.get("complete") for _, d in baselines):
        raise ValueError("Reference detection is incomplete")
    audit = audit_sheets(out, detected, baselines, sample_size, seed)
    data = {
        "inputs": inputs(out, "detect"),
        "detection": detection_summary(detected),
        "reference_detection": {name: detection_summary(d) for name, d in baselines},
        "reference_detect_sha256": reference_hashes,
        "random_audit": audit,
        "target_duration_seconds": target_duration,
        "seconds_per_cut": seconds_per_cut,
        "required_cut_count": required,
        "seconds": time.monotonic() - start,
    }
    if (Path(out) / "report.json").exists():
        result = load(out, "report")
        selected = load(out, "select")
        data["inputs"].update(inputs(out, "report"))
        data["quality_filter_pass_count"] = result["selected_count"]
        data["quality_gate_pass_before_cap"] = selected[
            "usable_before_cluster_and_limit"
        ]
        data["shortfall_from_target"] = max(0, required - result["selected_count"])
        data["filter_drop_counts"] = result.get("filter_drop_counts", {})
        data["continuity"] = result["continuity"]
        data["alignment_consistency"] = result.get("alignment_consistency")
    review_path = Path(out) / "qa" / "visual-review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if (
            review.get("detect_sha256") == digest(Path(out) / "detect.json")
            and review.get("reference_detect_sha256") == reference_hashes
            and review.get("sample_files") == [r["file"] for r in audit["samples"]]
        ):
            data["visual_review"] = review
            data["visual_review_status"] = "notes_match_current_detection_and_sample"
        else:
            data["visual_review_status"] = "stale_notes_not_used"
    return save(out, "evaluation", data)
