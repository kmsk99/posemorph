"""Detection diagnostics and reproducible, static visual-audit sheets."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .geometry import EDGES
from .quality import pose_confidence


def distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return None
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "fraction_ge_020": float(np.mean(values >= 0.2)),
        "fraction_ge_035": float(np.mean(values >= 0.35)),
    }


def detection_summary(detected):
    records = detected["records"]
    posed = [r for r in records if r.get("people") and not r.get("error")]
    groups = {
        "head_best": [],
        "shoulders_min": [],
        "hips_min": [],
        "wrists_min": [],
        "required_min": [],
    }
    for r in posed:
        scores = np.asarray(r["scores"])
        for name, value in [
            ("head_best", scores[:5].max()),
            ("shoulders_min", scores[[5, 6]].min()),
            ("hips_min", scores[[11, 12]].min()),
            ("wrists_min", scores[[9, 10]].min()),
            ("required_min", pose_confidence(r)),
        ]:
            groups[name].append(float(value))
    return {
        "input_count": len(records),
        "complete": detected.get("complete", False),
        "pose_backend": detected.get("pose_backend", "body7"),
        "inference_seconds": detected["seconds"],
        "error_count": sum(bool(r.get("error")) for r in records),
        "actual_box_count": sum(r.get("box_source") == "detector" for r in posed),
        "fallback_count": sum(
            r.get("box_source") == "full_image_fallback" for r in posed
        ),
        "pose_record_count": len(posed),
        "required_confidence_ge_035_count": sum(
            x >= 0.35 for x in groups["required_min"]
        ),
        "confidence": {k: distribution(v) for k, v in groups.items()},
        "score_caveat": "raw SimCC maxima clipped to [0,1], not calibrated probabilities or accuracy; scales can differ across checkpoints",
        "models": detected.get("models"),
    }


def audit_sheets(out, detected, baselines, sample_size=30, seed=20261108):
    out = Path(out)
    target = out / "qa"
    target.mkdir(exist_ok=True)
    records = detected["records"]
    count = min(sample_size, len(records))
    indices = (
        np.random.default_rng(seed).choice(len(records), count, replace=False).tolist()
    )

    def source(r, d):
        return (
            Path(r["source_path"])
            if "source_path" in r
            else Path(d["source"]) / r["file"]
        )

    lookups = [
        (name, {str(source(r, d).resolve()): r for r in d["records"]})
        for name, d in baselines
    ]
    lookups.append(
        (
            detected.get("pose_backend", "current"),
            {str(source(r, detected).resolve()): r for r in records},
        )
    )
    tw, th, label_height = 280, 360, 42
    cols = 1 + len(lookups)
    pages = []
    audit = []
    for offset in range(0, count, 5):
        members = indices[offset : offset + 5]
        sheet = Image.new(
            "RGB", (cols * tw, len(members) * (th + label_height)), "#171717"
        )
        draw = ImageDraw.Draw(sheet)
        for row, index in enumerate(members):
            r = records[index]
            path = source(r, detected)
            with Image.open(path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
            w, h = rgb.size
            factor = min(tw / w, th / h)
            iw, ih = round(w * factor), round(h * factor)
            ox, oy = (tw - iw) // 2, (th - ih) // 2
            base = Image.new("RGB", (tw, th), "#080808")
            base.paste(rgb.resize((iw, ih)), (ox, oy))
            y = row * (th + label_height)
            sheet.paste(base, (0, y))
            draw.text(
                (4, y + th + 2),
                f"#{offset + row + 1:02d} {r['file'][:31]}",
                fill="white",
            )
            item = {
                "number": offset + row + 1,
                "detect_index": index,
                "file": r["file"],
                "models": {},
            }
            for col, (name, lookup) in enumerate(lookups, 1):
                person = lookup.get(str(path.resolve()))
                pixels = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
                label = name
                if person and person.get("people") and not person.get("error"):
                    p = np.asarray(person["keypoints"]) * [iw / w, ih / h] + [ox, oy]
                    scores = np.asarray(person["scores"])
                    for a, b in EDGES:
                        if min(scores[a], scores[b]) >= 0.2:
                            cv2.line(
                                pixels,
                                tuple(np.round(p[a]).astype(int)),
                                tuple(np.round(p[b]).astype(int)),
                                (30, 255, 80),
                                1,
                                cv2.LINE_AA,
                            )
                    for point, score in zip(p, scores):
                        color = (
                            (40, 220, 80)
                            if score >= 0.35
                            else (0, 170, 255)
                            if score >= 0.2
                            else (30, 40, 240)
                        )
                        cv2.circle(
                            pixels, tuple(np.round(point).astype(int)), 2, color, -1
                        )
                    box = np.asarray(person["box"]).reshape(2, 2) * [iw / w, ih / h] + [
                        ox,
                        oy,
                    ]
                    cv2.rectangle(
                        pixels,
                        tuple(np.round(box[0]).astype(int)),
                        tuple(np.round(box[1]).astype(int)),
                        (255, 150, 30),
                        1,
                    )
                    confidence = pose_confidence(person)
                    label += (
                        f" min={confidence:.2f} {person.get('box_source', 'import')}"
                    )
                    item["models"][name] = {
                        "box_source": person.get("box_source"),
                        "required_confidence": confidence,
                    }
                else:
                    label += " no pose"
                sheet.paste(
                    Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)),
                    (col * tw, y),
                )
                draw.text((col * tw + 4, y + th + 2), label[:45], fill="white")
            audit.append(item)
        name = f"random-30-page-{len(pages) + 1:02d}.jpg"
        sheet.save(target / name, quality=94)
        pages.append(name)
    full = Image.new("RGB", (cols * tw, count * (th + label_height)), "#171717")
    y = 0
    for page in pages:
        with Image.open(target / page) as im:
            full.paste(im, (0, y))
            y += im.height
    full.save(out / "random-30-contact-sheet.jpg", quality=92)
    data = {
        "seed": seed,
        "sampling": "without replacement from all detect records, not selected poses",
        "count": count,
        "pages": [f"qa/{p}" for p in pages],
        "samples": audit,
    }
    (target / "random-30-sample.json").write_text(json.dumps(data, indent=2) + "\n")
    return data
