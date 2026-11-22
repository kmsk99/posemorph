"""Optional SigLIP2 worker; torch/transformers never enter the base process."""

import json
import math
import sys
from pathlib import Path

MODEL = "google/siglip2-so400m-patch14-384"
REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"


def read_prompts(path):
    prompts = json.loads(Path(path).read_text())
    if not isinstance(prompts, dict) or any(
        not isinstance(prompts.get(k), list)
        or not prompts[k]
        or any(not isinstance(p, str) or not p.strip() for p in prompts[k])
        for k in ["positive", "negative"]
    ):
        raise ValueError(
            "--relevance-prompts needs nonempty positive and negative text lists"
        )
    return {k: prompts[k] for k in ["positive", "negative"]}


def contrast_score(logits, positive_count):
    """Group maximum contrast: prompt count does not change group prior odds."""
    positive, negative = max(logits[:positive_count]), max(logits[positive_count:])
    margin = positive - negative
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, margin))))


def worker(request_path, response_path):
    import numpy as np
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoModel, AutoProcessor

    request = json.loads(Path(request_path).read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = (
        AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    prompts = request["prompts"]
    text = prompts["positive"] + prompts["negative"]
    cache_path = Path(response_path).with_suffix(".embeddings.npz")
    stamps = [
        json.dumps([r["path"], r["stamp"], r.get("aligned_matrix"), r.get("view_size")])
        for r in request["records"]
    ]
    cached = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                cache["stamps"].tolist() == stamps
                and str(cache["revision"]) == REVISION
            ):
                cached = cache["features"].copy()
    with torch.inference_mode():
        tokens = processor(
            text=text, padding="max_length", max_length=64, return_tensors="pt"
        ).to(device)
        features = model.get_text_features(**tokens)
        features = getattr(features, "pooler_output", features)
        text_features = features / features.norm(dim=-1, keepdim=True)
        image_features = []
        if cached is None:
            for offset in range(0, len(request["records"]), 4):
                images = []
                for record in request["records"][offset : offset + 4]:
                    path = Path(record["path"])
                    stat = path.stat()
                    if [stat.st_size, stat.st_mtime_ns] != record["stamp"]:
                        raise ValueError(
                            f"Source changed before relevance: {path.name}; rerun detect"
                        )
                    with Image.open(path) as im:
                        rgb = ImageOps.exif_transpose(im).convert("RGB")
                        if "aligned_matrix" in record:
                            matrix = np.vstack([record["aligned_matrix"], [0, 0, 1]])
                            inverse = np.linalg.inv(matrix)[:2].ravel().tolist()
                            rgb = rgb.transform(
                                tuple(record["view_size"]),
                                Image.Transform.AFFINE,
                                inverse,
                                Image.Resampling.BICUBIC,
                            )
                        images.append(rgb)
                pixels = processor(images=images, return_tensors="pt").to(device)
                pixels["pixel_values"] = pixels["pixel_values"].to(dtype)
                features = model.get_image_features(**pixels)
                features = getattr(features, "pooler_output", features)
                features = features / features.norm(dim=-1, keepdim=True)
                image_features.append(features.float().cpu().numpy())
                print(
                    f"relevance {min(offset + 4, len(request['records']))}/{len(request['records'])}",
                    flush=True,
                )
            cached = np.concatenate(image_features)
            np.savez_compressed(
                cache_path, stamps=np.array(stamps), features=cached, revision=REVISION
            )
        features = torch.as_tensor(cached, dtype=dtype, device=device)
        logits = (
            (features @ text_features.T * model.logit_scale.exp() + model.logit_bias)
            .float()
            .cpu()
            .numpy()
        )
    records = []
    for record, values in zip(request["records"], logits):
        scores = values.astype(float).tolist()
        positive_count = len(prompts["positive"])
        records.append(
            {
                "file": record["file"],
                "view": record.get("view", "full"),
                "stamp": record["stamp"],
                "logits": scores,
                "score": contrast_score(scores, positive_count),
                "positive_prompt": prompts["positive"][
                    int(np.argmax(values[:positive_count]))
                ],
                "negative_prompt": prompts["negative"][
                    int(np.argmax(values[positive_count:]))
                ],
            }
        )
    result = {
        "model": MODEL,
        "revision": REVISION,
        "license": "Apache-2.0",
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "method": "sigmoid(max_positive_logit - max_negative_logit)",
        "embedding_cache_reused": bool(image_features == []),
        "prompts": prompts,
        "records": records,
    }
    target = Path(response_path)
    partial = target.with_suffix(".tmp")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(target)


if __name__ == "__main__":
    worker(*sys.argv[1:])
