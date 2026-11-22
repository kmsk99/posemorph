# posemorph usage reference

Detailed options for each stage. Start with the [README](../README.md) for an overview.

## Continuity, rhythm and finishing

All finishing options are off by default: `--mirror off --order path`, fixed
`--hold`, `--zoom 1:1`, no tone matching or material grouping. An audio track
alone retains fixed timing. Existing cut/morph invocations continue to work.

```sh
nice -n 10 posemorph run ./images-a ./images-b --out ./out/demo-v1 \
  --confidence 0.20 --size 1080x1920 --mirror auto --order phase --hook 3 \
  --mode cut --fps 30 --duration 35.2 \
  --speed-curve '0:0.25,0.6:0.2,0.9:0.1,1:0.1' \
  --audio ./music.wav --beat-tolerance 0.045 --zoom 1:1.12 --normalize-tone

# Compare an existing order on the shared selected images:
posemorph report --out ./out/demo-v1 --baseline-order ./out/demo/order.json
```

| Option / stage | Behavior |
| --- | --- |
| `--mirror off\|auto` / align | Confidence-weighted majority of raised-wrist handedness, with a weak torso-relative face orientation fallback. Flip confident opposite poses; leave ambiguous poses unchanged. Reflect pixels and swap COCO left/right labels. Record cue, confidence, permutation and flips in alignment/ordering JSON. The stored matrix maps original source coordinates after the recorded landmark permutation. |
| `--order path\|phase` / order | `path` keeps the nearest-neighbour + 2-opt heuristic. `phase` derives a scalar from wrist height and upper-arm/forearm angles around the torso, then traverses raise → overhead peak → across/down strike. Three coarse phase bins by default, with pose-distance optimization inside each bin and across its boundaries. `--phase-bins N` allows finer bins at the cost of larger pose jumps. |
| `--cycles N` / order | Divide selected images across N phase trajectories; every selected image appears once. Sparse phases can leave incomplete cycles. Requires phase order. |
| `--hook N` / order | Choose up to N dramatic opening images from the nearest quartile around the starting pose/medoid, constrained to the first phase bin in phase mode. Never prepend the global distant outliers used in v0. `--start` pins a selected image; phase mode requires it in the first phase bin/material run. |
| `--group-by-material` / order | Cluster aligned Lab means/stds and edge density into up to three visual runs, then preserve phase order inside each run. These are appearance clusters, not semantic material labels. Each run can restart the trajectory and introduce a boundary jump. |
| `--speed-curve '0:.25,.6:.2,.9:.1,1:.1'` / render | Fraction of video time → seconds per image, linearly interpolated in time. Integrate reciprocal hold duration to place cuts. Points must increase from 0 to 1 with positive holds. |
| `--duration 35.2` / render | Rescale the relative speed curve to this duration; without a curve use uniform holds. Round to whole frames and preserve at least one frame per selected image. This can make absolute holds shorter than the supplied curve. |
| `--beats beats.json` / render | Snap internal cuts to nearby timestamps, within `--beat-tolerance` seconds (default 0.06). Accept `[0.5, 1.0]` or `{"beats": [0.5, 1.0]}`. Timestamps are seconds in the original audio; subtract `--audio-start` before snapping. Endpoints stay fixed, cuts remain distinct, and no image is duplicated or skipped. |
| `--audio track.wav --snap-beats` / render | Detect spectral-flux onset candidates with existing FFmpeg/NumPy/SciPy. Also enabled by `--audio` + `--speed-curve`; explicit beat JSON takes precedence. No added dependency or model. This detects attacks, including speech, not musical meter/downbeats. |
| `--zoom 1.0:1.12` / render | Interpolate zoom across every output frame about the fixed aligned torso target, including across cuts. Apply after alignment and tone matching. Factors must be in `[1,4]`. |
| `--normalize-tone` / render | Match Lab luminance mean/contrast to a slowly moving target (8% update per image), at 75% strength; clamp gain to 0.8–1.25 and mean adjustment to ±28 in 8-bit L. Preserve a/b chroma, subject to RGB gamut clipping. Log adjacent luminance/contrast jumps before and after. |

Variable timing/beat options currently apply to **cut only**. Camera and tone
options also work on morph frames. At 30 fps, aggressive curves or beat snapping
can yield one-frame images; reduce image count or increase duration if needed.

Multiple input folders are read without modifying them. Multi-folder IDs use
`01/filename.jpg`, `02/filename.jpg`, etc., avoiding filename collisions; use those
IDs in imported poses, person overrides and `--start`. Single-folder IDs stay
unchanged. Repeat folders in the same order when resuming detection.

Static art does not reveal motion direction or reliably identify a weapon hand.
The phase/handedness values are auditable geometric hypotheses; no weapon-line
detector or semantic person recognition is implemented. Reflections also reverse
inscriptions and scene composition. Phase order is not guaranteed to minimize
pose distance; inspect both the metrics and the encoded result.

`report.json` includes mean and p90 torso-normalized pose distance between
consecutive shown images and a seeded random baseline. `--baseline-order` also
compares the shared selection in its old and new order, using original and
reflected poses respectively. New-only images are skipped in that shared-subset
comparison; its jumps are therefore not the actual full video's jumps. Reports
print both comparisons. The contact sheet shows aligned skeleton thumbnails in
shown order, before camera/tone finishing; inspect encoded frames for final color.

## Scene relevance and consistent framing (v1.1)

SigLIP2 scoring is optional. Install `uv pip install -e '.[dev,relevance]'`, or
pass `--relevance-python /path/to/another/env/bin/python` to use an existing
PyTorch/Transformers environment without enlarging the base environment.
Model weights stay in the Hugging Face cache; see [licenses](models.md).

```sh
nice -n 10 posemorph run ./images-a ./images-b --out ./out/demo-v11 \
  --relevance-prompts examples/michael-prompts.json --min-relevance .8 \
  --confidence .2 --min-pose-conf .35 --min-person-area .12 \
  --alignment head-torso --max-head-error 75 --max-upscale 100 \
  --size 1080x1920 --mirror auto --order phase --hook 3 \
  --duration 35.2 --speed-curve '0:.25,.6:.2,.9:.1,1:.1' \
  --audio ./music.wav --beat-tolerance .045 --zoom 1:1.12 --normalize-tone

# Optional stage when detect.json already exists:
posemorph relevance --out ./out/demo-v11 \
  --relevance-prompts examples/michael-prompts.json --alignment head-torso

# More than one reference order is accepted:
posemorph report --out ./out/demo-v11 \
  --baseline-order ./out/demo-v0/order.json --baseline-order ./out/demo-v1/order.json
```

| Option / stage | Meaning |
| --- | --- |
| `--relevance-prompts prompts.json` / run, relevance | A JSON object with nonempty `positive` and `negative` text lists. For each image, score the original and a preview of the selected person's aligned viewport. Score each view as `sigmoid(max positive logit − max negative logit)` and use the lower view score. Save prompts, per-view logits, winning texts, model revision and stamps to `relevance.json`. Examples include both painting and sculpture descriptions and human-donor/painter/monk negatives. |
| `--min-relevance FLOAT` / run, select | Drop lower scores. `run` with prompts defaults to 0.5 unless specified; standalone `select` can reuse an existing relevance manifest. Scores depend on prompts and are not probabilities or a verified identification of Michael. |
| `--min-pose-conf FLOAT` / run, select | Minimum of the best facial-point confidence and both shoulders, hips **and wrists**. This is separate from the older `--confidence` torso/head gate. |
| `--min-person-area FLOAT` / run, select | Minimum chosen xyxy box area divided by image area. A full-image fallback is still a crop hypothesis, so its box area alone cannot validate the person. |
| `--alignment head-torso` / run, relevance, select, align | Use a confidence-weighted nose/eye center (fallback to best facial point), rotate the shoulder→hip axis upright, fix torso length, then translate the face center to the target. This preserves anatomy and fixes detected head-center/torso-scale variation; it cannot fix incorrect landmarks or head-size differences caused by perspective. Default remains `similarity`. |
| `--max-head-error PX` / run, select | Maximum post-alignment distance from the selected highest-confidence facial landmark to the head target. This landmark can be an eye/ear rather than the centered face point, making this a conservative face-offset gate. Record both the initial similarity error and final error. Values refer to `--size` output pixels. |
| `--max-upscale PERCENT` / run, select | Maximum enlargement over source pixels: 100 permits scale ≤2.0, 0 permits no enlargement. Uses the actual proposed alignment scale, not the image's total dimensions. |
| `--max-images N` / run, select | Explicit count cap. With quality filtering/stable alignment, choose by `relevance × minimum pose confidence / (1 + alignment RMS / (0.02 × output height))`; otherwise retain legacy confidence ranking. Without an explicit cap, `--duration` derives one by integrating the speed curve's reciprocal holds (or duration × fps / hold). Fewer passing images remain fewer; rejected images are never restored to reach the cap. |

Without these options, the original selection/alignment defaults remain. Quality
preview gates and relevance scoring must use the same `--size` and `--alignment`
as the final alignment; changing them requires rerunning those stages. No
material-specific or source-filename blacklist is used. Full-image scoring alone
can accept a good scene while the selected skeleton crops a different person;
viewport scoring reduces that failure but does not eliminate false matches.

`report.json` records exclusive first-failing filter counts (including zeros),
head-center cut-boundary mean/p90 jitter in pixels, head-position variance,
torso-length jitter/variance, and the alternate selected facial-landmark jitter.
The report also propagates those landmarks through holds and global zoom to
estimate consecutive **output-frame** and cut-boundary jitter. These are model
geometry diagnostics, not an independent measurement of encoded-image head
motion. Near-zero head-center and torso-length residuals in `head-torso` mode
are consequences of its hard constraints; inspect the encoded frames too.
Reference alignment metrics are expressed at the current output resolution.
Reference continuity comparisons now use the actual stored canonical aligned
poses, including any old mirrors, rather than reconstructing an unmirrored v1.

## Art-domain pose backend and evaluation

`--pose-backend` is accepted by `detect` and `run`:

- `body7` (default): historical YOLOX-s COCO + RTMPose-m Body7, including the
  old BGR pose preprocessing, to reproduce v0/v1/v1.1.
- `body7-rgb`: the same original checkpoints with the SDK-required RGB pose
  conversion corrected. This isolates a verified preprocessing bug from a
  change of model/domain training.
- `humanart`: experimental YOLOX-M + RTMPose-L trained on HumanArt+COCO, using
  official FP32 ONNX exports and correct RGB pose input. It runs on CPU with
  one ONNX thread and requires no MMPose/mmcv/PyTorch installation.

**HumanArt weight commercial scope is unresolved.** Code licenses do not settle
trained-weight and training-data permissions. This backend is explicitly opt-in,
weights stay outside Git, and it is not the permissive default. Exact artifacts,
SHA-256 hashes, ViTPose/RTMDet alternatives and sources are in [models.md](models.md).
Backend identity invalidates the detection cache when changed. Do not mix
cached pose records from different backends.

```sh
nice -n 10 posemorph run ./images-a ./images-b --out ./out/artpose \
  --pose-backend humanart --confidence .2 \
  --relevance-prompts examples/michael-prompts.json --min-relevance .8 \
  --min-pose-conf .35 --min-person-area .12 --alignment head-torso \
  --max-head-error 75 --max-upscale 100 --mirror auto --order phase \
  --duration 35.2 --speed-curve '0:.25,.6:.2,.9:.1,1:.1' \
  --audio ./music.wav --zoom 1:1.12 --normalize-tone

# Static diagnostics, sampling all inputs rather than just the selected frames:
nice -n 10 posemorph evaluate --out ./out/artpose \
  --baseline-out ./out/old --baseline-out ./out/rgb-control \
  --sample-size 30 --seed 20261108
```

`evaluation.json` separates actual detected boxes from full-frame fallbacks and
records head/shoulder/hip/wrist confidence distributions, quality counts, the
shortfall for a target duration/cut interval (defaults 35.2 s / 0.2 s = 176
cuts), and reference results. Confidence is not accuracy and
is not calibrated across checkpoints. The seeded static audit contains original
images and skeleton overlays from each backend: `random-30-contact-sheet.jpg`,
readable five-image pages in `qa/`, and a JSON sample index. Existing visual
review notes are included only when detection/reference hashes and sample IDs
match. This adds no human-review UI.

The local [v2 validation](validation.md) improved actual box returns from
181/398 to 396/398 and strict-filter output from 60 to 113 images. It still falls
63 images short of 176; a one-use 0.2-second cut is 22.6 seconds, not 35.2. More
confident detections alone do not prove better semantic subject selection.

## Pipeline and stage files

Every stage writes a schema-versioned JSON manifest into `--out`. Paths to
source images are stored locally in that output directory. Keep it private if
source paths are sensitive; output directories and all media are gitignored.

```sh
posemorph detect ./images --out ./out/demo
posemorph select --out ./out/demo --confidence 0.35 --cluster all --max-images 170
posemorph align --out ./out/demo --size 1080x1920
posemorph order --out ./out/demo --start first.jpg --hook 3
posemorph render --out ./out/demo --mode cut --hold 6
posemorph report --out ./out/demo
```

1. **detect.json**: EXIF-corrected image sizes, 17 COCO keypoints and confidence
   scores (raw SimCC maxima clipped to `[0,1]`, not probabilities), all person candidates, selected person index, xyxy bounding box and
   model URLs/hashes. Choose the largest central person by area and centrality.
   If no person box is detected, try the full image and label the box source
   `full_image_fallback`; this is a crop hypothesis, not a detected box.
   Hidden files (including macOS `._` resource forks) are ignored. Detection
   checkpoints every ten images; reruns reuse unchanged image size/mtime stamps.
**Optional relevance.json** sits between detection and selection when enabled.
It scores original/displayed views, records model provenance, and participates
in downstream manifest invalidation.

2. **select.json**: rejection reasons and selected records. Require one confident
   facial point (nose/eye/ear), both shoulders, and both hips; reject tiny torsos
   and anchors outside the image. `--cluster largest` keeps the largest
   average-linkage cluster (`--cluster-threshold 0.55`); `all` keeps every usable
   pose. Distances use torso-normalized, pairwise Procrustes rotation alignment,
   confidence weights, and extra torso/arm weight. `--max-images` keeps the
   highest-confidence usable poses before ordering in legacy mode; quality
   mode uses the product score described above.
3. **align.json**, **aligned/*.jpg**: least-squares similarity matrices, scale,
   rotation, transformed keypoints, RMS alignment errors in output pixels, and
   fill fractions. Head, shoulders, and hips aim at fixed frame coordinates.
   Similarity preserves anatomy, so different proportions cannot match all five
   anchors exactly. Crop to fill; pixels beyond the source boundary use blurred
   replicated edges. The report counts these frames. No letterbox bars are added.
4. **order.json**: nearest-neighbour path improved by open-path 2-opt, or
   phase-constrained trajectories; path length and a reproducible random
   baseline. See continuity options for the revised hook behavior.
5. **render-cut.json** / **render-morph.json**, **cut.mp4** / **morph.mp4**: frame
   count, timing, encoding settings and video duration. `render.json` also points
   to the latest render. Frames stream directly to FFmpeg; only aligned source
   frames are stored, avoiding thousands of duplicate output files.
6. **report.json**, **contact-sheet.jpg**: ordered thumbnails with skeletons,
   counts and rejection reasons, fill count, mean alignment error, timing and
   video metadata. Rerun report after rendering another mode.

Downstream manifests record upstream JSON hashes. Rerunning an upstream stage
invalidates dependent stages; the CLI explains which stage to rerun. Outputs
are atomically replaced so interrupted renders do not become finished videos.
Run `posemorph COMMAND --help` for stage-specific options.

## Choosing a person and importing poses

Inspect `detect.json`'s `people` list, then supply a JSON object mapping filenames
to zero-based candidate indices with `detect --person-overrides overrides.json`.
The override reuses cached detections. Rerun select/align/order/render/report.

For manual annotation, external detectors, or offline tests, pass `--poses poses.json`
to `detect` or `run`. It is a mapping from image filename to a list of people:

```json
{
  "example.jpg": [
    {"box": [0, 0, 100, 200], "keypoints": [[50, 30]], "scores": [0.9]}
  ]
}
```

The abbreviated example must be expanded to **17** `[x,y]` keypoints and **17**
scores in COCO order: nose, left/right eye, left/right ear, left/right shoulder,
left/right elbow, left/right wrist, left/right hip, left/right knee, left/right
ankle. Coordinates refer to the image after EXIF orientation correction. Missing
joints use finite placeholder coordinates with score zero; absent people use
an empty list. Scores must be in `[0,1]`.

## Modes

- **cut**: each image is held for `--hold` frames (default 6 at 30 fps = 0.2 s).
- **morph**: hold each image, then insert `--transition` intermediate frames
  (default 4). Interpolate confident common keypoints, piecewise-affine warp
  both images onto the intermediate pose using Delaunay triangles and eight
  border anchors, then cross-dissolve. Endpoints stay out of transition frames
  to avoid duplicate holds. Low-confidence or cropped-out joints are excluded.

Mixed cut/morph ranges and fixed text overlays are outside v0.

## Limitations in detail

- A confident skeleton can still belong to the wrong person, a demon, or a
  background figure. Prominence cannot identify a saint semantically; use person
  overrides and inspect the contact sheet.
- Stylized faces, wings, helmets, armour, occlusion, stained glass divisions and
  sculptures seen from the side can defeat photographic pose models. Confidence
  is not a calibrated correctness probability. This is a 2D art-domain baseline,
  not a HumanArt-trained model or segmentation model.
- A similarity transform aligns the torso, not every body part. Strongly different
  proportions and viewpoints leave residual jumps. It can crop limbs and wings.
  Wide artworks or close crops often need visible blurred edge extension.
- Morphs warp the entire artwork, including backgrounds. Triangle folds, ghosting,
  rubbery faces, dissolving weapons, and double limbs can occur. There is no depth,
  optical flow, person mask, occlusion model, or guarantee of nonfolding warps.
- Inputs are treated as sRGB after EXIF correction; embedded ICC profiles and HDR
  are not color-managed. Output encoding is BT.709 SDR.
- Pose ordering is heuristic and quadratic; intended for hundreds of images.
  Source fingerprints use size/mtime rather than content hashes. The detection
  cache also checks an inference revision; custom inference changes must bump it.

## Tests

Tests cover recovery of a known similarity transform, pose-distance invariance,
shorter pose paths, landmarks landing on intermediate warp targets, cut/morph
integration with synthetic stick figures, manifest invalidation and person
overrides. FFmpeg integration tests skip when FFmpeg is absent.
