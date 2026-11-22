# Local v0 validation — 2026-11-08

Validated on Apple Silicon macOS, Python 3.12.14, ONNX Runtime 1.31.1 and
FFmpeg 9.0.2. The private local input set contains 233 distinct St Michael
artworks, including paintings, sculptures and stained glass. No source media or
source-specific filesystem paths are distributed in this repository.

## Real-data result

| Measurement | Result |
| --- | --- |
| Input images inferred | 233; zero image/inference errors |
| Selection threshold | 0.20 for a facial anchor, both shoulders and both hips |
| Poses passing selection | 154 |
| Dropped | 79: head confidence 30, shoulder confidence 36, hip confidence 13 |
| Output | 720×1280, 30 fps, silent H.264, yuv420p, limited-range BT.709 |
| Cut | 924 frames, 30.800 s; hold 6 |
| Morph | 920 frames, 30.667 s; hold 2, transition 4 |
| Mean alignment RMS | 24.677 output pixels across five torso/head anchors |
| Frames requiring edge extension | 101/154; mean uncovered area across all frames 8.42% |
| Pose path length | 49.197, versus seeded random baseline 108.917 (54.8% shorter) |
| Full-image crop fallback | 132/233 inputs; 59/154 selected frames |

“Passing selection” means that the confidence/geometry rules passed; it is not a
human-verified accuracy count. The default threshold 0.35 retained 127 images;
0.20 was explicitly chosen for this art-domain validation to reach the target
30–40 second duration. `--cluster all --max-images 170` was used; the limit did
not remove any of the 154 passing images.

Measured stage wall times with already-downloaded weights and fresh inference:

| Stage | Seconds |
| --- | ---: |
| Detect all 233 images | 94.11 |
| Select | 0.01 |
| Align | 23.26 |
| Order | 0.72 |
| Cut render | 10.26 |
| Morph render | 132.41 |
| Final contact sheet/report | 1.10 |

The listed stages total approximately **4 min 22 s**, excluding model downloads,
command startup, initial experiments and visual review. Jobs used `nice -n 10`,
sequential one-thread ONNX CPU sessions, one OpenCV thread and FFmpeg with two
encoder threads. The two video renders were run sequentially.

## Visual review

The ordered contact sheet, six decoded cut frames, and twelve consecutive
decoded morph frames at 9.000–9.367 seconds were inspected. A two-second
morph excerpt was also extracted. The checks were against the encoded MP4s,
not just the pre-encoding source frames.

- Many frontal sculptures and stained-glass figures produce convincing torso
  alignment. Body placement is much steadier than unaligned source images.
- Stylized/side-facing sculptures and strongly bent bodies can have large
  residual alignment errors. Confidence filtering does not catch every wrong
  skeleton. Several low-quality outliers remain in the montage.
- On multi-panel stained glass, window dividers can distract the detector and
  low-confidence faces can force a full-image crop hypothesis. Similarity
  alignment rotates the architecture as well as the figure.
- Large altarpieces can select a central non-Michael figure. The prominence
  heuristic has no semantic identification; the person override hook is needed
  for those frames. The real run intentionally documents the automatic result.
- Blurred replicated edges are visible on narrow crops and heavily rotated
  frames; there are no added letterbox bars. Some source images contain their
  own blank backgrounds, which the tool does not remove.
- The sampled sculpture → stained glass → stained glass sequence keeps the
  torso roughly in place but shows ghosted faces, double arms, bending window
  outlines and cross-dissolving backgrounds. Morph is an experimental effect;
  cut is the cleaner v0 result for this heterogeneous collection.

## Automated checks

- Synthetic, network-free tests: 11 passed; one real-model test skipped by default.
- Including explicit real-model smoke test: 12 passed.
- `ruff check`, `ruff format --check`, `compileall`, and `git diff --check` passed.
- `uv sync --extra dev` and `uv build` succeeded; built wheel/source archive
  were inspected and contained no images, videos, weights or `out/` artifacts.
- Both full MP4s decoded to FFmpeg's null output without errors. FFprobe
  confirmed frame counts, dimensions, 30 fps, H.264/yuv420p and all BT.709 tags.
- An integration test verified audio offset, padding a short track to video
  length, and output color metadata.

Local outputs live in the gitignored `out/michael-v0/` directory: `cut.mp4`,
`morph.mp4`, `contact-sheet.jpg`, all stage JSON files, and `qa/` with decoded
frame sheets, the two-second excerpt and FFprobe output.

## Local v1 validation — 2026-11-08

The second pass reads both local artwork collections (233 + 165 = 398 files)
without writing to either source directory. Inference uses the existing cached
YOLOX/RTMPose weights; no new model or dependency was added. The spectral-flux
onset implementation uses existing NumPy/SciPy (BSD-3-Clause) and external FFmpeg.
This is an automatic baseline, not a manually curated final edit.

Settings: `--confidence .20 --size 1080x1920 --mirror auto --order phase
--phase-bins 3 --cycles 1 --hook 3`, cut at 30 fps, `--duration 35.2`,
`--speed-curve '0:.25,.6:.2,.9:.1,1:.1' --beat-tolerance .045 --zoom 1:1.12
--normalize-tone`. Material grouping is implemented and synthetically tested,
but disabled for this run to preserve a single phase trajectory.

The audio track was extracted losslessly to a local PCM WAV from the supplied
35.2-second master and muxed as AAC. No offset or time stretch was applied.
The two highest one-second source RMS windows are 33–34 and 34–35 seconds,
so the loudest late passage remains near the end. RMS is not a semantic music
climax detector, and the master track is not separated into speech and music.

| Measurement | v1 result |
| --- | ---: |
| Inputs inferred / read or inference errors | 398 / 0 |
| Usable and shown images | 267, each exactly once; no count cap |
| Rejections | 131: head 58, shoulder 52, hip 20, anchor outside source 1 |
| Mirrored frames | 49; 189 wrist cues, 6 weak torso/profile cues, 72 ambiguous poses left unchanged |
| Output | 1080×1920, 30 fps, 1,056 frames, 35.200 seconds |
| Video/audio | H.264 yuv420p limited-range BT.709 / AAC 48 kHz stereo, both 35.200 seconds |
| Onset candidates / snapped internal cuts | 48 / 23 of 266, tolerance 45 ms |
| Per-image hold range after frame rounding and snaps | 1–7 frames (0.033–0.233 s) |
| First / final third mean hold | 0.1753 / 0.0820 seconds |
| Global zoom | 1.00 → 1.12 about the fixed aligned torso target |
| Mean torso alignment RMS | 38.375 output pixels |
| Edge extension | 167/267 frames; mean missing area 7.77% |

The requested speed curve is relative after `--duration` scaling: showing all
267 usable images in 35.2 seconds requires shorter absolute holds than the
example curve. Onset snapping can produce one-frame images. No onset is forced
when it would collapse, repeat or skip an image. This is an attack detector,
not a beat/downbeat tracker, and only 23 cuts actually snapped.

### Continuity and tone before/after

Distances are dimensionless confidence-weighted, torso-normalized Procrustes
pose distances; lower is better. They do not measure semantic correctness,
weapon continuity, background changes or color flicker.

| Comparison and scope | Before mean / p90 | After mean / p90 |
| --- | ---: | ---: |
| Seeded random vs final phase order, same 267 canonical poses | 0.705151 / 1.108184 | 0.311323 / 0.480760 |
| Existing v0 order vs v1 order, 154 shared images only | 0.321548 / 0.519133 | 0.349379 / 0.538238 |
| Initial 12-bin phase implementation vs final 3-bin boundary optimization, all 267 | 0.361460 / 0.606900 | 0.311323 / 0.480760 |

The actual existing `out/michael-v0/order.json` contains **154 images**, despite
the feedback referring to 176. The shared-subset comparison removes 113 new-only
images from v1 and compares original unmirrored v0 poses with canonical mirrored
v1 poses. Removing those intervening frames changes adjacency; this is explicitly
labeled in `report.json`. The full video's actual adjacent metrics are the
267-image row above. The shared comparison is **8.7% worse in mean and 3.7% worse
in p90 than v0**, so this run does not establish an improvement over v0's pose
continuity. The unconstrained old path heuristic on the full new 267-image
selection, without mirroring, also achieved mean 0.303310 / p90 0.484398 in a
local diagnostic. Phase order adds a directional hypothesis, not a proof of the
best pose path. Increasing phase bins to 6/8/12 made pose jumps worse in the
local diagnostic. Boundary-aware 2-opt reduces this penalty within fixed bins.

| Adjacent per-image Lab L statistic (8-bit units) | Before tone matching | After tone matching |
| --- | ---: | ---: |
| Mean absolute luminance-mean jump | 38.7535 | 21.0642 (45.6% lower) |
| Mean absolute luminance-std jump | 16.9261 | 9.2574 (45.3% lower) |

These tone measurements are from 96×96 samples of aligned images immediately
before and after correction, before camera scaling and video encoding. They
measure a flicker proxy, not perceived quality or a color-management guarantee.

### Encoded visual review

Reviewed the ordered skeleton contact sheet, 20 decoded video frames spread
through 0–35.167 seconds, and the two sides of the six largest pose jumps.
The two QA sheets are generated from the encoded MP4, with frame/time labels.

- The aligned figure often occupies a stable central area, and luminance
  correction reduces harsh exposure differences while retaining saturated
  stained glass and warm paintings. Strong changes in background and material
  still read as flashes; this is a cut montage, not continuous animation.
- Zoom progresses on every frame, rather than restarting at each image. A 12%
  change over 35.2 seconds is subtle across such heterogeneous crops; it does
  not overcome extreme perspective and anatomy changes.
- The geometric phase order broadly collects similar arm configurations, but
  cannot identify whether a still pose is raising or striking. Several hands
  hold scales/shields rather than a weapon; uncertain or incorrect joints remain.
- A first-bin outlier was initially chosen as the hook and created a 0.953
  opening jump. The hook selection was corrected to use the first bin's medoid
  neighbourhood, then the full video and encoded QA were regenerated. The
  remote outlier is no longer prepended; it still exists later in the montage.
- False positives are still conspicuous. One late frame shows mostly a stone
  architectural cap with no visible central figure; another depicts a different
  figure in a multi-person painting. The final frame is a strongly tilted rear
  view of a sculpture with large blurred edge extension. These need person
  overrides/manual selection; no semantic model or human accuracy count is
  claimed. The output is **not ready for publication without curation**.
- Some 1–2 frame late cuts are too fast to inspect, and body/background changes
  remain visible despite lower flicker. Holding fewer curated images longer is
  an artistic alternative, but was not used: all 267 selected images appear once.

### Checks and local artifacts

- Network-free synthetic suite: **20 passed, 1 model smoke test skipped**.
  New checks cover reflection labels/matrices and majority voting, monotonic
  phase bins and cycles, safe hooks, material runs, multi-folder IDs, manifest
  invalidation, v0 shared-subset comparison, variable timing/frame preservation,
  beat tolerance/audio offsets, generated pulse onset detection, global camera
  center, chroma preservation and reduced luminance flicker, and an encoded
  end-to-end video with all main options enabled.
- `ruff check`, `ruff format --check`, `compileall`, and `git diff --check` passed.
- FFprobe confirms 1080×1920/30 fps, 1,056 frames, 35.2-second video/audio,
  H.264/yuv420p, limited-range BT.709 and AAC. Full decode with FFmpeg reports
  no errors. Long jobs use `nice -n 10`; FFmpeg uses two threads.
- One final-pass stage total is about **4 min 54 s**: detect 157.35 s, select
  0.01 s, align 101.31 s, order 2.10 s, render 28.90 s, report 3.84 s. This
  excludes iterative ordering experiments, earlier renders, audio extraction,
  visual review and checks.

All media and local source paths remain gitignored in `out/michael-v1/`:
`cut.mp4`, `source-audio.wav`, `detect.json`, `select.json`, `align.json`,
`order.json`, `render-cut.json`, `render.json`, `report.json`,
`contact-sheet.jpg`, and `qa/encoded-frames.jpg`, `qa/largest-jumps.jpg`,
`qa/ffprobe.json`, `qa/decode.log`, `qa/verification.json`.
No source images, output media or weights are committed.

## Local v1.1 validation — 2026-11-08

This pass addresses the coordinator's observation that v1's opening contained
unrelated figures and unstable head framing, while its 25-second spear-statue
run was much more coherent. It reuses the same 398 read-only inputs and cached
pose detections. The final output is `out/michael-v1.1/cut.mp4`, with the exact
same extracted source audio as v1 (SHA-256 equality checked).

### Relevance and selection

The optional SigLIP2 model and license verification are recorded in
[models.md](models.md). All 398 original images and their aligned viewport
previews were scored locally: 796 views, five positive and thirteen negative
prompts in `examples/michael-prompts.json`. Use the lower of original/view
positive-vs-negative contrast scores. Per-view logits, winning texts, prompts,
model revision and embedding-cache provenance are in `relevance.json`.

Two practical errors were verified and corrected during review:

- Full-image scoring accepted an artwork containing Michael while alignment
  showed its painter/donor/another saint. For one painter image, the full view
  scored about 0.968 but its displayed crop scored about 0.003 after adding the
  corresponding negative description. The final pipeline gates both views.
- Painting-oriented descriptions rejected good spear-statue images. Adding
  explicit wooden-church-statue and spear/dragon positives raised five checked
  statues from roughly 0.12–0.57 to 0.82–0.99. The result is prompt-dependent,
  not calibrated, and can still miss or falsely accept a scene. No source-name
  blacklist or manual person overrides were used.

Final settings: `--confidence .2 --min-relevance .8 --min-pose-conf .35
--min-person-area .12 --alignment head-torso --max-head-error 75
--max-upscale 100 --size 1080x1920 --mirror auto --order phase --hook 3`,
three phase bins, one cycle, the existing time-based speed curve
`0:.25,.6:.2,.9:.1,1:.1`, `--duration 35.2 --beat-tolerance .045 --zoom 1:1.12
--normalize-tone`. The external optional environment used torch 2.14.1,
Transformers 5.19.0 and MPS float16. `nice -n 10` is inherited by the scorer;
FFmpeg uses two threads.

Filter counts are **exclusive first-failing reasons**, in pipeline order:

| Gate | Dropped | Remaining |
| --- | ---: | ---: |
| Input | — | 398 |
| Original head confidence | 58 | 340 |
| Original shoulder confidence | 52 | 288 |
| Original hip confidence | 20 | 268 |
| Original anchor outside image | 1 | 267 |
| Scene relevance below 0.8 | 112 | 155 |
| Chosen person box below 12% of source area | 8 | 147 |
| Minimum head/shoulder/hip/wrist confidence below 0.35 | 56 | 91 |
| Post-fit selected facial landmark >75 px from head target | 29 | 62 |
| Source enlargement >100% (scale >2.0) | 2 | 60 |
| Count cap | 0 | 60 |

The automatic duration/curve cap is **203**, but only 60 pass the gates. The
pipeline does not refill the cap with rejected images. Quality scores are
`relevance × minimum pose confidence / (1 + alignment RMS / 38.4)` at this
resolution. Explicit smaller `--max-images N` caps use that ranking. The cap,
its source and component scores are recorded in `select.json`.

### Final video and metrics

60 distinct selected images appear exactly once, versus 267 in v1 and 154 in
the existing v0. The output remains 1,056 frames, 1080×1920, 30 fps and 35.200 s,
H.264/yuv420p limited-range BT.709, with 35.200 s AAC stereo at 48 kHz. Ten images
are mirrored. Holds span 10–25 frames (0.333–0.833 s), so this is a slower edit
than v1. The same audio provides 48 onset candidates; three internal cuts snap
within 45 ms. Global zoom still progresses from 1.00 to 1.12 without restarting.

| Pose-distance scope | Before mean / p90 | v1.1 mean / p90 |
| --- | ---: | ---: |
| Seeded random vs final, same 60 poses | 0.695090 / 1.072366 | 0.363235 / 0.605496 |
| v1 order vs final, same 60 shared images | 0.445409 / 0.720192 | 0.363235 / 0.605496 |
| v0 order vs final, 30 shared images | 0.510479 / 0.913215 | 0.448417 / 0.709843 |

The shared v1 mean improves by 18.4%; the shared v0 mean by 12.2%. These
comparisons use actual stored canonical aligned poses, including mirrors.
They remove reference-only images, changing adjacency. The full original
videos have lower actual mean pose steps (v1 0.311323, v0 0.321548) than this
60-image edit's 0.363235; different counts/poses prevent claiming a universal
continuity win. `report.json` labels both shared and full-reference metrics.

Head jitter uses a confidence-weighted nose/eye center, falling back to the
selected facial point; reference measurements are converted to 1080×1920.
Cut-boundary measurements exclude duplicate held frames and global zoom.

| Geometric diagnostic | v0 full edit | v1 full edit | v1.1 |
| --- | ---: | ---: | ---: |
| Detected face-center jitter, mean / p90 px | 55.885 / 105.646 | 57.292 / 108.824 | ≈0 / ≈0 |
| Torso-length jump, mean / p90 px | 23.187 / 43.482 | 24.505 / 50.021 | ≈0 / ≈0 |
| Torso-length variance, px² | 1112.488 | 954.135 | ≈0 |
| Selected best facial-landmark jitter, mean / p90 px | 45.105 / 78.783 | 43.323 / 74.518 | 49.806 / 93.267 |

**The near-zero values are imposed alignment constraints, not independent
visual tracking accuracy.** Nose/eye-center translation and fixed torso length
make those two diagnostics nearly zero by construction. The alternate facial
landmark can be a different eye/ear across shots and still jumps; head outlines,
size and perspective also vary visibly. Propagating the model centers through
holds and global zoom yields predicted output-frame head jitter mean/p90
0.042586/0.042586 px, including cut boundaries. This likewise is a geometric
prediction, not a measured post-encoding face track.

The new head/torso constraints increase five-anchor RMS to 66.997 px, versus
38.375 in v1's compromise fit; they trade exact torso/head-anchor least squares
for a fixed face center and torso length. 38/60 images use some edge extension,
with mean uncovered area 4.72% (v1: 7.77%). Tone matching reduces adjacent Lab-L
mean jumps from 32.3133 to 16.8489 (47.9%) and contrast jumps from 14.9927 to
8.6927. These are pre-camera, pre-encoding 96×96-sample statistics.

### Encoded review and remaining limits

Reviewed decoded opening frames at 0, 0.5, …, 4.5, 4.75 and 4.99 seconds;
25.0–25.7 seconds at 0.1-second spacing; and 20 frames spanning the full video.
The same opening/climax samples from both v0 and v1 were inspected side by side
as version-labelled sheets. First-five-second and 24–29-second audio/video
excerpts are also retained in `qa/`.

- **Opening versus v1:** materially clearer. The first shots show Michael with
  a defeated creature rather than the unrelated red-robed saint or painter
  seen in earlier passes. Figure placement is steadier and 0.7–0.8 s early
  holds give time to read the subject. Watermarked modern sculpture photos,
  gold icons, painted illustrations and very different head outlines still
  change the look abruptly. This is not a photometrically uniform sequence.
- **Versus v0:** relevance and deliberate framing are better controlled, but
  v0's denser cut sequence can feel smoother as motion. v1.1 is slower, with
  more visible pose steps. No claim that it is uniformly better than v0.
- **Climax versus v1:** the checked 25-second v1.1 passage is now two sword-raised
  paintings with defeated figures visible. It is coherent, but the coordinator's
  original v1 spear-statue run remains the stronger material-consistent run at
  that exact time. Several approved statues survive v1.1 later in the edit.
  Maintaining the identical 25-second statue run was not a selection constraint.
- **Remaining semantic/crop failures:** a Coventry statue crop still pushes the
  defeated figure out of the visible frame, and a ceiling/flying-angel image
  with large lower-edge extension still passes SigLIP2. These are documented
  false accepts, despite checking the viewport. Stronger automatic thresholds
  also reject genuine images; a semantic model score alone is insufficient for
  publication-ready curation. The model's zero face-center jitter does not hide
  these defects. Manual rejection/person overrides remain necessary for a final edit.

### Checks and artifacts

- Synthetic, network-free suite: **26 passed, 1 real-model smoke test skipped**.
  Added prompt validation/group contrast, optional-worker contract and minimum
  of full/crop scores, relevance dependencies/staleness, per-filter rejection,
  product ranking/automatic count, geometry stabilization, reference resolution
  and actual canonical pose comparisons, and explicit rendered-frame/cut jitter.
- `ruff check`, `ruff format --check`, `compileall`, `git diff --check` passed.
- FFprobe verified video/audio dimensions, format, timing and color metadata;
  full video/audio decode completed without errors. Source audio matches v1
  byte-for-byte. All 60 shown filenames are unique; primary drop counts sum to
  338; all 1,056 render frames are accounted for by holds.
- Final prompt-only rescoring reused the 796-view embeddings (about 5.31 s).
  The final alignment and render took about 22.41 s and 19.94 s respectively.
  These exclude the first full SigLIP2 pass, rejected experiments and visual QA.

Gitignored outputs: `out/michael-v1.1/cut.mp4`, `source-audio.wav`,
`relevance.json`, `select.json`, `align.json`, `order.json`, `report.json`,
`contact-sheet.jpg`, and `qa/` with version-labelled opening/climax JPEGs,
`encoded-frames.jpg`, `opening-5s.mp4`, `climax-24-29s.mp4`, `ffprobe.json`,
`decode.log`, `verification.json`. No source media or weights are committed.

## Art-domain backend validation (v2) — 2026-11-08

An art-domain pose model was evaluated instead of building a human-review UI.
This pass keeps the same two read-only Michael folders (398 inputs), the exact
v1.1 relevance prompts/thresholds and the same head-torso alignment, phase order,
mirror, tone and zoom settings. The experimental backend is official HumanArt+
COCO YOLOX-M + RTMPose-L, both pre-exported FP32 ONNX on CPU. See
[models.md](models.md) for candidate selection, exact artifacts/hashes and the
unresolved HumanArt weight commercial scope. It is not the default backend.
No dataset, images, videos or weights are committed.

A verified implementation issue was found independently of model replacement:
the original and new pose SDK `pipeline.json` files require RGB, while the old
adapter fed BGR. `body7-rgb` is a separate corrected control using the old
weights/detector; `body7` preserves historical outputs for reproduction. Thus
v2 changes detector, pose capacity/training and color preprocessing; improvements
cannot be attributed to HumanArt training alone.

### Detection, confidence and strict-filter counts

An actual box means the detector returned a person box, not that the correct
Michael figure has been independently verified. A full-frame pose fallback is
not counted as a successful detection. All three 398-image runs completed
without read/inference errors.

| Run | Inputs | Actual person-box returns | Full-frame fallback | Required-joint min ≥0.35 | Median required-joint min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical v1/v1.1 Body7 | 398 | 181 (45.5%) | 217 | 151 | 0.2627 |
| Body7 RGB-only control | 398 | 181 (45.5%) | 217 | 168 | 0.2844 |
| HumanArt v2 | 398 | 396 (99.5%) | 2 | 258 | 0.4372 |

Required confidence is the minimum of the best facial point, both shoulders,
hips and wrists. Scores are raw SimCC maxima clipped to [0,1], **not calibrated
probabilities or an accuracy score**; distributions differ across checkpoints.
The new adapter retains raw maxima too. Full distributions are in
`evaluation.json`; selected v2 percentiles are:

| Confidence group, all 398 chosen poses | p10 | Median | p90 |
| --- | ---: | ---: | ---: |
| Best facial point | 0.5647 | 0.8505 | 0.9781 |
| Minimum shoulder | 0.4347 | 0.6342 | 0.7483 |
| Minimum hip | 0.3648 | 0.5922 | 0.7058 |
| Minimum wrist | 0.2088 | 0.4511 | 0.7368 |
| Minimum of all required groups | 0.2040 | 0.4372 | 0.6630 |

Exactly the v1.1 gates were applied: base confidence .2, relevance .8 on both
original/displayed views, minimum pose confidence .35, minimum person area .12,
post-alignment selected facial-landmark error ≤75 px at 1080×1920, enlargement
≤100% (scale ≤2), head-torso alignment. Same SigLIP2 model/revision and prompts;
no threshold relaxation, source blacklist or manual person override.
Counts are exclusive first-failing reasons, so later gates inspect different
populations; a rise in relevance rejections is not a detector accuracy measure.

| First-failing gate | v1.1 | RGB-only control | HumanArt v2 |
| --- | ---: | ---: | ---: |
| Original head/shoulder/hip/geometry gates combined | 131 | 124 | 16 |
| Relevance | 112 | 114 | 161 |
| Small chosen box | 8 | 8 | 17 |
| Required pose confidence | 56 | 50 | 64 |
| Selected head-landmark error | 29 | 32 | 18 |
| Excessive upscale | 2 | 2 | 9 |
| Count cap | 0 | 0 | 0 |
| **Pass all / shown once** | **60** | **68** | **113** |

V2 supplies 53 more strict-filter images than v1.1 (+88.3%), or 45 more than the
RGB-only control (+66.2%). The duration/curve cap remains 203 and never binds.
At 0.2 seconds per image, 35.2 seconds requires **176** unique cuts. V2 supplies
**113/176 = 64.2%**, still **63 cuts / 12.6 seconds short**. The run does not
silently restore rejected images or repeat images to fill the gap.

### Video and continuity comparison

Two outputs preserve the distinction:

- `cut-020.mp4`: 113 images ×6 frames = 678 frames at 30 fps, **22.600 s**,
  fixed 0.2-second cuts. Audio is the first 22.6 seconds of the same master;
  its original late music climax is not reached in this diagnostic version.
- `cut.mp4`: same 113 images once, **1,056 frames / 35.200 s**, original
  `0:.25,.6:.2,.9:.1,1:.1` relative speed curve, full same audio track, tone and
  continuous zoom 1.00→1.12. Holds are 5–14 frames, with 19 of 112 internal cuts
  snapping within 45 ms of 48 onset candidates. This is not a uniform 0.2-s edit.

Both are 1080×1920 H.264/yuv420p limited-range BT.709 with AAC. All encoding
uses two FFmpeg threads, and long jobs use `nice -n 10`. The main pass mirrors
12 images; 56/113 need edge extension, averaging 3.50% missing area.

Actual full shown-sequence distances, using the existing confidence-weighted,
torso-normalized pairwise-Procrustes definition (lower is better):

| Version | Inputs / shown | Full-sequence mean | Full-sequence p90 |
| --- | ---: | ---: | ---: |
| v0 | 233 / 154 | 0.321548 | 0.519133 |
| v1 | 398 / 267 | 0.311323 | 0.480760 |
| v1.1 | 398 / 60 | 0.363235 | 0.605496 |
| RGB-only control | 398 / 68 | 0.358687 | 0.520042 |
| HumanArt v2 | 398 / 113 | 0.325638 | 0.505559 |
| Seeded random v2 order | 398 / 113 | 0.662789 | 1.060016 |

**0.445409 is not the full v0 value**: it was the v1 order restricted to the
60 images shared with v1.1. Its corresponding v1.1 result was 0.363235. The
existing v0 manifest has 154 images, not the 176 mentioned in earlier feedback.
Counts and pose estimates change here, so the full table alone does not prove
a fair, constant-image continuity gain. Shared-subset comparisons are included:

| Reference vs v2, shared images only | Count | Before mean / p90 | V2 mean / p90 |
| --- | ---: | ---: | ---: |
| v0 | 55 | 0.454846 / 0.717054 | 0.388447 / 0.613690 |
| v1 | 100 | 0.381249 / 0.692200 | 0.340854 / 0.532071 |
| v1.1 | 45 | 0.377309 / 0.617912 | 0.400111 / 0.592309 |
| RGB-only control | 49 | 0.373404 / 0.572884 | 0.383367 / 0.583235 |

These use actual stored canonical aligned poses, not a common set of ground
truth joints; checkpoint confidence/coordinate differences also affect distance.
Reference-only/new-only images are skipped, changing adjacency. V2 improves the
full v1.1 result but its shared-45 mean is slightly **worse**; no universal
continuity optimum is claimed. Head-torso mode still imposes near-zero detected
face-center/torso-length variance by construction; that is not independent
pixel tracking and cannot validate anatomy or semantic subject identity.

### Random 30-image visual audit

Seed **20261108**, sampling without replacement from **all 398 inputs**, before
quality filtering. Four columns show the source, historical Body7, RGB-only
control and HumanArt. The full `random-30-contact-sheet.jpg`, six readable
five-image pages, the sample JSON and per-image Korean notes are in `qa/`.
All 30 were inspected; notes are tied to current detection/reference hashes and
sample filenames. This is a single unblinded qualitative review, without joint
ground truth; the grades below are **not mAP or a validated accuracy rate**.

| Informal grade | Historical Body7 | RGB-only control | HumanArt |
| --- | ---: | ---: | ---: |
| Intended figure + roughly useful head/torso/arm pose | 7 | 8 | 15 |
| Figure broadly located, important limb/geometry errors remain | 7 | 6 | 10 |
| Wrong subject/region or no coherent useful pose, including low-confidence failures | 15 | 15 | 4 |
| Abstract depiction cannot be anatomically adjudicated | 1 | 1 | 1 |

Clear improvements include line-art figures (#1/#3), stained-glass head location
(#2/#4/#26), a small medal figure (#7), choosing the large angel instead of a
small donor (#9), and locating a painting inside an altar instead of peripheral
ornament (#18). RGB-only changes do not explain these large gains. Existing
models already perform fairly well on many conventional statues (#12/#21/#24/
#27); a complex icon's raised arm (#28) is better in the RGB control than in
HumanArt. Dense multi-person scenes (#11/#15/#16) and a vase/base false target
(#10) still defeat automatic selection. Those four major-failure sampled cases
are all rejected by the final strict filters. Better person localization does
not make the prominence heuristic a semantic Michael identifier.

Also inspected the encoded main video's first five seconds, 25.0–25.7 seconds
and 20 frames across the duration. The opening shows clearer Michael/defeated
creature pairs than v1's unrelated-figure examples. The 25-second spear-statue
passage again reads coherently, with similar arm configurations. Large artwork
style/brightness/head-size changes, some missing feet/defeated figures outside
the crop, and tilted/rear-view sculptures still cause visible jumps. A ceiling
image with broad blurred extension still survives the semantic gate. The
conclusion is **a real improvement in art-domain localization/pose utility,
especially line art and stained glass, but not an automatic publication-quality
montage or a solution to the 176-cut shortage**. No human-review UI was built.

### Validation and resources

- **32 synthetic tests passed, 1 real-model smoke test skipped**. New checks
  cover post-NMS xyxy decoding without a second grid decode, RGB/BGR behavior,
  backend-aware cache invalidation, fallback accounting, deterministic audit
  selection/source immutability, stale review-note rejection and configurable
  target duration/cut interval. The actual 398-image inference is the real-model
  integration exercise.
- Ruff check/format, compileall and `git diff --check` passed. FFprobe verified
  both frame counts/durations, codec, pixel format, range/color tags and audio;
  full decode of both videos completed without errors.
- Main HumanArt inference: 491.99 s; same legacy-weight RGB control: see its
  detection manifest. HumanArt relevance: 225.07 s; alignment: 67.88 s; paced
  render: 21.16 s. These are local wall times with concurrent low-priority work,
  not a hardware-normalized benchmark. The model cache occupies about 390 MiB.
- Outputs are gitignored under `out/michael-v2-artpose/`. `evaluation.json`,
  `report.json`, `qa/comparison.json`, `qa/visual-review.json`, all ordered/random
  sheets, both video versions and encoder/decode/probe logs are retained locally.
