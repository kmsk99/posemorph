# Model and dependency licenses

Verified against the linked upstream repositories/model cards on 2026-11-08.
No weights are committed or redistributed by posemorph. posemorph's own code is
licensed under Apache-2.0 (see [LICENSE](../LICENSE)); that license does not
cover downloaded weights, training data or input artwork.

## Default inference pair

| Component | Exact artifact | Published license and evidence |
| --- | --- | --- |
| Person detector | Official YOLOX-s COCO ONNX, release `0.1.1rc0`, `yolox_s.onnx` | Apache-2.0: [upstream LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE), [official ONNX instructions and weights](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime). The release is distributed by that upstream Apache-2.0 repository, with no separate weight license stated. |
| Body keypoints | RTMPose-m Body7, 256×192, `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip` | Apache-2.0 explicitly declared by [the RTMPose maintainer's model mirror card](https://huggingface.co/Tau-J/RTMPose/blob/cd4d709/README.md); [exact checkpoint archive](https://huggingface.co/Tau-J/RTMPose/blob/cd4d709/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip). Original code: [MMPose LICENSE](https://github.com/open-mmlab/mmpose/blob/main/LICENSE). |
| Runtime | ONNX Runtime CPU | MIT: [LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |

The pose checkpoint URL pins the mirror's `cd4d709` revision. Downloaded ONNX
SHA-256 values are checked against constants in the adapter and written with
URLs into `detect.json`; downloads and extraction are atomic and
stored under `~/.cache/posemorph` (`POSEMORPH_CACHE` or `--cache` overrides it).
Raw SimCC confidence maxima are clipped to `[0,1]` for the output schema;
these scores are not calibrated probabilities.
No CUDA, PyTorch, mmcv, or mmpose installation is needed. ONNX Runtime sessions
are sequential, with one intra-op and one inter-op thread each. Input images
stay on the local machine; first-use downloads request only public weights.

These are the publishers' stated software/weight licenses. They do **not** clear
rights to training images, input artwork, or output distribution. The
[RTMPose model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#body-2d)
identifies Body7's datasets as AIC, COCO, CrowdPose, MPII, sub-JHMDB, Halpe and
PoseTrack18. Dataset terms can differ from model-card licenses. This is not a
claim that every training dataset has unrestricted commercial terms.

## Alternatives examined

- **RTMPose through rtmlib:** [rtmlib code is Apache-2.0](https://github.com/Tau-J/rtmlib/blob/main/LICENSE),
  and its [README](https://github.com/Tau-J/rtmlib#model-zoo) lists convenient
  ONNX models. Its default person detectors are HumanArt-trained. posemorph uses
  a small independent ONNX adapter instead, to explicitly choose weights, keep
  thread counts bounded, and avoid installing two OpenCV packages.
- **HumanArt-trained RTMPose/YOLOX:** excluded from the default; now available
  as the experimental v2 backend documented below. The
  [HumanArt README](https://github.com/IDEA-Research/HumanArt#dataset-download)
  requests authorization for non-commercial dataset use. A repository code
  license alone does not resolve that restriction.
- **DWPose / RTMW:** [DWPose code](https://github.com/IDEA-Research/DWPose) uses
  Apache-2.0, but whole-body datasets introduce additional terms; 133 landmarks
  also add cost without helping v0's torso alignment. Not downloaded.
- **ViTPose ONNX:** larger models and exports exist; the
  [ViTPose repository](https://github.com/ViTAE-Transformer/ViTPose) is Apache-2.0.
  Exact export provenance and training terms require separate validation. Not downloaded.
- **MediaPipe Pose:** [MediaPipe code](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
  is Apache-2.0. Its video/fitness-oriented detector was not chosen for this
  artwork collection; individual model bundle terms would need checking. Not downloaded.

## Other dependencies

NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), Pillow (HPND), OpenCV (Apache-2.0
for current releases), and pytest (MIT; development only). See each installed
package's license for notices. FFmpeg is an external executable, not bundled;
its license depends on the installed build (libx264-enabled builds are GPL).

## Optional scene-relevance scorer (v1.1)

`google/siglip2-so400m-patch14-384`, revision
`e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, is opt-in. On 2026-11-08,
Google's [official model card](https://huggingface.co/google/siglip2-so400m-patch14-384)
and the [pinned model-card source](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/README.md)
both declare `license: apache-2.0`. The pinned source was fetched and checked
before use. The model card identifies WebLI pretraining; the published model
license does not clear training/input artwork rights or guarantee semantic
accuracy on religious artwork. Weights remain in the user's Hugging Face cache,
never in this repository.

The optional `relevance` extra adds PyTorch (BSD-style) and Transformers
(Apache-2.0), or `--relevance-python` can use an existing environment. The base
runtime imports neither. The verified local environment used torch 2.14.1,
Transformers 5.19.0 and MPS float16, with two CPU threads. EXIF-corrected images
and aligned viewport previews stay on the local machine. Only public model and
processor files are downloaded if absent from cache. Full-image and displayed
viewport embeddings are cached under the gitignored output folder; the score
is the lower of their positive-vs-negative prompt contrasts, not a calibrated
probability or a guarantee that the chosen skeleton belongs to Michael.

## Experimental art-domain backend (v2)

`--pose-backend humanart` selects an official **YOLOX-M HumanArt+COCO** detector
and **RTMPose-L HumanArt+COCO** 17-keypoint estimator. It is opt-in; the default
`body7` remains the historical pair. The local evaluation uses the published
FP32 ONNX exports on one-thread CPU ONNX Runtime, not CUDA/MPS. No conversion
or new inference dependency was needed. Both ONNX files together are about
202 MiB; archives plus extracted models occupy about 400 MiB in
`~/.cache/posemorph/artpose` (`--cache`/`POSEMORPH_CACHE` changes the parent).
Images and weights are not uploaded or redistributed.

### Code, weights and data are distinct

The [MMPose code license](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
and [MMDetection code license](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)
are Apache-2.0. HumanArt's [repository code license](https://github.com/IDEA-Research/HumanArt/blob/main/LICENSE)
is also Apache-2.0. However, its [dataset instructions](https://github.com/IDEA-Research/HumanArt#dataset-download)
require approval for non-commercial data use and prohibit private dataset
transfer. The README refers generically to a CC license without resolving the
exact dataset terms applicable to these weight artifacts.

For the **trained weights**, official availability and an Apache-licensed code
repository do not establish an independent commercial-use grant. No definitive
model-specific commercial/redistribution permission was found. The exact
YOLOX-M ONNX artifact is the subject of [MMPose issue 3271](https://github.com/open-mmlab/mmpose/issues/3271),
and pretrained-model commercial use is also raised in [HumanArt issue 11](https://github.com/IDEA-Research/HumanArt/issues/11).
Treat the HumanArt backend as **research/experimental, commercial scope
unresolved**, including any local ONNX conversion. This does not assert that
a dataset restriction automatically transfers to all trained weights; it records
that the scope has not been authoritatively resolved. This experiment downloads
only published weights, not the HumanArt dataset, and does not redistribute them.
Do not make it the permissive default on the strength of the code license alone.

### Candidate comparison and exact selection

Reviewed the official [HumanArt RTMPose model table](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/humanart/rtmpose_humanart.md),
[HumanArt ViTPose table](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/humanart/vitpose_humanart.md)
and [art-domain detector/export list](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
Source snapshot: MMPose `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` and HumanArt
`9650297c5c2eb73ca4f2c26fe41a842383d7aba6`, checked 2026-11-08.

| Candidate | Published evidence / decision | Code / exact weights scope |
| --- | --- | --- |
| RTMPose-M / L HumanArt+COCO | HumanArt detected-box AP 0.355 / 0.378, versus COCO-only 0.239 / 0.260 in the same table; official ONNX exports. Select L for the modest 105.5 MiB size and existing SimCC adapter. | MMPose Apache-2.0; exact weights commercial scope unresolved. |
| ViTPose-B / L / H HumanArt+COCO | Official `.pth` checkpoints and configs `td-hm_ViTPose-{base,large,huge}_8xb64-210e_humanart-256x192`; detected-box AP 0.410 / 0.459 / 0.468. Higher published AP than the selected RTMPose-L, so not ruled out on quality. Not converted/evaluated here: heavier backbones and heatmap/UDP export/decoder path; RTMPose already has official usable CPU ONNX. | MMPose Apache-2.0; same HumanArt weight/data uncertainty. |
| RTMDet tiny / s HumanArt | Official person-detector weights/configs, AP 46.6 / 50.6. M/L/X configs also exist. Investigated, not executed or exported in this pass. | MMDetection Apache-2.0; HumanArt weight/data scope unresolved. |
| YOLOX-M HumanArt+COCO | Official person AP 59.1, pre-exported CPU ONNX; selected over the above small RTMDet checkpoints for adapter compatibility and the published result. | MMDetection/MMPose Apache-2.0; HumanArt weight/data scope unresolved. |

Published AP is from HumanArt validation, not this Michael corpus. Models change
capacity, detector, training mixture and preprocessing together; this is not a
controlled estimate of the contribution of HumanArt training alone. Runtime
single-pass inference also does not reproduce benchmark flip-test settings.

Selected archives, each linked by the official export list:

- [YOLOX-M ONNX archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip):
  `yolox_m_8xb8-300e_humanart-c2c7a14a.zip`, extracted ONNX SHA-256
  `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23`.
- [RTMPose-L ONNX archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.zip):
  `rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.zip`, extracted ONNX SHA-256
  `443c883202be9afa2dac1cb03257e9475f4c5205aea3b8be97086293ee12b237`.

Hashes are measured from the downloaded official artifacts and pinned to detect
later corruption/change, not upstream digital signatures. Archive `detail.json`
links back to the corresponding official `.pth` checkpoints. The detection
export returns decoded xyxy boxes/scores plus labels with NMS already applied;
it must not be decoded again as raw YOLOX grid predictions.

### RGB control and historical reproducibility

Both the existing Body7 SDK and the HumanArt pose SDK specify `to_rgb: true` in
`pipeline.json`, while detection specifies BGR. The historical posemorph adapter
normalized BGR for pose inference. `--pose-backend body7-rgb` corrects only the
pose color conversion with the original permissive checkpoint pair. `body7`
retains historical BGR behavior to reproduce v0/v1/v1.1; `humanart` uses RGB as
specified by its SDK. The corrected control separates this implementation bug
from model replacement effects. Backend identity participates in detection-cache
invalidation; old manifests without it are treated as historical `body7`.
Do not attribute all v2 changes to art-domain training. See [validation](validation.md).
