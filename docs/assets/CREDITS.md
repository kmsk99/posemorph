# Demo credits

The README demo is made from **90 renders of one 3D character**: "Animated Woman"
by [Quaternius](https://quaternius.com), downloaded from
[Poly Pizza](https://poly.pizza/m/9kF7eTDbhO) and released under
**CC0 1.0 (public domain)**. Six animations (Jump, Jump2, PickUp, Punch, Running,
Walking — five frames each from the rigged animations) were rendered in Blender
from three camera angles (−40°, 0°, +40°) on a plain backdrop.

The renders themselves are not stored in this repository; `before-after.gif`,
`morph.gif` and `contact-sheet.jpg` in this folder are derived from them.

## Reproduce

```sh
# 1. Render the inputs (Blender 4.2+); see examples/demo/render_poses.py
blender -b --python examples/demo/render_poses.py -- animated-woman.glb ./renders

# 2. Shuffle them so the input order means nothing (the demo used seed 7),
#    then run posemorph
posemorph run ./renders-shuffled --out ./out/woman --size 540x720 \
  --confidence .2 --alignment head-torso --max-upscale 40 --mode cut --hold 4
posemorph render --out ./out/woman --mode morph --hold 2 --transition 3
posemorph report --out ./out/woman
```

85 of 90 renders passed; five crouching or high-jump frames were dropped by
`--max-upscale 40`. `before-after.gif` shows the shuffled inputs (left) next to
`cut.mp4` (right); `morph.gif` is `morph.mp4`; `contact-sheet.jpg` shows the
first 24 images of the report's contact sheet. GIFs were downscaled with
FFmpeg's palette filters.
