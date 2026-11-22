# posemorph

Open-source tool (Apache-2.0): pose-aligned montage and morph video from many images of the same subject.

- Do not commit images, videos, model weights or output directories (`out/`). Keep source paths of private collections out of the repository.
- Check the license of every model/weight before adding it as a dependency or default; record it in `docs/models.md`.
- Keep the default backend reproducible; new backends are opt-in and invalidate the detection cache.
- Run `pytest -q` and `ruff check src tests` before committing.
