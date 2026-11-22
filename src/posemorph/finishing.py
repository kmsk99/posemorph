"""Frame timing, lightweight onset analysis, camera and luminance finishing."""

import json
import subprocess

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, stft


def parse_curve(value):
    try:
        points = np.array([list(map(float, p.split(":"))) for p in value.split(",")])
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
            or points[0, 0] != 0
            or points[-1, 0] != 1
            or np.any(np.diff(points[:, 0]) <= 0)
            or np.any(points[:, 1] <= 0)
        ):
            raise ValueError
        return points
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "--speed-curve needs increasing fraction:seconds points from 0 to 1, with positive holds"
        ) from exc


def parse_zoom(value):
    try:
        values = tuple(map(float, value.split(":")))
        if (
            len(values) != 2
            or not np.isfinite(values).all()
            or min(values) < 1
            or max(values) > 4
        ):
            raise ValueError
        return values
    except (ValueError, TypeError) as exc:
        raise ValueError("--zoom needs start:end factors in [1,4]") from exc


def cut_timing(count, fps, hold, curve=None, duration=None, beats=None, tolerance=0.06):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("--beat-tolerance must be finite and nonnegative")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("--duration must be finite and positive")
    if curve:
        points = parse_curve(curve)
        fraction = np.linspace(0, 1, 10001)
        density = 1 / np.interp(fraction, points[:, 0], points[:, 1])
        cumulative = np.r_[
            0, np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(fraction))
        ]
        seconds = count / cumulative[-1] if duration is None else duration
        boundaries = (
            np.interp(np.linspace(0, cumulative[-1], count + 1), cumulative, fraction)
            * seconds
        )
    else:
        seconds = count * hold / fps if duration is None else duration
        boundaries = np.linspace(0, seconds, count + 1)
    total = round(seconds * fps)
    if total < count:
        raise ValueError(
            "Duration too short: every selected image needs at least one frame"
        )
    frames = np.round(boundaries * fps).astype(int)
    # Enforce one frame per image even with steep curves and rounding.
    for i in range(1, count):
        frames[i] = np.clip(frames[i], frames[i - 1] + 1, total - (count - i))
    snapped = 0
    beat_values = np.asarray([] if beats is None else beats, float)
    if (
        beat_values.ndim != 1
        or not np.isfinite(beat_values).all()
        or np.any(beat_values < 0)
    ):
        raise ValueError("Beats must be a finite list of nonnegative seconds")
    beat_frames = np.unique(np.round(beat_values * fps).astype(int))
    for i in range(1, count):
        candidates = beat_frames[
            (beat_frames > frames[i - 1]) & (beat_frames < frames[i + 1])
        ]
        if candidates.size:
            candidate = int(
                candidates[np.argmin(abs(candidates - boundaries[i] * fps))]
            )
            if abs(candidate / fps - boundaries[i]) <= tolerance:
                frames[i] = candidate
                snapped += 1
    return np.diff(frames).tolist(), {
        "cut_times": (frames / fps).tolist(),
        "snapped_cuts": snapped,
        "beat_count": len(beat_values),
        "beat_tolerance": tolerance,
        "requested_duration": duration,
        "speed_curve": curve,
    }


def read_beats(path, audio_start=0):
    """JSON timestamps refer to the original audio timeline, before --audio-start."""
    data = json.loads(path.read_text())
    data = data.get("beats") if isinstance(data, dict) else data
    try:
        values = np.asarray(data, float)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "--beats must contain a list of nonnegative seconds or {beats: [...]} "
        ) from exc
    return np.sort(values[values >= audio_start] - audio_start).tolist()


def audio_onsets(path, audio_start, duration):
    """Spectral-flux onset candidates, not a musical beat/downbeat tracker.

    Uses the already required FFmpeg/NumPy/SciPy; no weights or new dependencies.
    """
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-ss",
            str(audio_start),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "11025",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    samples = np.frombuffer(raw, dtype="<f4")
    if len(samples) < 1024:
        return []
    _, times, spectrum = stft(samples, fs=11025, nperseg=1024, noverlap=768)
    magnitude = np.log1p(20 * np.abs(spectrum))
    flux = np.maximum(np.diff(magnitude, axis=1), 0).sum(0)
    flux = gaussian_filter1d(flux, 0.8)
    if flux.max(initial=0) <= 1e-8:
        return []
    peaks, _ = find_peaks(
        flux, distance=8, prominence=max(1e-6, float(np.std(flux) * 0.5))
    )
    return times[peaks + 1].tolist()


def camera_frame(image, progress, zoom, center):
    factor = zoom[0] + (zoom[1] - zoom[0]) * progress
    if factor == 1:
        return image
    matrix = np.array(
        [[factor, 0, center[0] * (1 - factor)], [0, factor, center[1] * (1 - factor)]],
        dtype=float,
    )
    h, w = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def luminance_stats(image):
    small = cv2.resize(image, (96, 96))
    light = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
    return np.array([light.mean(), max(light.std(), 1.0)])


class ToneNormalizer:
    def __init__(self):
        self.target = None
        self.before, self.after = [], []

    def apply(self, image, enabled):
        stats = luminance_stats(image)
        self.before.append(stats.tolist())
        if self.target is None:
            self.target = stats.copy()
        if enabled:
            # Slow target follows original moments, not its own corrected output.
            self.target = 0.92 * self.target + 0.08 * stats
            gain = np.clip(self.target[1] / stats[1], 0.8, 1.25)
            shift = np.clip(self.target[0] - stats[0], -28, 28)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            light = lab[:, :, 0].astype(np.float32)
            matched = (light - stats[0]) * gain + stats[0] + shift
            lab[:, :, 0] = np.clip(light * 0.25 + matched * 0.75, 0, 255).astype(
                np.uint8
            )
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.after.append(luminance_stats(image).tolist())
        return image

    def report(self):
        def jumps(values):
            differences = np.abs(np.diff(np.asarray(values), axis=0))
            return {
                "mean_luminance_jump": float(differences[:, 0].mean())
                if len(differences)
                else 0.0,
                "mean_contrast_jump": float(differences[:, 1].mean())
                if len(differences)
                else 0.0,
            }

        return {
            "before": jumps(self.before),
            "after": jumps(self.after),
            "per_image_before": self.before,
            "per_image_after": self.after,
        }
