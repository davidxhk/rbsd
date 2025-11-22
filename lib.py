import io

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import percentile_filter
from scipy.signal import resample_poly, welch

PARAMS = {
    "fs_hz": 8000,
    "frame_ms": 100,
    "hop_ms": 25,
    "band_min_hz": 50,
    "band_max_hz": 300,
    "quiet_pct": 20,
    "quiet_delta": 4.0,
    "quiet_win_s": 6.0,
    "dur_min_s": 0.5,
    "dur_max_s": 5.0,
}


def load_audio(source: str | bytes | io.BufferedIOBase):
    """
    Load audio from a file path, bytes, or file-like object.
    Returns (fs, x) where x is a mono float32 array.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    fs, x = wavfile.read(source)

    # to float
    if x.dtype.kind in "iu":
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)

    # to mono
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    return fs, x


def compute_band_power(
    x, fs, frame_len, hop_len, band_min_hz, band_max_hz, window="hann"
):
    """
    Computes the power in a specific frequency band over time using Welch's method on frames.
    """
    n = len(x)
    band_powers = []
    frame_times = []

    for i in range(0, n - frame_len, hop_len):
        frame = x[i : i + frame_len]

        # Use Welch's method for PSD on this specific frame
        f_w, Pxx = welch(frame, fs=fs, nperseg=len(frame), window=window)

        # Calculate power in the specific snore band
        idx_min = np.searchsorted(f_w, band_min_hz)
        idx_max = np.searchsorted(f_w, band_max_hz)

        # Sum of PSD components in band (proportional to power)
        e = np.sum(Pxx[idx_min:idx_max])
        p_db = 10 * np.log10(e + 1e-15)

        band_powers.append(p_db)
        frame_times.append((i + frame_len / 2) / fs)

    return np.array(band_powers), np.array(frame_times)


def process_file(
    file: str | bytes | io.BufferedIOBase,
    *,
    fs_hz=PARAMS["fs_hz"],
    frame_ms=PARAMS["frame_ms"],
    hop_ms=PARAMS["hop_ms"],
    band_min_hz=PARAMS["band_min_hz"],
    band_max_hz=PARAMS["band_max_hz"],
    quiet_pct=PARAMS["quiet_pct"],
    quiet_delta=PARAMS["quiet_delta"],
    quiet_win_s=PARAMS["quiet_win_s"],
    dur_min_s=PARAMS["dur_min_s"],
    dur_max_s=PARAMS["dur_max_s"],
):
    # 1. Load and Downsample
    fs, x = load_audio(file)
    x_ds = resample_poly(x, fs_hz, fs)

    frame_s = frame_ms * 1e-3
    hop_s = hop_ms * 1e-3

    # 2. Setup parameters
    frame_len = max(1, int(frame_s * fs_hz))
    hop_len = max(1, int(hop_s * fs_hz))

    # Calculate filter size in frames
    size = max(1, int(quiet_win_s / hop_s))

    # 3. Compute Features (Welch)
    p_pow, frame_times = compute_band_power(
        x_ds, fs_hz, frame_len, hop_len, band_min_hz, band_max_hz
    )

    if len(p_pow) == 0:
        return {
            "snore_detected": False,
            "snore_count": 0,
            "segments": [],
        }

    # 4. Calculate quiet percentile (adaptive background noise level)
    p_quiet = percentile_filter(p_pow, quiet_pct, size)

    # 5. Apply criteria
    mask_valid = p_pow > (p_quiet + quiet_delta)

    # 6. Group Events
    events = []
    in_event = False
    ev_start = None

    def add_event(start, end):
        dur = end - start
        if dur_min_s <= dur <= dur_max_s:
            events.append(
                {"start": float(start), "end": float(end), "duration": float(dur)}
            )

    for i, valid in enumerate(mask_valid):
        if valid and not in_event:
            in_event = True
            ev_start = frame_times[i]
        elif not valid and in_event:
            in_event = False
            ev_end = frame_times[i]
            add_event(ev_start, ev_end)

    if in_event:
        add_event(ev_start, frame_times[-1])

    return {
        "snore_detected": len(events) > 0,
        "snore_count": len(events),
        "segments": events,
    }
