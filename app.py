import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram
from scipy.ndimage import percentile_filter

from lib import compute_band_power, load_audio, process_file, PARAMS

# -------------------------- CONFIG --------------------------
FIGSIZE = (12, 6)
DPI = 300


# -------------------------- UTILS ---------------------------
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def parse_time(t_str: str) -> float:
    """Parses a time string (e.g. '1:30', '90', '1:05.5') into seconds."""
    try:
        parts = t_str.split(":")
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    return 0.0


def format_time(seconds: float) -> str:
    """Formats seconds into mm:ss string."""
    if seconds < 60:
        return f"{seconds:.1f}"

    m = int(seconds // 60)
    s = seconds % 60
    # If seconds is effectively an integer (close enough to 0 fractional part)
    if abs(s - round(s)) < 0.001:
        return f"{m}:{int(round(s)):02d}"
    return f"{m}:{s:04.1f}"


def get_qp(key, default, caster=float):
    """Helper to get query params with default fallback."""
    if key in st.query_params:
        try:
            return caster(st.query_params[key])
        except:
            pass
    return default


def update_qp(key, value, default):
    """Helper to update query params if value differs from default."""
    if value != default:
        st.query_params[key] = value
    elif key in st.query_params:
        del st.query_params[key]


@st.cache_data
def generate_plot(
    x,
    fs,
    segments,
    t_start=0.0,
    **params,
):
    """
    Generates the composite plot (Spectrogram + Power) and returns:
    1. The image bytes (PNG)
    2. The x_start, x_end pixel coordinates of the axes for the overlay
    """
    fs_hz = params["fs_hz"]
    frame_ms = params["frame_ms"]
    hop_ms = params["hop_ms"]
    band_min_hz = params["band_min_hz"]
    band_max_hz = params["band_max_hz"]
    quiet_pct = params["quiet_pct"]
    quiet_delta = params["quiet_delta"]
    quiet_win_s = params["quiet_win_s"]

    # 1. Compute Data
    # Resample audio to target rate (e.g. 8kHz) for consistent processing
    x_ds = resample_poly(x, fs_hz, fs)
    n = len(x_ds)

    frame_s = frame_ms * 1e-3
    hop_s = hop_ms * 1e-3

    # 2. Setup parameters
    frame_len = max(1, int(frame_s * fs_hz))
    hop_len = max(1, int(hop_s * fs_hz))

    # Calculate filter size in frames
    size = max(1, int(quiet_win_s / hop_s))

    # Compute Spectrogram using scipy
    f, t_spec, Sxx = spectrogram(
        x_ds,
        fs_hz,
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        scaling="density",
        mode="psd",
    )
    Sxx_dB = 10 * np.log10(Sxx + 1e-15)  # Convert to dB

    # Limit freq for display (only show up to 1000Hz)
    f_mask = f <= 1000
    S = Sxx_dB[f_mask, :]

    # Power curve calculation
    p_pow, t_pow = compute_band_power(
        x_ds, fs_hz, frame_len, hop_len, band_min_hz, band_max_hz
    )

    # Calculate quiet percentile (adaptive background noise level)
    p_quiet = percentile_filter(p_pow, quiet_pct, size)

    # Shift time axis to match original audio timestamp
    t_spec += t_start
    t_pow += t_start
    t_end = t_start + n / fs_hz

    # 2. Plot
    with plt.style.context("default"):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # Top: Spectrogram
        # Use calculated time extent based on the slice duration
        vmax = np.percentile(S, 95)
        vmin = vmax - 60
        ax_top.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=[t_start, t_end, 0, 1000],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax_top.set_ylabel("Frequency [Hz]")
        ax_top.set_title("Spectrogram & Relative Power")

        # Overlays
        for seg in segments:
            # Only draw if in range (matplotlib handles clipping, but optimization helps)
            if seg["end"] >= t_start and seg["start"] <= t_end:
                ax_top.axvspan(seg["start"], seg["end"], color="red", alpha=0.2)
                ax_bot.axvspan(seg["start"], seg["end"], color="red", alpha=0.2)

        # Bottom: Power
        ax_bot.plot(t_pow, p_pow, lw=1, color="blue", label="Rel Power")
        ax_bot.plot(
            t_pow,
            p_quiet + quiet_delta,
            lw=1.5,
            color="red",
            linestyle="--",
            label=f"Threshold [P{quiet_pct}({quiet_win_s} s) + {quiet_delta} dB]",
        )
        ax_bot.set_ylabel("Relative Power [dB]")
        ax_bot.set_xlabel("Time [s]")
        ax_bot.set_xlim(t_start, t_end)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper right", fontsize="small")

        fig.tight_layout()

        # 3. Calculate pixel coordinates for the JS overlay
        # Force a draw so we can get bbox
        fig.canvas.draw()

        # Get the bounding box of the bottom axis (shared x) in display coordinates
        # We need the x-range of the data area relative to the full image width
        bbox = ax_bot.get_position()  # in figure fraction (0..1)

        width_px = FIGSIZE[0] * DPI
        x_start_px = int(bbox.x0 * width_px)
        x_end_px = int((bbox.x0 + bbox.width) * width_px)

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue(), x_start_px, x_end_px


@st.cache_data
def process_audio(file, params):
    return process_file(file, **params)


# -------------------------- APP -----------------------------
st.set_page_config("Rule-based Snore Detection", page_icon="😴", layout="wide")

st.title("😴 Rule-based Snore Detection Algorithm")

st.markdown(
    """
    Upload a WAV file to analyze snoring using a Rule‑based Snore Detection algorithm.

    #### Quick Start
    - Upload a WAV (mono or stereo). The app will resample and convert to mono using the target sample rate set in the sidebar.
    - The app processes the file locally; returned events are shown as red overlays on the spectrogram.
    - Examine the visualization to see the relative power line and the adaptive threshold used by the algorithm.

    #### Tips & Tricks
    - Use the sidebar to tune the snore band, adaptive threshold, and event duration to match your recordings.
    - Player controls: Space/Enter = play/pause, ←/→ = seek, ↑/↓ = volume, Home/End = jump to start/end.
    """
)

with st.expander("ℹ️ Algorithm background"):
    st.markdown(
        f"""
        ### Rule-based detection approach
        1. **Pre-processing** — audio is downsampled to {PARAMS["fs_hz"]} Hz, converted to mono, and split into {PARAMS["frame_ms"]} ms frames with {PARAMS["hop_ms"]} ms hops.
        2. **Spectral analysis** — each frame is windowed and evaluated with Welch’s PSD; energy within {PARAMS["band_min_hz"]}–{PARAMS["band_max_hz"]} Hz is analysed.
        3. **Adaptive threshold** — a rolling {PARAMS["quiet_win_s"]} s of the quietest {PARAMS["quiet_pct"]} % of frames models background noise. Frames that exceed it by {PARAMS["quiet_delta"]} dB are marked as snore candidates.
        4. **Event shaping** — consecutive candidates are merged and only durations between {PARAMS["dur_min_s"]} s and {PARAMS["dur_max_s"]} s are retained.
        
        **Reference**  
        Nakano, H., Ikeda, T., Hayashi, M., Ohshima, E., Itoh, M., & Nishikata, N. (2014). *Monitoring sound to quantify snoring and sleep apnea severity using a smartphone: proof of concept.* Journal of Clinical Sleep Medicine, 10(1), 73–78. [https://doi.org/10.5664/jcsm.3364](https://doi.org/10.5664/jcsm.3364)
        """
    )

# --- Sidebar Configuration ---
st.sidebar.header("Detection Parameters")

st.sidebar.subheader("Pre-processing")

# Prepare parameters for algorithm
params = {}

params["fs_hz"] = st.sidebar.number_input(
    "Target sample rate (Hz)",
    min_value=1000,
    max_value=48000,
    value=get_qp("fs_hz", PARAMS["fs_hz"], int),
    step=1000,
    help=f"Resamples audio to this rate before processing. [Default: {PARAMS["fs_hz"]} Hz]",
)

params["frame_ms"] = st.sidebar.number_input(
    "Frame length (ms)",
    min_value=10,
    max_value=1000,
    value=get_qp("frame_ms", PARAMS["frame_ms"], int),
    step=10,
    help=f"Duration of each analysis frame. [Default: {PARAMS["frame_ms"]} ms]",
)

params["hop_ms"] = st.sidebar.number_input(
    "Hop length (ms)",
    min_value=5,
    max_value=500,
    value=get_qp("hop_ms", PARAMS["hop_ms"], int),
    step=5,
    help=f"Time step between consecutive frames. [Default: {PARAMS["hop_ms"]} ms]",
)

st.sidebar.subheader("Spectral analysis")

params["band_min_hz"], params["band_max_hz"] = st.sidebar.slider(
    "Snore band (Hz)",
    min_value=20,
    max_value=1000,
    value=(
        get_qp("band_min_hz", PARAMS["band_min_hz"], int),
        get_qp("band_max_hz", PARAMS["band_max_hz"], int),
    ),
    step=5,
    help=f"Selects the frequency window emphasized for snore likelihood. [Default: {PARAMS["band_min_hz"]}–{PARAMS["band_max_hz"]} Hz]",
)

st.sidebar.subheader("Adaptive threshold")

params["quiet_pct"] = st.sidebar.number_input(
    "Quiet percentile",
    min_value=1,
    max_value=50,
    value=get_qp("quiet_pct", PARAMS["quiet_pct"], int),
    step=1,
    help=f"Sets the percentile used to model quiet frames for gating. [Default: {PARAMS["quiet_pct"]}th percentile]",
)

params["quiet_delta"] = st.sidebar.number_input(
    "Quiet delta (dB)",
    min_value=1.0,
    max_value=20.0,
    value=get_qp("quiet_delta", PARAMS["quiet_delta"], float),
    step=0.5,
    help=f"Adds headroom above the quiet percentile to flag candidates. [Default: {PARAMS["quiet_delta"]} dB]",
)

params["quiet_win_s"] = st.sidebar.number_input(
    "Quiet window (s)",
    min_value=1.0,
    max_value=10.0,
    value=get_qp("quiet_win_s", PARAMS["quiet_win_s"], float),
    step=0.5,
    help=f"Controls how many frames used to model quiet frames. [Default: {PARAMS["quiet_win_s"]} s]",
)

st.sidebar.subheader("Event shaping")

params["dur_min_s"] = st.sidebar.number_input(
    "Min Duration (s)",
    min_value=0.1,
    max_value=2.0,
    value=get_qp("dur_min_s", PARAMS["dur_min_s"], float),
    step=0.1,
    help=f"Minimum duration for a valid snore event. [Default: {PARAMS["dur_min_s"]} s]",
)

params["dur_max_s"] = st.sidebar.number_input(
    "Max Duration (s)",
    min_value=1.0,
    max_value=10.0,
    value=get_qp("dur_max_s", PARAMS["dur_max_s"], float),
    step=0.1,
    help=f"Maximum duration for a valid snore event. [Default: {PARAMS["dur_max_s"]} s]",
)

# Update query parameters
for key in params:
    update_qp(key, params[key], PARAMS[key])

file = st.file_uploader("Upload WAV file", type=["wav"])

if not file:
    st.info("No file uploaded. Using the default snoring audio sample.")
    file = "./assets/snore.wav"


# 1. Process Audio
try:
    results = process_audio(file, params)
    segments = results.get("segments", [])
    st.success(f"Detected {len(segments)} snore events.")
except Exception as e:
    st.error(f"Processing Error: {e}")
    st.stop()

# 2. Process Audio for Visualization
fs, x = load_audio(file)
n = len(x) / fs

# Initialize session state for zoom if not present
if "t_start" not in st.session_state:
    st.session_state.t_start = get_qp("t_start", 0.0, float)
if "t_end" not in st.session_state:
    st.session_state.t_end = get_qp("t_end", min(120.0, n), float)

# Ensure zoom is within bounds (e.g. if file changed)
t_start = min(st.session_state.t_start, n)
t_end = min(st.session_state.t_end, n)

# Validate order
if t_end <= t_start:
    t_end = min(t_start + 10.0, n)
    if t_end <= t_start:  # if at end of file
        t_start = max(0.0, t_end - 10.0)

st.session_state.t_start = t_start
st.session_state.t_end = t_end
st.session_state.t_start_str = format_time(t_start)
st.session_state.t_end_str = format_time(t_end)

# Sync QPs on load/render to ensure consistency if bounds logic changed values
update_qp("t_start", t_start, 0.0)
update_qp("t_end", t_end, min(120.0, n))

# --- Zoom Control ---
st.markdown("### Visualization")


# Callbacks for text inputs
def update_start():
    s = parse_time(st.session_state.t_start_str)
    s = max(0.0, min(s, n))
    st.session_state.t_start = s
    st.session_state.t_start_str = format_time(s)
    update_qp("t_start", s, 0.0)


def update_end():
    s = parse_time(st.session_state.t_end_str)
    s = max(0.0, min(s, n))
    st.session_state.t_end = s
    st.session_state.t_end_str = format_time(s)
    update_qp("t_end", s, min(120.0, n))


c1, c2 = st.columns(2)
c1.text_input("Start Time (mm:ss.ms)", key="t_start_str", on_change=update_start)
c2.text_input("End Time (mm:ss.ms)", key="t_end_str", on_change=update_end)

# Slice audio for plotting
idx_start = int(t_start * fs)
idx_end = int(t_end * fs)
x_view = x[idx_start:idx_end]

img_bytes, x_start, x_end = generate_plot(
    x_view,
    fs,
    segments,
    t_start,
    **params,
)

# Create sliced audio for playback so player controls match the zoom
buf_wav = io.BytesIO()
wavfile.write(buf_wav, fs, x_view)
wav_view_bytes = buf_wav.getvalue()

# Prepare Data URIs
img_b64 = f"data:image/png;base64,{_b64(img_bytes)}"
audio_b64 = f"data:audio/wav;base64,{_b64(wav_view_bytes)}"

# 3. Render HTML/JS Player
# We inject the calculated x_start/x_end into the JS
html = f"""
<style>
.snore-wrap {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 1280px;
  aspect-ratio: 12 / 7;
  margin: 0 auto;
}}
.stage {{
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08);
}}
.spectrogram {{
  width: 100%;
  height: auto;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}}
.overlay {{
  position: absolute;
  inset: 0;
  cursor: crosshair;
  touch-action: none;
  border-radius: 8px;
  outline: none;
}}
.shade {{
  position: absolute;
  top: 0;
  height: 100%;
  background: rgba(0, 0, 0, 0.25);
  pointer-events: none;
}}
#shadeL {{ left: 0; width: 0; }}
#shadeR {{ right: 0; width: 0; }}
.fill {{
  position: absolute;
  top: 0;
  height: 100%;
  background: rgba(255, 255, 255, 0.40);
  left: 0;
  width: 0;
  transform: translateX(0) scaleX(0);
  transform-origin: left center;
  will-change: transform;
  pointer-events: none;
}}
.cursor {{
  position: absolute;
  top: 0;
  height: 100%;
  width: 0;
  border-left: 2px solid #000;
  filter: drop-shadow(1px 0 0 #fff) drop-shadow(-1px 0 0 #fff);
  transform: translateX(0);
  will-change: transform;
  pointer-events: none;
}}
.info {{
  font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  opacity: 0.75;
}}
</style>

<div class="snore-wrap">
  <div class="stage" id="stage">
    <img id="spectrogram" class="spectrogram" src="{img_b64}" alt="spectrogram analysis">
    <div id="overlay" class="overlay" aria-label="scrub surface" tabindex="0">
      <div id="shadeL" class="shade"></div>
      <div id="fill" class="fill"></div>
      <div id="cursor" class="cursor"></div>
      <div id="shadeR" class="shade"></div>
    </div>
  </div>
  <audio id="audio" src="{audio_b64}" controls preload="auto" style="width: 100%"></audio>
  <div class="info">
    Showing {t_start:.1f}s - {t_end:.1f}s of {n:.1f}s. 
    Detected {len(segments)} events total.
  </div>
</div>

<script>
const audio   = document.getElementById('audio');
const img     = document.getElementById('spectrogram');
const overlay = document.getElementById('overlay');
const fillEl  = document.getElementById('fill');
const curEl   = document.getElementById('cursor');
const shadeL  = document.getElementById('shadeL');
const shadeR  = document.getElementById('shadeR');

// Dynamic coordinates from Python
const X_START_NATURAL = {x_start};
const X_END_NATURAL   = {x_end};
const IMG_WIDTH_NATURAL = {FIGSIZE[0] * DPI};

let dragging = false, running = false;
let xsCss = 0, xeCss = 0;

function clamp(v, a, b) {{ return Math.max(a, Math.min(b, v)); }}

function computeRegion() {{
  // Calculate ratio based on current display width vs original generated width
  const ratio = img.clientWidth / IMG_WIDTH_NATURAL;
  
  xsCss = Math.round(X_START_NATURAL * ratio);
  xeCss = Math.round(X_END_NATURAL   * ratio);

  overlay.style.width  = img.clientWidth + "px";
  overlay.style.height = img.clientHeight + "px";

  fillEl.style.left  = xsCss + "px";
  fillEl.style.width = Math.max(0, xeCss - xsCss) + "px";

  shadeL.style.width = Math.max(0, xsCss) + "px";
  shadeR.style.width = Math.max(0, img.clientWidth - xeCss) + "px";
}}

function setVisuals(){{
  const t = audio.currentTime || 0;
  const d = audio.duration || 1; // avoid div/0
  
  // Since audio is sliced to match the view, t=0 is start, t=duration is end.
  const r = t / d;
  
  const xCss  = xsCss + r * (xeCss - xsCss);
  
  fillEl.style.transform = `translateX(0) scaleX(${{clamp(r,0,1)}})`;
  curEl.style.transform  = `translateX(${{xCss}}px)`;
}}

function startLoop(){{
  if (running) return;
  running = true;
  const step = () => {{
    if (!running) return;
    setVisuals();
    requestAnimationFrame(step);
  }};
  requestAnimationFrame(step);
}}

function stopLoop(){{ running = false; }}

function ratioFromCssX(xCss){{
  const xClamped = Math.max(xsCss, Math.min(xeCss, xCss));
  return (xClamped - xsCss) / Math.max(1e-6, (xeCss - xsCss));
}}

function seekBy(seconds){{
  if (!isFinite(audio.duration)) return;
  audio.currentTime = clamp((audio.currentTime || 0) + seconds, 0, audio.duration);
  setVisuals();
}}

function volBy(delta){{ audio.volume = clamp((audio.volume ?? 1) + delta, 0, 1); }}
function toggleMute() {{ audio.muted = !audio.muted; }}
function togglePlay() {{ if (audio.paused) audio.play(); else audio.pause(); }}

// Mouse/touch seeking
overlay.addEventListener('pointerdown', (e) => {{
  dragging = true;
  overlay.focus();
  const rect = overlay.getBoundingClientRect();
  
  // Calculate ratio based on click position
  const r = ratioFromCssX(e.clientX - rect.left);
  
  if (isFinite(audio.duration)) {{
      audio.currentTime = r * audio.duration;
  }}
  setVisuals();
  startLoop();
}});
window.addEventListener('pointerup', () => {{ dragging = false; if (audio.paused) stopLoop(); }});
overlay.addEventListener('pointermove', (e) => {{
  if (!dragging) return;
  const rect = overlay.getBoundingClientRect();
  const r = ratioFromCssX(e.clientX - rect.left);
  if (isFinite(audio.duration)) {{
      audio.currentTime = r * audio.duration;
  }}
}});

// Keyboard controls
overlay.addEventListener('keydown', (e) => {{
  const mod = (e.ctrlKey || e.metaKey) ? 'ctrl' : (e.shiftKey ? 'shift' : 'none');
  switch (e.key) {{
    case ' ':
    case 'Enter': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft':
      e.preventDefault();
      if (mod === 'ctrl') seekBy(-10);
      else if (mod === 'shift') seekBy(-1);
      else seekBy(-5);
      break;
    case 'ArrowRight':
      e.preventDefault();
      if (mod === 'ctrl') seekBy(+10);
      else if (mod === 'shift') seekBy(+1);
      else seekBy(+5);
      break;
    case 'ArrowUp':   e.preventDefault(); volBy(+0.05); break;
    case 'ArrowDown': e.preventDefault(); volBy(-0.05); break;
    case 'PageUp':    e.preventDefault(); volBy(+0.10); break;
    case 'PageDown':  e.preventDefault(); volBy(-0.10); break;
    case 'Home':      e.preventDefault(); if (isFinite(audio.duration)) audio.currentTime = 0; setVisuals(); break;
    case 'End':       e.preventDefault(); if (isFinite(audio.duration)) audio.currentTime = audio.duration; setVisuals(); break;
    case 'm':
    case 'M':         e.preventDefault(); toggleMute(); break;
    default: break;
  }}
}});

// Audio + controls
audio.addEventListener('play', startLoop);
audio.addEventListener('pause', () => {{ if (!dragging) stopLoop(); }});
audio.addEventListener('timeupdate', () => {{ if (!running) setVisuals(); }});

// Init
function init(){{ computeRegion(); setVisuals(); }}
if (img.complete) init(); else img.onload = init;
window.addEventListener('resize', () => init());
</script>
"""

components.html(html, height=800, scrolling=False)
