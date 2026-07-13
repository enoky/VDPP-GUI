"""VDPP GUI — temporally consistent depth post-processing for depth-map sequences.

Runs the VDPP model on an existing depth-map input (16/8-bit grayscale PNG
sequence or a depth video file) and writes the refined result as either a
16-bit grayscale PNG sequence or a 10-bit grayscale MP4 (via FFmpeg).

Unlike run_video.py, this tool does NOT run an image-to-depth model: the
input is assumed to already be depth maps, which are fed directly into VDPP.

Memory model: the input is never loaded whole. Scene boundaries are found in
a low-resolution detection pass (OmniShotCut or a built-in threshold
detector), then each scene is processed in chunks of at most "max chunk
frames". Chunks within a scene overlap and are aligned to each other with a
least-squares scale/shift fit (the same strategy VDPP uses internally), so
temporal consistency is preserved across chunk borders while RAM/VRAM stay
bounded regardless of input length.

Optional dependency for AI scene detection (install without deps so it
cannot downgrade torch/transformers):
    pip install --no-deps git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git
    pip install decord

Usage:
    python app.py
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(REPO_ROOT, "settings.json")
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "vdpp.pth")

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

SCENE_MODE_OMNISHOTCUT = "OmniShotCut (AI)"
SCENE_MODE_THRESHOLD = "Threshold (fast)"
SCENE_MODE_NONE = "None (single scene)"
SCENE_MODES = (SCENE_MODE_OMNISHOTCUT, SCENE_MODE_THRESHOLD, SCENE_MODE_NONE)

OMNISHOTCUT_REPO = "uva-cv-lab/OmniShotCut"

# Frames shared between consecutive chunks of the same scene; used to fit the
# scale/shift alignment that keeps chunk outputs temporally consistent.
CHUNK_OVERLAP = 16
MIN_CHUNK_FRAMES = 64
MIN_SCENE_FRAMES = 4

# Hide console windows spawned by ffmpeg/ffprobe on Windows.
SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class CancelledError(Exception):
    pass


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def ffprobe_video(path):
    """Return (width, height, fps, frame_count_or_None) for a video file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            creationflags=SUBPROCESS_FLAGS)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])

    num, _, den = stream.get("r_frame_rate", "30/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    if fps <= 0:
        fps = 30.0

    nb_frames = stream.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames and nb_frames.isdigit() else None
    return width, height, fps, frame_count


def pick_10bit_encoder():
    """Return (codec_name, extra_args) for 10-bit encoding, preferring HEVC."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                            capture_output=True, text=True,
                            creationflags=SUBPROCESS_FLAGS)
    encoders = result.stdout if result.returncode == 0 else ""
    if "libx265" in encoders:
        return "libx265", ["-tag:v", "hvc1"]
    if "libx264" in encoders:
        return "libx264", ["-profile:v", "high10"]
    raise RuntimeError("FFmpeg build has neither libx265 nor libx264; "
                       "cannot encode 10-bit MP4.")


# ---------------------------------------------------------------------------
# Frame sources (streaming input)
# ---------------------------------------------------------------------------

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_image_sequence(folder):
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]


def load_image_as_depth(path):
    """Read an image as a float32 grayscale depth map in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


class ImageSequenceSource:
    """Depth frames stored as individual image files in a folder."""

    def __init__(self, folder):
        self.paths = list_image_sequence(folder)
        if not self.paths:
            raise RuntimeError(f"No image files found in {folder}")
        first = load_image_as_depth(self.paths[0])
        self.height, self.width = first.shape
        self.frame_count = len(self.paths)
        self.fps = None

    def frames(self, cancelled):
        for path in self.paths:
            if cancelled.is_set():
                raise CancelledError()
            frame = load_image_as_depth(path)
            if frame.shape != (self.height, self.width):
                raise RuntimeError(
                    f"Frame size mismatch: {os.path.basename(path)} is "
                    f"{frame.shape[1]}x{frame.shape[0]}, expected "
                    f"{self.width}x{self.height}")
            yield frame

    def lowres_frames(self, width, height, cancelled):
        """Yield uint8 grayscale frames at (height, width) for scene detection."""
        for path in self.paths:
            if cancelled.is_set():
                raise CancelledError()
            frame = load_image_as_depth(path)
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            yield (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)


class VideoSource:
    """Depth frames stored in a video file, decoded through FFmpeg pipes.

    Full-resolution decoding uses gray16le so 10/12/16-bit depth videos keep
    their full precision (OpenCV's VideoCapture would clamp them to 8-bit).
    """

    def __init__(self, path):
        self.path = path
        self.width, self.height, self.fps, self.frame_count = ffprobe_video(path)

    def _pipe(self, extra_args, pix_fmt):
        cmd = ["ffmpeg", "-v", "error", "-i", self.path, "-map", "0:v:0",
               *extra_args, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                creationflags=SUBPROCESS_FLAGS)

    def _read_pipe(self, proc, frame_bytes, decode, cancelled):
        try:
            while True:
                if cancelled.is_set():
                    raise CancelledError()
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield decode(buf)
            stderr = proc.stderr.read().decode(errors="replace").strip()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg decode failed:\n{stderr}")
        finally:
            if proc.poll() is None:
                proc.kill()

    def frames(self, cancelled):
        proc = self._pipe([], "gray16le")
        width, height = self.width, self.height

        def decode(buf):
            frame = np.frombuffer(buf, dtype=np.uint16).reshape(height, width)
            return frame.astype(np.float32) / 65535.0

        yield from self._read_pipe(proc, width * height * 2, decode, cancelled)

    def lowres_frames(self, width, height, cancelled):
        proc = self._pipe(["-vf", f"scale={width}:{height}"], "gray")

        def decode(buf):
            return np.frombuffer(buf, dtype=np.uint8).reshape(height, width).copy()

        yield from self._read_pipe(proc, width * height, decode, cancelled)


class FrameTaker:
    """Pull-based wrapper around a frame generator."""

    def __init__(self, generator):
        self._gen = generator
        self.exhausted = False

    def take(self, n):
        out = []
        for _ in range(n):
            try:
                out.append(next(self._gen))
            except StopIteration:
                self.exhausted = True
                break
        return out

    def close(self):
        self._gen.close()


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def ranges_from_cuts(cut_starts, total):
    """Build contiguous [start, end) scene ranges from sorted cut positions."""
    starts = sorted({0, *(c for c in cut_starts if 0 < c < total)})
    ranges = [[s, e] for s, e in zip(starts, starts[1:] + [total])]
    # Merge scenes that are too short into their predecessor.
    merged = []
    for r in ranges:
        if merged and r[1] - r[0] < MIN_SCENE_FRAMES:
            merged[-1][1] = r[1]
        elif merged and merged[-1][1] - merged[-1][0] < MIN_SCENE_FRAMES:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def detect_scenes_threshold(source, log, progress, cancelled):
    """Cheap streaming cut detector on downscaled depth frames.

    Flags a cut when the mean absolute inter-frame difference spikes well
    above the recent median difference.
    """
    cuts = []
    diffs = []
    prev = None
    count = 0
    for frame in source.lowres_frames(128, 96, cancelled):
        if prev is not None:
            diff = float(np.mean(np.abs(frame.astype(np.int16)
                                        - prev.astype(np.int16)))) / 255.0
            recent = diffs[-60:]
            baseline = float(np.median(recent)) if recent else 0.0
            if diff > max(0.08, 5.0 * baseline + 0.02):
                cuts.append(count)
            diffs.append(diff)
        prev = frame
        count += 1
        progress("detect", count, source.frame_count)
    log(f"Threshold detector: {len(cuts)} cut(s) in {count} frames.")
    return cuts, count


def detect_scenes_omnishotcut(source, log, progress, cancelled):
    """Learned shot-boundary detection via OmniShotCut on downscaled frames."""
    import torch  # noqa: F401 — must be imported before decord (DLL conflict)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OmniShotCut requires a CUDA GPU. Switch scene detection to "
            f"'{SCENE_MODE_THRESHOLD}' instead.")
    try:
        import omnishotcut
    except ImportError:
        raise RuntimeError(
            "OmniShotCut is not installed. Install it with:\n"
            "pip install --no-deps git+https://github.com/"
            "UVA-Computer-Vision-Lab/OmniShotCut.git\npip install decord\n\n"
            f"Or switch scene detection to '{SCENE_MODE_THRESHOLD}'.")

    log("Loading OmniShotCut (downloads ~100MB checkpoint on first use)...")
    model = omnishotcut.load(OMNISHOTCUT_REPO)
    args = getattr(model, "_model_args", None)
    height = getattr(args, "process_height", 96)
    width = getattr(args, "process_width", 128)

    frames = []
    for frame in source.lowres_frames(width, height, cancelled):
        frames.append(frame)
        progress("detect", len(frames), source.frame_count)
    total = len(frames)
    if total == 0:
        raise RuntimeError("Input contains no frames.")

    log(f"Running OmniShotCut on {total} frames at {width}x{height}...")
    array = np.repeat(np.stack(frames)[..., None], 3, axis=-1)  # (T,H,W,3)
    del frames
    # "default" mode keeps every shot (clean_shot drops transition shots,
    # which would leave gaps in the timeline). Every returned boundary is
    # treated as a scene cut.
    ranges, intra_labels, _ = model.inference(array, mode="default")
    del array, model
    torch.cuda.empty_cache()

    cuts = [int(r[0]) for r in ranges if r[0] > 0]
    log(f"OmniShotCut: {len(cuts)} cut(s) in {total} frames "
        f"(shot types: {sorted(set(intra_labels))}).")
    return cuts, total


# ---------------------------------------------------------------------------
# Output writers (streaming)
# ---------------------------------------------------------------------------

class PngSequenceWriter:
    def __init__(self, outdir):
        self.outdir = outdir
        self.index = 0
        os.makedirs(outdir, exist_ok=True)

    def write(self, frame_u16):
        path = os.path.join(self.outdir, f"{self.index:08d}.png")
        if not cv2.imwrite(path, frame_u16):
            raise RuntimeError(f"Failed to write {path}")
        self.index += 1

    def close(self, abort=False):
        pass


class VideoWriter10Bit:
    """Streams uint16 grayscale frames into a 10-bit MP4 through FFmpeg."""

    def __init__(self, path, width, height, fps, crf):
        codec, extra = pick_10bit_encoder()
        self.codec = codec
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "gray16le",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
            "-c:v", codec, "-pix_fmt", "yuv420p10le",
            "-crf", str(crf), "-preset", "medium",
            *extra, path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     creationflags=SUBPROCESS_FLAGS)

    def write(self, frame_u16):
        self.proc.stdin.write(np.ascontiguousarray(frame_u16).tobytes())

    def close(self, abort=False):
        if self.proc.poll() is None:
            if abort:
                self.proc.kill()
                self.proc.wait()
                return
            self.proc.stdin.close()
            stderr = self.proc.stderr.read().decode(errors="replace").strip()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg encoding failed:\n{stderr}")


class SceneOutputBuffer:
    """Collects a scene's chunk outputs, then normalizes the scene as a whole.

    Single-chunk scenes stay in RAM. Multi-chunk scenes are spilled to disk
    quantized to uint16 within each chunk's own range; at finalize time the
    chunks are requantized into the scene-wide range (error <= 1 LSB of the
    16-bit output), so RAM stays bounded by one chunk regardless of scene
    length.
    """

    def __init__(self, tmpdir, scene_index):
        self.tmpdir = tmpdir
        self.scene_index = scene_index
        self.ram_chunk = None            # (array, lo, hi) for single-chunk scenes
        self.disk_chunks = []            # list of (path, lo, hi)
        self.frame_count = 0

    def _spill(self, array, lo, hi):
        span = max(hi - lo, 1e-8)
        quantized = np.round((array - lo) / span * 65535.0).astype(np.uint16)
        path = os.path.join(self.tmpdir,
                            f"scene{self.scene_index:05d}_"
                            f"chunk{len(self.disk_chunks):05d}.npy")
        np.save(path, quantized)
        self.disk_chunks.append((path, lo, hi))

    def add(self, array):
        lo, hi = float(array.min()), float(array.max())
        self.frame_count += array.shape[0]
        if self.ram_chunk is None and not self.disk_chunks:
            self.ram_chunk = (array, lo, hi)
            return
        if self.ram_chunk is not None:
            self._spill(*self.ram_chunk)
            self.ram_chunk = None
        self._spill(array, lo, hi)

    def finalize(self, writer, on_frame, cancelled):
        chunks = ([self.ram_chunk] if self.ram_chunk is not None
                  else list(self.disk_chunks))
        lo = min(c[1] for c in chunks)
        hi = max(c[2] for c in chunks)
        span = max(hi - lo, 1e-8)
        try:
            for chunk in chunks:
                if self.ram_chunk is not None:
                    array = chunk[0]
                else:
                    path, c_lo, c_hi = chunk
                    quantized = np.load(path)
                    array = quantized.astype(np.float32) / 65535.0 \
                        * (c_hi - c_lo) + c_lo
                normalized = np.clip((array - lo) / span, 0.0, 1.0)
                out_u16 = np.round(normalized * 65535.0).astype(np.uint16)
                for i in range(out_u16.shape[0]):
                    if cancelled.is_set():
                        raise CancelledError()
                    writer.write(out_u16[i])
                    on_frame()
        finally:
            for path, _, _ in self.disk_chunks:
                try:
                    os.remove(path)
                except OSError:
                    pass


def fit_scale_shift(x, y):
    """Least-squares (scale, shift) so that scale * x + shift ~= y."""
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    n = x.size
    a00 = float((x * x).sum())
    a01 = float(x.sum())
    b0 = float((x * y).sum())
    b1 = float(y.sum())
    det = a00 * n - a01 * a01
    if det <= 1e-12:
        return 1.0, float(y.mean() - x.mean())
    scale = (n * b0 - a01 * b1) / det
    shift = (a00 * b1 - a01 * b0) / det
    if scale <= 0:  # degenerate fit; fall back to shift-only alignment
        return 1.0, float(y.mean() - x.mean())
    return scale, shift


# ---------------------------------------------------------------------------
# Processing pipeline (runs on a worker thread)
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, params, log, progress, cancelled):
        self.p = params
        self.log = log
        self.progress = progress          # progress(phase, current, total_or_None)
        self.cancelled = cancelled        # threading.Event

    def check_cancel(self):
        if self.cancelled.is_set():
            raise CancelledError()

    # -- input / detection ----------------------------------------------------

    def open_source(self):
        indir = self.p["input_path"]
        if os.path.isdir(indir):
            source = ImageSequenceSource(indir)
            self.log(f"Input: image sequence, {source.frame_count} frames, "
                     f"{source.width}x{source.height}.")
        else:
            source = VideoSource(indir)
            self.log(f"Input: video {os.path.basename(indir)}, "
                     f"{source.width}x{source.height} @ {source.fps:.3f} fps"
                     + (f", {source.frame_count} frames."
                        if source.frame_count else "."))
        return source

    def detect_scenes(self, source):
        """Return list of (start, end) scene ranges covering the whole input.

        In SCENE_MODE_NONE, returns [(0, None)] — a single scene of unknown
        length that is consumed until the source is exhausted.
        """
        mode = self.p["scene_mode"]
        if mode == SCENE_MODE_NONE:
            self.log("Scene detection disabled; treating input as one scene.")
            return [(0, source.frame_count)]

        self.log("Scene detection pass (low resolution)...")
        if mode == SCENE_MODE_OMNISHOTCUT:
            cuts, total = detect_scenes_omnishotcut(
                source, self.log, self.progress, self.cancelled)
        else:
            cuts, total = detect_scenes_threshold(
                source, self.log, self.progress, self.cancelled)
        source.frame_count = total
        scenes = ranges_from_cuts(cuts, total)
        self.log(f"{len(scenes)} scene(s): "
                 + ", ".join(f"[{a}..{b})" for a, b in scenes[:20])
                 + (" ..." if len(scenes) > 20 else ""))
        return scenes

    # -- VDPP -----------------------------------------------------------------

    @staticmethod
    def _xformers_usable(device):
        """True if xformers memory-efficient attention works in fp32 here."""
        if device != "cuda":
            return False
        try:
            import torch
            import xformers.ops
            q = torch.zeros(1, 8, 2, 16, device=device)
            xformers.ops.memory_efficient_attention(q, q, q)
            return True
        except Exception:
            return False

    def load_vdpp(self):
        import torch

        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        self.log(f"Using device: {device}")

        # The vdpp modules use xformers when importable, but its fp32 kernels
        # do not support every GPU (e.g. the newest compute capabilities).
        # Every xformers call site in vdpp has a pure-PyTorch fallback, so if
        # xformers cannot run here, block its import before vdpp loads.
        if "vdpp.vdpp_model" not in sys.modules and not self._xformers_usable(device):
            self.log("xformers unavailable/incompatible on this device; "
                     "using PyTorch scaled_dot_product_attention.")
            sys.modules["xformers"] = None
            sys.modules["xformers.ops"] = None

        from vdpp.vdpp_model import VDPP

        import vdpp.dinov2_layers.attention as dino_attention
        if not dino_attention.XFORMERS_AVAILABLE:
            # Without xformers, DINOv2's fallback materializes the full
            # (tokens x tokens) attention matrix — ~75GB for a 32-frame
            # window at 1080p. Replace it with torch SDPA, which computes
            # the same result with memory-efficient kernels on any GPU.
            self._patch_dino_attention(dino_attention)

        checkpoint = self.p["checkpoint"]
        if not os.path.isfile(checkpoint):
            raise RuntimeError(f"Checkpoint not found: {checkpoint}")

        self.log("Loading VDPP model...")
        model = VDPP(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        state = torch.load(checkpoint, map_location="cpu")
        # The released vdpp.pth carries extra keys (e.g. shift_head.*) that the
        # model class does not define; tolerate extras but never missing weights.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Checkpoint is missing model weights: {missing}")
        if unexpected:
            self.log(f"Ignoring unused checkpoint keys: {sorted(unexpected)}")
        return model.to(device).eval(), device

    @staticmethod
    def _patch_dino_attention(dino_attention):
        import torch.nn.functional as F

        def sdpa_forward(self, x, attn_bias=None):
            B, N, C = x.shape
            qkv = (self.qkv(x)
                   .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
            # SDPA's default scale equals self.scale (head_dim ** -0.5), and
            # attn_drop is a no-op in eval mode, so this matches the original.
            out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2],
                                                 attn_mask=attn_bias)
            out = out.transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(out))

        dino_attention.Attention.forward = sdpa_forward
        dino_attention.MemEffAttention.forward = sdpa_forward

    def _forward_window(self, model, device, window):
        """One VDPP forward pass with bf16 + OOM-downsize fallbacks."""
        import contextlib
        import torch

        use_bf16 = (self.p["bf16"] and device == "cuda"
                    and torch.cuda.is_bf16_supported())
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
                    else contextlib.nullcontext())
        try:
            with torch.no_grad(), autocast:
                return model(window, downsize=self.p["downsize"]).float()
        except torch.OutOfMemoryError:
            if self.p["downsize"] or device != "cuda":
                raise
            # Not enough VRAM at full resolution: retry downsized and stay
            # downsized for the rest of the job so all chunks match.
            self.p["downsize"] = True
            self.log("Warning: GPU out of memory at full resolution; "
                     "retrying with 'Downsize' enabled for the rest of "
                     "this job.")
            torch.cuda.empty_cache()
            with torch.no_grad(), autocast:
                return model(window, downsize=True).float()

    def infer_chunk(self, model, device, frames):
        """Run VDPP on a list of float32 (H, W) frames; returns (S, H, W).

        Reimplements VDPP.infer_video_depth's sliding-window loop (window 32,
        overlap 4, scale/shift alignment between windows) but keeps the chunk
        input and output on the CPU, moving only one window to the GPU at a
        time — VRAM use is then bounded by the window, not the chunk size.
        """
        import torch
        import torch.nn.functional as TF
        from vdpp.vdpp_model import compute_scale_and_shift, make_multiple_of

        stack = np.stack(frames, axis=0)
        if self.p["normalization"] == "per-frame":
            mins = stack.min(axis=(1, 2), keepdims=True)
            maxs = stack.max(axis=(1, 2), keepdims=True)
            stack = (stack - mins) / np.maximum(maxs - mins, 1e-6)
        else:  # per-chunk
            lo, hi = stack.min(), stack.max()
            stack = (stack - lo) / max(hi - lo, 1e-6)

        total, height, width = stack.shape
        window_size = model.num_frames            # 32
        overlap = model.infer_overlap_size        # 4
        r_height = make_multiple_of(height, 14)
        r_width = make_multiple_of(width, 14)
        needs_resize = (r_height, r_width) != (height, width)

        outputs = []
        prev_tail = None  # last `overlap` aligned output frames, on device
        start, end = 0, min(window_size, total)
        while True:
            self.check_cancel()
            window = torch.from_numpy(stack[start:end]).unsqueeze(0).to(device)
            if needs_resize:
                # (1, S, H, W): S acts as channels, so this resizes each
                # frame spatially — identical to the original per-frame view.
                window = TF.interpolate(window, size=(r_height, r_width),
                                        mode="bilinear", align_corners=True)
            out = self._forward_window(model, device, window)

            if prev_tail is None:
                keep = out
            else:
                ones = torch.ones_like(prev_tail.flatten(1, 2))
                scale, shift = compute_scale_and_shift(
                    out[:, :overlap].flatten(1, 2),
                    prev_tail.flatten(1, 2), ones)
                out = out * scale.view(1, 1, 1, 1) + shift.view(1, 1, 1, 1)
                keep = out[:, overlap:]
            prev_tail = out[:, -overlap:].clone()

            if needs_resize:
                keep = TF.interpolate(keep, size=(height, width),
                                      mode="bilinear", align_corners=True)
            outputs.append(keep.squeeze(0).cpu().numpy())
            del window, out, keep
            if end >= total:
                break
            start += window_size - overlap
            end = min(start + window_size, total)

        del prev_tail
        return np.concatenate(outputs, axis=0)

    # -- per-scene processing ---------------------------------------------------

    def process_scene(self, model, device, taker, scene_len, buffer,
                      processed_offset, total_frames):
        """Consume scene_len frames (or all remaining if None) from taker.

        Consecutive chunks share CHUNK_OVERLAP input frames. The next chunk's
        output is scale/shift-aligned to the previous one over that overlap,
        and the overlap frames themselves are crossfaded between the two
        predictions so any residual mismatch is spread over the whole overlap
        instead of popping at a single frame.
        """
        max_chunk = self.p["max_chunk"]
        prev_in_tail = []      # input frames shared with the previous chunk
        pending = None         # previous chunk's output for those frames,
                               # withheld from the buffer until blended
        done = 0
        while scene_len is None or done < scene_len:
            self.check_cancel()
            want = max_chunk - len(prev_in_tail)
            if scene_len is not None:
                want = min(want, scene_len - done)
            new_frames = taker.take(want)
            if not new_frames:
                if scene_len is not None and done < scene_len:
                    self.log(f"Warning: input ended {scene_len - done} frames "
                             "early for this scene.")
                break
            chunk_frames = prev_in_tail + new_frames
            output = self.infer_chunk(model, device, chunk_frames)

            if pending is not None:
                overlap = len(prev_in_tail)
                scale, shift = fit_scale_shift(output[:overlap], pending)
                output = output * scale + shift
                weights = np.linspace(0.0, 1.0, overlap + 2,
                                      dtype=np.float32)[1:-1, None, None]
                buffer.add(pending * (1.0 - weights) + output[:overlap] * weights)
                body = output[overlap:]
            else:
                body = output

            done += len(new_frames)
            still_going = ((scene_len is None or done < scene_len)
                           and not taker.exhausted)
            if still_going:
                keep = min(CHUNK_OVERLAP, len(chunk_frames))
                prev_in_tail = chunk_frames[-keep:]
                # copy() so the tail does not pin the whole chunk in memory
                pending = body[-keep:].copy()
                buffer.add(body[:-keep].copy())
            else:
                buffer.add(body)
                pending = None
            self.progress("process", processed_offset + done, total_frames)
            del output, body, chunk_frames, new_frames
        if pending is not None:  # input ended early; flush withheld frames
            buffer.add(pending)
        return done

    # -- orchestration -----------------------------------------------------------

    def run(self):
        source = self.open_source()
        scenes = self.detect_scenes(source)
        model, device = self.load_vdpp()

        fps = self.p["fps"] if self.p["fps"] > 0 else (source.fps or 30.0)
        if self.p["output_format"] == "png":
            writer = PngSequenceWriter(self.p["output_path"])
            self.log(f"Writing 16-bit grayscale PNGs to {writer.outdir}")
        else:
            writer = VideoWriter10Bit(self.p["output_path"], source.width,
                                      source.height, fps, crf=self.p["crf"])
            self.log(f"Encoding 10-bit MP4 ({writer.codec}, CRF {self.p['crf']}, "
                     f"{fps:.3f} fps) to {self.p['output_path']}")

        total = source.frame_count  # None only when detection was skipped
        written = 0
        taker = FrameTaker(source.frames(self.cancelled))
        tmpdir = tempfile.mkdtemp(prefix="vdpp_gui_")
        try:
            for i, (start, end) in enumerate(scenes):
                self.check_cancel()
                scene_len = None if end is None else end - start
                self.log(f"Scene {i + 1}/{len(scenes)}: "
                         + (f"frames {start}..{end - 1} ({scene_len} frames)"
                            if end is not None else "all frames"))
                buffer = SceneOutputBuffer(tmpdir, i)
                self.process_scene(model, device, taker, scene_len, buffer,
                                   processed_offset=start, total_frames=total)
                if buffer.frame_count == 0:
                    break

                def on_frame():
                    nonlocal written
                    written += 1
                    self.progress("write", written, total)

                buffer.finalize(writer, on_frame, self.cancelled)
                if taker.exhausted:
                    break
            writer.close()
        except BaseException:
            writer.close(abort=True)
            raise
        finally:
            taker.close()
            shutil.rmtree(tmpdir, ignore_errors=True)
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()

        if written == 0:
            raise RuntimeError("No frames were produced.")
        self.log(f"Done. {written} frames written to: {self.p['output_path']}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VDPP — Video Depth Post-Processing")
        self.minsize(640, 600)
        self.resizable(True, True)

        self.msg_queue = queue.Queue()
        self.worker = None
        self.cancelled = threading.Event()

        self._build_ui()
        self._load_settings()
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # Input
        in_frame = ttk.LabelFrame(root, text="Input (depth maps)")
        in_frame.pack(fill="x", **pad)
        self.input_var = tk.StringVar()
        ttk.Entry(in_frame, textvariable=self.input_var).pack(
            side="left", fill="x", expand=True, padx=(8, 4), pady=8)
        ttk.Button(in_frame, text="PNG Folder...",
                   command=self._browse_input_folder).pack(side="left", padx=2, pady=8)
        ttk.Button(in_frame, text="Video File...",
                   command=self._browse_input_video).pack(side="left", padx=(2, 8), pady=8)

        # Output
        out_frame = ttk.LabelFrame(root, text="Output")
        out_frame.pack(fill="x", **pad)

        fmt_row = ttk.Frame(out_frame)
        fmt_row.pack(fill="x", padx=8, pady=(8, 0))
        self.format_var = tk.StringVar(value="png")
        ttk.Radiobutton(fmt_row, text="16-bit grayscale PNG sequence",
                        variable=self.format_var, value="png",
                        command=self._on_format_change).pack(side="left")
        ttk.Radiobutton(fmt_row, text="10-bit grayscale MP4",
                        variable=self.format_var, value="mp4",
                        command=self._on_format_change).pack(side="left", padx=16)

        path_row = ttk.Frame(out_frame)
        path_row.pack(fill="x", padx=8, pady=8)
        self.output_var = tk.StringVar()
        ttk.Entry(path_row, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        self.output_browse_btn = ttk.Button(path_row, text="Browse...",
                                            command=self._browse_output)
        self.output_browse_btn.pack(side="left")

        # Options
        opt_frame = ttk.LabelFrame(root, text="Options")
        opt_frame.pack(fill="x", **pad)
        grid = ttk.Frame(opt_frame)
        grid.pack(fill="x", padx=8, pady=8)

        ttk.Label(grid, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        self.checkpoint_var = tk.StringVar(value=DEFAULT_CHECKPOINT)
        ttk.Entry(grid, textvariable=self.checkpoint_var).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=4)
        ttk.Button(grid, text="...", width=3,
                   command=self._browse_checkpoint).grid(row=0, column=4)

        ttk.Label(grid, text="Scene detection:").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        self.scene_var = tk.StringVar(value=SCENE_MODE_OMNISHOTCUT)
        ttk.Combobox(grid, textvariable=self.scene_var, width=20,
                     values=SCENE_MODES, state="readonly").grid(
            row=1, column=1, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(grid, text="Max chunk frames:").grid(
            row=1, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        self.chunk_var = tk.StringVar(value="256")
        ttk.Entry(grid, textvariable=self.chunk_var, width=6).grid(
            row=1, column=3, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(grid, text="Input normalization:").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        self.norm_var = tk.StringVar(value="per-frame")
        ttk.Combobox(grid, textvariable=self.norm_var, width=12,
                     values=("per-frame", "per-chunk"), state="readonly").grid(
            row=2, column=1, sticky="w", padx=4, pady=(6, 0))

        self.downsize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Downsize (faster, lower quality)",
                        variable=self.downsize_var).grid(
            row=2, column=2, columnspan=3, sticky="w", padx=8, pady=(6, 0))

        self.bf16_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="bf16 inference (saves VRAM, recommended)",
                        variable=self.bf16_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(grid, text="FPS (0 = auto/30):").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.fps_var = tk.StringVar(value="0")
        ttk.Entry(grid, textvariable=self.fps_var, width=8).grid(
            row=3, column=1, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(grid, text="CRF (MP4 quality):").grid(
            row=3, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        self.crf_var = tk.StringVar(value="16")
        ttk.Entry(grid, textvariable=self.crf_var, width=5).grid(
            row=3, column=3, sticky="w", padx=4, pady=(6, 0))

        grid.columnconfigure(1, weight=1)

        # Run / Cancel + progress
        run_row = ttk.Frame(root)
        run_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_row, text="Run VDPP", command=self._on_run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(run_row, text="Cancel",
                                     command=self._on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(run_row, textvariable=self.status_var).pack(side="left", padx=8)

        self.progressbar = ttk.Progressbar(root, mode="determinate")
        self.progressbar.pack(fill="x", **pad)

        # Log
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=10, wrap="word",
                                state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True, padx=(8, 0),
                           pady=8)

    # -- browse handlers -----------------------------------------------------

    def _browse_input_folder(self):
        path = filedialog.askdirectory(title="Select depth-map PNG sequence folder")
        if path:
            self.input_var.set(path)

    def _browse_input_video(self):
        path = filedialog.askopenfilename(
            title="Select depth-map video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                       ("All files", "*.*")])
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        if self.format_var.get() == "png":
            path = filedialog.askdirectory(title="Select output folder for PNG sequence")
        else:
            path = filedialog.asksaveasfilename(
                title="Save output MP4", defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.output_var.set(path)

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select VDPP checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_var.set(path)

    def _on_format_change(self):
        # If the current output path clearly matches the other format, clear it.
        path = self.output_var.get().strip()
        if not path:
            return
        is_mp4_path = path.lower().endswith(".mp4")
        if self.format_var.get() == "png" and is_mp4_path:
            self.output_var.set("")
        elif self.format_var.get() == "mp4" and not is_mp4_path:
            self.output_var.set("")

    # -- run / cancel ---------------------------------------------------------

    def _validate(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            raise ValueError("Please select an input folder or video file.")
        if not os.path.exists(input_path):
            raise ValueError(f"Input path does not exist:\n{input_path}")
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                raise ValueError(
                    f"Input file has unrecognized video extension '{ext}'.\n"
                    "Select a video file, or a folder for a PNG sequence.")

        output_format = self.format_var.get()
        output_path = self.output_var.get().strip()
        if not output_path:
            raise ValueError("Please select an output "
                             + ("folder." if output_format == "png" else "MP4 file."))
        if output_format == "mp4" and not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
            self.output_var.set(output_path)
        if output_format == "png" and os.path.isfile(output_path):
            raise ValueError("PNG output path must be a folder, not a file.")
        in_abs = os.path.abspath(input_path)
        out_abs = os.path.abspath(output_path)
        if output_format == "png" and os.path.isdir(in_abs) and in_abs == out_abs:
            raise ValueError("Output folder must differ from the input folder "
                             "(frames would be overwritten).")

        checkpoint = self.checkpoint_var.get().strip()
        if not os.path.isfile(checkpoint):
            raise ValueError(
                f"VDPP checkpoint not found:\n{checkpoint}\n\n"
                "Download it with:\nwget https://github.com/injun-baek/VDPP/"
                "releases/download/v1.0/vdpp.pth -O checkpoints/vdpp.pth")

        try:
            fps = float(self.fps_var.get() or 0)
            if fps < 0:
                raise ValueError
        except ValueError:
            raise ValueError("FPS must be a non-negative number (0 = auto).")
        try:
            crf = int(self.crf_var.get())
            if not 0 <= crf <= 51:
                raise ValueError
        except ValueError:
            raise ValueError("CRF must be an integer between 0 and 51.")
        try:
            max_chunk = int(self.chunk_var.get())
            if max_chunk < MIN_CHUNK_FRAMES:
                raise ValueError
        except ValueError:
            raise ValueError(f"Max chunk frames must be an integer >= "
                             f"{MIN_CHUNK_FRAMES}. Lower it to reduce VRAM "
                             "use, raise it for fewer chunk seams.")

        return {
            "input_path": input_path,
            "output_path": output_path,
            "output_format": output_format,
            "checkpoint": checkpoint,
            "scene_mode": self.scene_var.get(),
            "max_chunk": max_chunk,
            "normalization": self.norm_var.get(),
            "downsize": self.downsize_var.get(),
            "bf16": self.bf16_var.get(),
            "fps": fps,
            "crf": crf,
        }

    def _on_run(self):
        try:
            params = self._validate()
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self._save_settings()
        self.cancelled.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_var.set("Running...")
        self._log_clear()

        self.worker = threading.Thread(target=self._worker_main,
                                       args=(params,), daemon=True)
        self.worker.start()

    def _on_cancel(self):
        self.cancelled.set()
        self.status_var.set("Cancelling...")

    def _worker_main(self, params):
        q = self.msg_queue
        try:
            job = Job(
                params,
                log=lambda msg: q.put(("log", msg)),
                progress=lambda phase, cur, total: q.put(("progress", phase, cur, total)),
                cancelled=self.cancelled,
            )
            job.run()
            q.put(("done", "Finished successfully."))
        except CancelledError:
            q.put(("done", "Cancelled by user."))
        except Exception:
            q.put(("error", traceback.format_exc()))

    # -- queue polling / UI updates -------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "progress":
                    _, phase, cur, total = msg
                    label = {"detect": "Detecting scenes",
                             "process": "Running VDPP",
                             "write": "Writing output"}.get(phase, phase)
                    if total:
                        if self.progressbar["mode"] != "determinate":
                            self.progressbar.stop()
                            self.progressbar.configure(mode="determinate")
                        self.progressbar.configure(maximum=total, value=cur)
                        self.status_var.set(f"{label}  ({cur}/{total})")
                    else:
                        if self.progressbar["mode"] != "indeterminate":
                            self.progressbar.configure(mode="indeterminate")
                            self.progressbar.start(12)
                        self.status_var.set(f"{label}  ({cur})")
                elif kind in ("done", "error"):
                    self.progressbar.stop()
                    self.progressbar.configure(mode="determinate", value=0)
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if kind == "error":
                        self._log(msg[1])
                        self.status_var.set("Error")
                        messagebox.showerror(
                            "Processing failed",
                            msg[1].strip().splitlines()[-1])
                    else:
                        self._log(msg[1])
                        self.status_var.set(msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_clear(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # -- settings persistence --------------------------------------------------

    def _save_settings(self):
        settings = {
            "input_path": self.input_var.get(),
            "output_path": self.output_var.get(),
            "output_format": self.format_var.get(),
            "checkpoint": self.checkpoint_var.get(),
            "scene_mode": self.scene_var.get(),
            "max_chunk": self.chunk_var.get(),
            "normalization": self.norm_var.get(),
            "downsize": self.downsize_var.get(),
            "bf16": self.bf16_var.get(),
            "fps": self.fps_var.get(),
            "crf": self.crf_var.get(),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except OSError:
            pass

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.input_var.set(s.get("input_path", ""))
        self.output_var.set(s.get("output_path", ""))
        self.format_var.set(s.get("output_format", "png"))
        self.checkpoint_var.set(s.get("checkpoint", DEFAULT_CHECKPOINT))
        scene_mode = s.get("scene_mode", SCENE_MODE_OMNISHOTCUT)
        self.scene_var.set(scene_mode if scene_mode in SCENE_MODES
                           else SCENE_MODE_OMNISHOTCUT)
        self.chunk_var.set(str(s.get("max_chunk", "256")))
        norm = s.get("normalization", "per-frame")
        self.norm_var.set("per-chunk" if norm == "global" else norm)
        self.downsize_var.set(bool(s.get("downsize", False)))
        self.bf16_var.set(bool(s.get("bf16", True)))
        self.fps_var.set(str(s.get("fps", "0")))
        self.crf_var.set(str(s.get("crf", "16")))

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(
                    "Quit", "Processing is still running. Quit anyway?"):
                return
            self.cancelled.set()
        self.destroy()


def main():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True,
                       creationflags=SUBPROCESS_FLAGS)
    except FileNotFoundError:
        tk.Tk().withdraw()
        messagebox.showerror(
            "FFmpeg not found",
            "FFmpeg was not found in PATH. It is required for video input "
            "and 10-bit MP4 output.\nInstall it from https://ffmpeg.org and "
            "make sure 'ffmpeg' is on your PATH.")
        return
    App().mainloop()


if __name__ == "__main__":
    main()
