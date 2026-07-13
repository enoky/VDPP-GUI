"""VDPP GUI — temporally consistent depth post-processing for depth-map sequences.

Runs the VDPP model on an existing depth-map input (16/8-bit grayscale PNG
sequence or a depth video file) and writes the refined result as either a
16-bit grayscale PNG sequence or a 10-bit grayscale MP4 (via FFmpeg).

Unlike run_video.py, this tool does NOT run an image-to-depth model: the
input is assumed to already be depth maps, which are fed directly into VDPP.

Usage:
    python app.py
"""

import json
import os
import queue
import re
import subprocess
import sys
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

# Hide console windows spawned by ffmpeg/ffprobe on Windows.
SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


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


def read_video_frames(path, width, height, frame_count, progress, cancelled):
    """Decode a video to float32 grayscale frames in [0, 1] via an FFmpeg pipe.

    Decoding at gray16le preserves the full precision of 10/12/16-bit
    depth videos (OpenCV's VideoCapture would clamp them to 8-bit).
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=SUBPROCESS_FLAGS)
    frame_bytes = width * height * 2
    frames = []
    try:
        while True:
            if cancelled.is_set():
                proc.kill()
                raise CancelledError()
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint16).reshape(height, width)
            frames.append(frame.astype(np.float32) / 65535.0)
            progress(len(frames), frame_count)
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace").strip()
        if proc.wait() != 0 and not frames:
            raise RuntimeError(f"ffmpeg failed to decode {path}:\n{stderr}")
    finally:
        if proc.poll() is None:
            proc.kill()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


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


# ---------------------------------------------------------------------------
# Input loading
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


class CancelledError(Exception):
    pass


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

    # -- loading ------------------------------------------------------------

    def load_frames(self):
        """Return (frames list[float32 HxW in 0..1], detected_fps_or_None)."""
        indir = self.p["input_path"]
        if os.path.isdir(indir):
            paths = list_image_sequence(indir)
            if not paths:
                raise RuntimeError(f"No image files found in {indir}")
            self.log(f"Loading {len(paths)} depth frames from image sequence...")
            frames = []
            shape = None
            for i, path in enumerate(paths):
                self.check_cancel()
                frame = load_image_as_depth(path)
                if shape is None:
                    shape = frame.shape
                elif frame.shape != shape:
                    raise RuntimeError(
                        f"Frame size mismatch: {os.path.basename(path)} is "
                        f"{frame.shape[1]}x{frame.shape[0]}, expected "
                        f"{shape[1]}x{shape[0]}")
                frames.append(frame)
                self.progress("load", i + 1, len(paths))
            return frames, None
        else:
            width, height, fps, count = ffprobe_video(indir)
            self.log(f"Decoding video {os.path.basename(indir)} "
                     f"({width}x{height} @ {fps:.3f} fps)...")
            frames = read_video_frames(
                indir, width, height, count,
                lambda cur, total: self.progress("load", cur, total),
                self.cancelled)
            self.log(f"Decoded {len(frames)} frames.")
            return frames, fps

    # -- inference ----------------------------------------------------------

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

    def run_vdpp(self, frames):
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
                     "using standard PyTorch attention.")
            sys.modules["xformers"] = None
            sys.modules["xformers.ops"] = None

        from vdpp.vdpp_model import VDPP

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
        model = model.to(device).eval()
        self.check_cancel()

        stack = np.stack(frames, axis=0)  # (S, H, W) float32
        del frames

        if self.p["normalization"] == "per-frame":
            mins = stack.min(axis=(1, 2), keepdims=True)
            maxs = stack.max(axis=(1, 2), keepdims=True)
            stack = (stack - mins) / np.maximum(maxs - mins, 1e-6)
        else:
            d_min, d_max = stack.min(), stack.max()
            stack = (stack - d_min) / max(d_max - d_min, 1e-6)

        tensor = torch.from_numpy(stack).unsqueeze(0).to(device)
        del stack

        self.log(f"Running VDPP on {tensor.shape[1]} frames "
                 f"({tensor.shape[3]}x{tensor.shape[2]})... this may take a while.")
        self.progress("infer", 0, None)
        with torch.no_grad():
            output = model.infer_video_depth(tensor, downsize=self.p["downsize"])
            output = (output - output.min()) / (output.max() - output.min() + 1e-6)
        result = output.squeeze(0).clamp(0, 1).cpu().numpy()

        del tensor, output, model
        if device == "cuda":
            torch.cuda.empty_cache()
        return result

    # -- output -------------------------------------------------------------

    def write_output(self, depths, fps):
        depths_u16 = np.round(depths * 65535.0).astype(np.uint16)
        total = depths_u16.shape[0]

        if self.p["output_format"] == "png":
            outdir = self.p["output_path"]
            os.makedirs(outdir, exist_ok=True)
            self.log(f"Writing {total} 16-bit grayscale PNGs to {outdir}...")
            for i in range(total):
                self.check_cancel()
                out_path = os.path.join(outdir, f"{i:08d}.png")
                if not cv2.imwrite(out_path, depths_u16[i]):
                    raise RuntimeError(f"Failed to write {out_path}")
                self.progress("write", i + 1, total)
            return outdir
        else:
            out_path = self.p["output_path"]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            height, width = depths_u16.shape[1:]
            writer = VideoWriter10Bit(out_path, width, height, fps,
                                      crf=self.p["crf"])
            self.log(f"Encoding 10-bit MP4 ({writer.codec}, CRF "
                     f"{self.p['crf']}) to {out_path}...")
            try:
                for i in range(total):
                    self.check_cancel()
                    writer.write(depths_u16[i])
                    self.progress("write", i + 1, total)
                writer.close()
            except BaseException:
                writer.close(abort=True)
                raise
            return out_path

    # -- orchestration -------------------------------------------------------

    def run(self):
        frames, detected_fps = self.load_frames()
        result = self.run_vdpp(frames)
        self.check_cancel()
        fps = self.p["fps"] if self.p["fps"] > 0 else (detected_fps or 30.0)
        out = self.write_output(result, fps)
        self.log(f"Done. Output saved to: {out}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VDPP — Video Depth Post-Processing")
        self.minsize(620, 560)
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

        ttk.Label(grid, text="Input normalization:").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        self.norm_var = tk.StringVar(value="per-frame")
        norm_box = ttk.Combobox(grid, textvariable=self.norm_var, width=12,
                                values=("per-frame", "global"), state="readonly")
        norm_box.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))

        self.downsize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Downsize (faster, lower quality)",
                        variable=self.downsize_var).grid(
            row=1, column=2, columnspan=3, sticky="w", padx=8, pady=(6, 0))

        ttk.Label(grid, text="FPS (0 = auto/30):").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        self.fps_var = tk.StringVar(value="0")
        ttk.Entry(grid, textvariable=self.fps_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(grid, text="CRF (MP4 quality):").grid(
            row=2, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        self.crf_var = tk.StringVar(value="16")
        ttk.Entry(grid, textvariable=self.crf_var, width=5).grid(
            row=2, column=3, sticky="w", padx=4, pady=(6, 0))

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

        return {
            "input_path": input_path,
            "output_path": output_path,
            "output_format": output_format,
            "checkpoint": checkpoint,
            "normalization": self.norm_var.get(),
            "downsize": self.downsize_var.get(),
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
                    if total:
                        if self.progressbar["mode"] != "determinate":
                            self.progressbar.stop()
                            self.progressbar.configure(mode="determinate")
                        self.progressbar.configure(maximum=total, value=cur)
                        self.status_var.set(
                            {"load": "Loading input", "infer": "Running VDPP",
                             "write": "Writing output"}.get(phase, phase)
                            + f"  ({cur}/{total})")
                    else:
                        if self.progressbar["mode"] != "indeterminate":
                            self.progressbar.configure(mode="indeterminate")
                            self.progressbar.start(12)
                        self.status_var.set("Running VDPP inference...")
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
            "normalization": self.norm_var.get(),
            "downsize": self.downsize_var.get(),
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
        self.norm_var.set(s.get("normalization", "per-frame"))
        self.downsize_var.set(bool(s.get("downsize", False)))
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
