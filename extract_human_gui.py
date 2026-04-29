"""
extract_human_gui.py
────────────────────
GUI app to extract humans from video with alpha channel output.
Pick your input video and output folder via file dialogs — no command line needed.

Requirements:
    pip install mediapipe opencv-python-headless numpy tqdm

Run:
    python extract_human_gui.py
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Core extraction logic (runs in a background thread)
# ──────────────────────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    ksize = radius * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def run_extraction(input_path, output_dir, threshold, feather,
                   make_mov, on_progress, on_log, on_done):
    """Runs in a background thread. Calls callbacks to update the GUI."""
    mp_selfie = mp.solutions.selfie_segmentation

    cap = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    on_log(f"📹  {input_path.name}  ({w}×{h} @ {fps:.2f} fps,  {total} frames)")
    on_log(f"📂  Output → {output_dir}")
    on_log(f"🎛️   Threshold={threshold}  Feather={feather}px\n")

    digit_count = len(str(total))

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result   = seg.process(rgb)
            raw_mask = result.segmentation_mask

            alpha_f  = np.where(raw_mask >= threshold, raw_mask, 0.0)
            alpha_f  = feather_mask(alpha_f, feather)
            alpha_u8 = (alpha_f * 255).clip(0, 255).astype(np.uint8)

            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha_u8

            name = f"frame_{str(i).zfill(digit_count)}.png"
            cv2.imwrite(str(output_dir / name), bgra)

            on_progress(i + 1, total)

    cap.release()
    on_log(f"✅  {total} PNG frames saved.\n")

    if make_mov:
        mov_path      = output_dir / (input_path.stem + "_alpha.mov")
        frame_pattern = str(output_dir / f"frame_%0{digit_count}d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-vendor", "apl0",
            str(mov_path),
        ]
        on_log("🎬  Compiling ProRes 4444 .mov …")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            on_log(f"✅  .mov saved → {mov_path}")
        else:
            on_log("⚠️  ffmpeg not found or failed. Use the PNG sequence directly.")

    on_done()


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Human Extractor — Alpha Channel")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 18
        BG  = "#1a1a2e"
        CARD= "#16213e"
        ACC = "#e94560"
        FG  = "#eaeaea"
        SUB = "#8892a4"
        FNT = ("Helvetica", 11)

        # ── title bar ──
        hdr = tk.Frame(self, bg=ACC, height=4)
        hdr.pack(fill="x")

        tk.Label(self, text="HUMAN EXTRACTOR", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=FG).pack(pady=(PAD, 2))
        tk.Label(self, text="Extract people from video · alpha channel output",
                 font=("Helvetica", 9), bg=BG, fg=SUB).pack(pady=(0, PAD))

        # ── card wrapper ──
        card = tk.Frame(self, bg=CARD, padx=PAD, pady=PAD, relief="flat")
        card.pack(padx=PAD, pady=(0, 8), fill="x")

        def row(parent, label, row_n):
            tk.Label(parent, text=label, font=FNT, bg=CARD, fg=SUB,
                     anchor="w", width=14).grid(row=row_n, column=0,
                     sticky="w", pady=5)

        # Input file
        row(card, "Input video", 0)
        self.input_var = tk.StringVar()
        e1 = tk.Entry(card, textvariable=self.input_var, width=42,
                      bg="#0f3460", fg=FG, insertbackground=FG,
                      relief="flat", font=FNT, bd=6)
        e1.grid(row=0, column=1, padx=(6, 6), pady=5, sticky="ew")
        tk.Button(card, text="Browse…", command=self._pick_input,
                  bg=ACC, fg="white", relief="flat", font=FNT,
                  cursor="hand2", padx=8).grid(row=0, column=2, pady=5)

        # Output folder
        row(card, "Output folder", 1)
        self.output_var = tk.StringVar()
        e2 = tk.Entry(card, textvariable=self.output_var, width=42,
                      bg="#0f3460", fg=FG, insertbackground=FG,
                      relief="flat", font=FNT, bd=6)
        e2.grid(row=1, column=1, padx=(6, 6), pady=5, sticky="ew")
        tk.Button(card, text="Browse…", command=self._pick_output,
                  bg=ACC, fg="white", relief="flat", font=FNT,
                  cursor="hand2", padx=8).grid(row=1, column=2, pady=5)

        # Separator
        tk.Frame(card, bg="#0f3460", height=1).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=10)

        # Threshold slider
        row(card, "Threshold", 3)
        self.thresh_var = tk.DoubleVar(value=0.6)
        sl1 = tk.Scale(card, variable=self.thresh_var, from_=0.1, to=0.95,
                       resolution=0.05, orient="horizontal", length=220,
                       bg=CARD, fg=FG, troughcolor="#0f3460",
                       highlightthickness=0, font=("Helvetica", 9))
        sl1.grid(row=3, column=1, sticky="w", padx=(6, 0))
        tk.Label(card, text="Lower = include more", font=("Helvetica", 8),
                 bg=CARD, fg=SUB).grid(row=3, column=2, sticky="w", padx=6)

        # Feather slider
        row(card, "Feather (px)", 4)
        self.feather_var = tk.IntVar(value=8)
        sl2 = tk.Scale(card, variable=self.feather_var, from_=0, to=30,
                       resolution=1, orient="horizontal", length=220,
                       bg=CARD, fg=FG, troughcolor="#0f3460",
                       highlightthickness=0, font=("Helvetica", 9))
        sl2.grid(row=4, column=1, sticky="w", padx=(6, 0))
        tk.Label(card, text="Softer edge blur", font=("Helvetica", 8),
                 bg=CARD, fg=SUB).grid(row=4, column=2, sticky="w", padx=6)

        # Make .mov checkbox
        self.mov_var = tk.BooleanVar(value=False)
        tk.Checkbutton(card, text="Also export ProRes 4444 .mov  (needs ffmpeg)",
                       variable=self.mov_var, bg=CARD, fg=SUB,
                       selectcolor=CARD, activebackground=CARD,
                       font=("Helvetica", 9)).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # ── progress bar ──
        prog_frame = tk.Frame(self, bg=BG)
        prog_frame.pack(padx=PAD, fill="x", pady=(4, 0))

        self.progress = ttk.Progressbar(prog_frame, length=460, mode="determinate")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#0f3460",
                         background=ACC, thickness=10)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var, font=("Helvetica", 9),
                 bg=BG, fg=SUB).pack(anchor="w", padx=PAD)

        # ── log box ──
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(padx=PAD, pady=(6, 0), fill="both")

        self.log = tk.Text(log_frame, height=9, bg="#0a0a1a", fg="#6ee7b7",
                           font=("Courier", 9), relief="flat",
                           state="disabled", bd=0)
        self.log.pack(fill="both")

        # ── run button ──
        self.run_btn = tk.Button(
            self, text="▶  EXTRACT HUMAN",
            command=self._start,
            bg=ACC, fg="white", font=("Helvetica", 13, "bold"),
            relief="flat", cursor="hand2", padx=20, pady=12,
            activebackground="#c73652", activeforeground="white")
        self.run_btn.pack(pady=PAD)

    # ── file pickers ──────────────────────────────────────────────────────────

    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                       ("All files", "*.*")])
        if path:
            self.input_var.set(path)
            # Auto-suggest output folder next to video
            if not self.output_var.get():
                suggested = str(Path(path).parent / (Path(path).stem + "_frames"))
                self.output_var.set(suggested)

    def _pick_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # ── extraction trigger ────────────────────────────────────────────────────

    def _start(self):
        inp = self.input_var.get().strip()
        out = self.output_var.get().strip()

        if not inp or not Path(inp).exists():
            messagebox.showerror("Missing input", "Please select a valid video file.")
            return
        if not out:
            messagebox.showerror("Missing output", "Please select an output folder.")
            return

        output_dir = Path(out)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.run_btn.config(state="disabled", text="Processing…")
        self.progress["value"] = 0
        self._log_clear()

        threading.Thread(
            target=run_extraction,
            args=(
                Path(inp), output_dir,
                self.thresh_var.get(),
                self.feather_var.get(),
                self.mov_var.get(),
                self._on_progress,
                self._on_log,
                self._on_done,
            ),
            daemon=True
        ).start()

    # ── callbacks (called from background thread → schedule on main thread) ──

    def _on_progress(self, current, total):
        pct = (current / total) * 100
        self.after(0, lambda: self._update_progress(pct, current, total))

    def _update_progress(self, pct, current, total):
        self.progress["value"] = pct
        self.status_var.set(f"Frame {current} / {total}  ({pct:.1f}%)")

    def _on_log(self, msg):
        self.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _on_done(self):
        self.after(0, self._finish)

    def _finish(self):
        self.run_btn.config(state="normal", text="▶  EXTRACT HUMAN")
        self.status_var.set("✅  Done! Import the frames folder into your video editor.")
        messagebox.showinfo("Done!", f"Frames saved to:\n{self.output_var.get()}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick dependency check
    missing = []
    for pkg in ("cv2", "mediapipe", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        root = tk.Tk()
        root.withdraw()
        pkg_names = {"cv2": "opencv-python-headless", "mediapipe": "mediapipe", "numpy": "numpy"}
        install_str = " ".join(pkg_names[p] for p in missing)
        messagebox.showerror(
            "Missing packages",
            f"Please install missing packages first:\n\n"
            f"pip install {install_str}"
        )
        sys.exit(1)

    app = App()
    app.mainloop()
