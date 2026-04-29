# Human Extraction from Video (Alpha Channel Output)

A command-line tool to extract a human subject from a video using AI-based segmentation and export it with transparency (alpha channel).

This is useful for:

* Video compositing (placing subjects over new backgrounds)
* Content creation (Reels, Shorts, ads)
* VFX pipelines
* Rapid prototyping without green screens

---

## 🔧 Features

* AI-based human segmentation using MediaPipe
* Outputs **PNG image sequence with alpha channel**
* Optional export to **.mov (ProRes 4444 with alpha)**
* Adjustable segmentation threshold
* Edge feathering for smooth blending
* Live preview option during processing

---

## 📦 Requirements

Install dependencies:

```
pip install mediapipe opencv-python-headless numpy tqdm
```

Optional (for `.mov` export):

* Install `ffmpeg` and ensure it's available in system PATH

Check installation:

```
ffmpeg -version
```

---

## 🚀 Usage

Basic usage:

```
python extract_human.py input.mp4
```

---

## ⚙️ Arguments

| Argument     | Type   | Default  | Description                   |
| ------------ | ------ | -------- | ----------------------------- |
| input        | string | —        | Path to input video           |
| --output-dir | string | ./frames | Directory to save PNG frames  |
| --threshold  | float  | 0.6      | Segmentation confidence (0–1) |
| --feather    | int    | 8        | Edge blur radius (pixels)     |
| --mov        | flag   | False    | Export `.mov` with alpha      |
| --preview    | flag   | False    | Show live preview window      |

---

## 🧠 How It Works

1. Each frame is processed using MediaPipe Selfie Segmentation
2. A probability mask (0–1) is generated for human pixels
3. Thresholding isolates the subject
4. Gaussian blur is applied for edge feathering
5. Alpha channel is constructed
6. Frame is saved as BGRA PNG

---

## 📂 Output

### Option 1 — PNG Sequence (Default)

* Stored in:

  ```
  ./frames/frame_0001.png
  ./frames/frame_0002.png
  ...
  ```
* Each image contains transparency (alpha channel)

---

### Option 2 — MOV with Alpha

Enable with:

```
--mov
```

Output:

```
input.alpha.mov
```

Format:

* Codec: ProRes 4444
* Pixel Format: yuva444p10le (supports alpha)

---

## 🎛️ Parameter Tuning (Important)

### Threshold (`--threshold`)

* Lower (0.3–0.5): captures more detail but may include noise
* Higher (0.7–0.9): cleaner edges but may cut fine details

**Practical recommendation:** 0.55–0.7

---

### Feather (`--feather`)

* 0: hard edges (bad for compositing)
* 5–15: natural blending
* > 20: overly soft edges

**Practical recommendation:** 6–10

---

## 👁️ Preview Mode

```
--preview
```

* Displays live composited output over black
* Press `Q` to stop early

---

## 🎬 Importing into Video Editors

### Premiere Pro / DaVinci Resolve / Final Cut

#### PNG Sequence:

1. Import as image sequence
2. Place on timeline above background
3. Alpha is automatically respected

#### MOV:

1. Directly import `.mov`
2. Drop onto timeline
3. No keying required

---

## ⚠️ Limitations

* Not production-grade segmentation (no temporal consistency)
* Struggles with:

  * Fast motion blur
  * Hair-level detail
  * Occlusions
* Works best with:

  * Static camera
  * Clear subject-background separation
  * Good lighting

---

## 📈 Performance Notes

* CPU-bound (MediaPipe runs on CPU unless optimized)
* Processing time ≈ 0.5x to 1x video duration depending on hardware
* SSD recommended for frame writes

---

## 🛠️ Possible Improvements

If you're serious about turning this into a product or pipeline:

* Add temporal smoothing (reduce flicker)
* Replace MediaPipe with:

  * MODNet
  * Robust Video Matting (RVM)
* Add GPU acceleration
* Batch processing support
* GUI wrapper

---

## 🧩 Project Structure

```
.
├── extract_human.py
├── frames/               # Output PNGs
└── README.md
```
---

## ✅ Done

You now have a clean pipeline for extracting human subjects without green screen.

Use it, test edge cases, and move forward.
