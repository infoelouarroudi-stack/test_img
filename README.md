# DeepAnimDance — Posture-guided Image Synthesis

**Project: Everybody Dance Now - Mini Implementation**

---

## 1. Context & Objectives

This project implements **posture-guided image synthesis**: transferring motion (pose sequences) from a **source video** to a **target person** learned from a target video dataset.

- **Pose extraction**: Using **MediaPipe** (provided in starter code)
- **Generation strategies**: Multiple approaches from baseline to GAN-based synthesis
- **Goal**: Synthesize realistic target frames driven by source poses

### Submission Requirements
- **ZIP file** containing:
  - All source code
  - Data files
  - Trained network weights (`.pth` files)
  - ~2 minute demo video
- **No separate report**: All explanations must be in this `README.md` (English or French accepted)

---

## 2. Repository Structure

```
project-root/
├── src/                          # All Python source files
│   ├── DanceDemo.py              # Main demo application
│   ├── VideoSkeleton.py          # Skeleton extraction & dataset
│   ├── GenNeirest.py             # Nearest neighbor baseline
│   ├── GenVanillaNN.py           # Vanilla neural network generator
│   ├── GenGAN.py                 # GAN-based generator
│   ├── VideoReader.py            # Video utilities
│   └── Vec3.py                   # Helper classes
│
├── data/                         # Videos, cache, and trained models
│   ├── Dance/                    # Trained model weights
│   │   ├── DanceGenGAN.pth
│   │   ├── DanceGenVanillaFromSke26.pth
│   │   └── DanceGenVanillaFromSkeim.pth
│   ├── taichi1.mp4               # Target dataset video
│   ├── taichi1.pkl               # Precomputed skeleton cache
│   ├── taichi1/                  # Extracted frames directory
│   ├── taichi2.mp4               # Source video example
│   └── karate1.mp4               # Additional source video
│
├── requirements.txt              # Python dependencies (pip)
├── environment.yml               # Conda environment file
└── README.md                     # This file
```

---

## 3. Installation

### Option A: Virtual Environment (pip)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate <ENV_NAME>
```

### Required Packages
- `numpy`
- `opencv-python`
- `torch`
- `torchvision`
- `mediapipe`
- `Pillow`
- `tensorboardX`

---

## 4. Running the Demo

The main program `src/DanceDemo.py` displays **three panels**:
- **SOURCE VIDEO** | **SKELETON** | **GENERATION**
- FPS counter in real-time

**Keyboard controls:**
- `q` — Quit the demo
- `n` — Skip ~100 frames forward

### 4.1 Running from Repository Root

⚠️ **Important**: Always run commands from the repository root directory, as the code uses relative paths like `data/Dance/...`

Since imports are written as `from VideoSkeleton import ...`, you must add `src/` to `PYTHONPATH`.

#### Linux / macOS
```bash
PYTHONPATH=src python src/DanceDemo.py
```

#### Windows (PowerShell)
```powershell
$env:PYTHONPATH="src"
python src\DanceDemo.py
```

### 4.2 Selecting Generation Method

In `src/DanceDemo.py`, set the `GEN_TYPE` variable to choose the generator:

| `GEN_TYPE` | Method | Description |
|------------|--------|-------------|
| `1` | Nearest Neighbor | Baseline (`GenNeirest`) |
| `2` | Vanilla NN (26D) | 26D skeleton vector → image |
| `3` | Vanilla NN (Image) | Stickman image → image |
| `4` | GAN | WGAN-GP + L1 loss |

### 4.3 Changing Source Video

The demo reads poses from the source video (default: `data/taichi2.mp4`) and applies them to the target dataset (built from `data/taichi1.mp4`).

To change the source video, edit the last lines of `DanceDemo.py`:
```python
DanceDemo("data/karate1.mp4", GEN_TYPE)  # Example: use karate video
```

---

## 5. Pipeline Overview

### 5.1 Pose Extraction & Caching (`VideoSkeleton`)

`VideoSkeleton` builds a dataset of (skeleton, image) pairs from the **target video**.

**Caching mechanism:**
- Saves skeleton data to `.pkl` file
- Stores extracted frames to disk
- Subsequent runs load precomputed data (unless `forceCompute=True`)

### 5.2 Demo Visualization (`DanceDemo`)

For each source frame:
1. **Crop** using target crop logic
2. **Draw** skeleton visualization panel
3. **Generate** target frame via `generator.generate(ske)`

**Performance notes:**
- Processes 1 frame out of 5 by default (configurable)
- Use `_ensure_uint8_bgr` helper to standardize generator outputs

---

## 6. Methods Implemented

Progressive implementation following the assignment stages: baseline → direct NN → stickman NN → GAN improvement.

### 6.1 Method 1: Nearest Neighbor (Baseline)

**Class**: `GenNeirest`

**Approach**: Searches the target dataset for the closest skeleton using joint distance metrics and returns the corresponding target image.

**Pros:**
- Always returns real images from dataset
- No training required

**Cons:**
- Slow search for large datasets
- No temporal continuity
- Poor generalization to unseen poses

---

### 6.2 Method 2: Vanilla NN (26D Skeleton → Image)

**Class**: `GenVanillaNN` (with `optSkeOrImage=1`)

**Approach**: 
- Input: Reduced skeleton representation (13 joints × 2D = 26 values)
- Output: 64×64 RGB image
- Direct mapping learned by neural network

**Dataset**: Outputs `(skeleton_tensor, target_image_tensor)` with proper normalization.

---

### 6.3 Method 3: Vanilla NN (Stickman Image → Image)

**Class**: `GenVanillaNN` (with `optSkeOrImage=2`)

**Approach**:
- Skeleton rendered as "stickman" image using `SkeToImageTransform`
- CNN learns: stickman image → target image
- Intermediate representation closer to paper's approach

**Benefits**: Generally improves learning compared to raw coordinates.

---

### 6.4 Method 4: GAN (WGAN-GP + L1)

**Class**: `GenGAN`

**Architecture**:
- Generator: Stickman input → target image
- Discriminator: PatchGAN-like critic
- Training: WGAN-GP (Gradient Penalty) for stability

**Training features:**
- Multiple discriminator steps (`ncritic`) per generator step
- **L1 reconstruction loss** (weighted) to preserve structure
- Gradient penalty for Wasserstein distance enforcement

**Inference**: Skeleton → stickman → generator → denormalized output image

---

## 7. Training the Networks

All training uses the target dataset built from `data/taichi1.mp4`.

### 7.1 (Optional) Rebuild Skeleton Cache

If needed, delete cache files:
```bash
rm data/taichi1.pkl
rm -rf data/taichi1/
```

Next run will rebuild the cache automatically (or set `forceCompute=True` in `VideoSkeleton`).

---

### 7.2 Train Vanilla NN

Open `src/GenVanillaNN.py` and enable training mode in the `__main__` section (set `train=True` and configure `nepoch`).

**Linux/macOS:**
```bash
PYTHONPATH=src python src/GenVanillaNN.py data/taichi1.mp4
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"
python src\GenVanillaNN.py data\taichi1.mp4
```

**Output weights:**
- `data/Dance/DanceGenVanillaFromSke26.pth` (26D skeleton input)
- `data/Dance/DanceGenVanillaFromSkeim.pth` (stickman image input)

---

### 7.3 Train GAN

**Linux/macOS:**
```bash
PYTHONPATH=src python src/GenGAN.py data/taichi1.mp4
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"
python src\GenGAN.py data\taichi1.mp4
```

**Training configuration:**
- Default: 50 epochs (configurable in main)
- Output: `data/Dance/DanceGenGAN.pth`

---

## 8. Demo Video Requirements

Record a **~2 minute video** showing:

1. Running `DanceDemo.py` from repository root
2. Switching between `GEN_TYPE` values (demonstrate at least one NN method + GAN)
3. Real-time window display:
   - Three panels: `SOURCE | SKELETON | GENERATION`
   - FPS counter
   - Keyboard controls (`q` to quit, `n` to skip)

---

## 9. Credits & Notes

**Based on:**
- Paper: "Everybody Dance Now" (Chan et al., ICCV 2019)
- TP assignment with MediaPipe integration for skeleton extraction

**Author**: `<YOUR NAME>`  
**Date**: `<DATE>`

---

## 10. Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure `PYTHONPATH=src` is set when running from repository root

**Issue**: Models not loading
- **Solution**: Verify `.pth` files exist in `data/Dance/` directory

**Issue**: Slow performance
- **Solution**: Reduce frame processing rate in `DanceDemo.py` (default: 1 out of 5 frames)

**Issue**: Cache rebuild needed
- **Solution**: Delete `.pkl` file and frames directory, then rerun

---

## 11. Future Improvements

Potential enhancements:
- Temporal consistency between frames
- Higher resolution output (beyond 64×64)
- Multi-person pose transfer
- Better discriminator architecture
- Perceptual loss integration
