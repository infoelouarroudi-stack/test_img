Voici un **README.md sous forme de mini-rapport** (prêt à copier-coller) adapté à ta structure (dossier `src/`, données dans `data/`, poids `.pth` déjà présents).[1][2]

```markdown
# DeepAnimDance — Posture-guided Image Synthesis (Everybody Dance Now - mini project)

## 1) Context & goal
This project implements **posture-guided image synthesis**: we transfer the motion (pose sequence) from a **source video** to a **target person** learned from a target video dataset. [file:402]  
The pose is extracted using **MediaPipe** (already provided in the starter code), then different generation strategies are used to synthesize target frames. [file:402]

### What you must upload (teacher requirements)
- A ZIP containing **all code + data needed** + **trained networks (.pth)** + a **~2 min demo video**. [file:402]  
- No separate report: all explanations must be inside this `README.md` (English or French). [file:402]

---

## 2) Repository structure (what is where)
From the repository root: [image:1]
- `src/`: all python source files (`DanceDemo.py`, `VideoSkeleton.py`, generators, etc.). [image:1]  
- `data/`: videos + cached skeleton/frames + trained models. [image:1]  
  - Trained weights are stored in `data/Dance/` (e.g. `DanceGenGAN.pth`, `DanceGenVanillaFromSke26.pth`, `DanceGenVanillaFromSkeim.pth`). [image:1][file:402]  
  - Target dataset video: `data/taichi1.mp4` and precomputed cache `data/taichi1.pkl` (plus extracted frames directory `data/taichi1/`). [image:1][file:404]  
  - Source videos examples: `data/taichi2.mp4`, `data/karate1.mp4`, etc. [image:1]

---

## 3) Installation (2 options)

### Option A — `venv` (pip)
```
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Option B — Conda (`environment.yml`)
```
conda env create -f environment.yml
conda activate <ENV_NAME>
```

> Notes:
> - You need packages such as: `numpy`, `opencv-python`, `torch`, `torchvision`, `mediapipe`, `Pillow`, `tensorboardX` (depending on your environment file / requirements). [file:403][file:160]

---

## 4) Run the demo (with trained networks)
The main program is `src/DanceDemo.py`: it displays **three panels**: `SOURCE VIDEO | SKELETON | GENERATION` and an FPS counter. [file:402]  
Keyboard controls: press `q` to quit, `n` to skip ~100 frames. [file:402]

### 4.1 Important: run from repo root (paths)
The code loads models from relative paths like `data/Dance/...` (so run commands **from the repository root**). [file:402]  
Because imports are written like `from VideoSkeleton import ...`, you must add `src/` to `PYTHONPATH` when running from root. [file:402]

#### Linux / macOS
```
PYTHONPATH=src python src/DanceDemo.py
```

#### Windows (PowerShell)
```
$env:PYTHONPATH="src"
python src\DanceDemo.py
```

### 4.2 Choose the generator (GEN_TYPE)
In `src/DanceDemo.py`, set `GEN_TYPE` then run the demo. [file:402]  
Available modes: [file:402]
- `1` — Nearest Neighbor baseline (`GenNeirest`)  
- `2` — Vanilla NN: **26D skeleton vector → image** (`GenVanillaNN(... optSkeOrImage=1)`)  
- `3` — Vanilla NN: **stickman image → image** (`GenVanillaNN(... optSkeOrImage=2)`)  
- `4` — GAN (WGAN-GP + L1) using `GenGAN(... loadFromFile=True)`  

### 4.3 Source video to test
The demo reads poses from the source video (example: `data/taichi2.mp4`) and applies them to the **target dataset** built from `data/taichi1.mp4`. [file:402]  
To change the source video, edit the last lines of `DanceDemo.py` (e.g. `DanceDemo("data/taichi2.mp4", GEN_TYPE)`). [file:402]

---

## 5) How it works (pipeline)
### 5.1 Pose extraction + caching (VideoSkeleton)
`VideoSkeleton` builds a dataset of pairs (skeleton, image) from the **target** video. [file:404]  
To avoid recomputing skeletons each time, it saves a `.pkl` cache and stores frames on disk; next runs will load the precomputed result unless `forceCompute=True`. [file:404]

### 5.2 Demo visualization (DanceDemo)
For each processed source frame, the code: (1) crops using the target crop logic, (2) draws a skeleton panel, (3) calls `generator.generate(ske)` to synthesize the target frame. [file:402]  
To keep real-time speed, the demo processes one frame out of 5 (you can change it to 1 for better smoothness but slower). [file:402]  
A helper function `_ensure_uint8_bgr` standardizes generator outputs (uint8 or float) before display. [file:402]

---

## 6) Methods implemented (and what was done)
This project follows the progressive stages suggested in the TP statement: baseline → direct NN → stickman NN → GAN improvement. [file:402]

### 6.1 Method 1 — Nearest Neighbor (baseline)
`GenNeirest` searches the target dataset for the closest skeleton (using a distance computed between skeleton joints) and returns the corresponding target image. [file:402]  
Pros: always returns a real image from the dataset. [file:402]  
Cons: slow search, no temporal continuity, poor generalization if pose is unseen. [file:402]

### 6.2 Method 2 — Vanilla NN (26D skeleton → image)
`GenVanillaNN` can take a reduced skeleton representation (13 joints in 2D → 26 values) and learn a mapping to a 64×64 image. [file:402]  
This uses a dataset class that outputs `(skeleton_tensor, target_image_tensor)` with normalization to the range expected by the network. [file:160][file:402]

### 6.3 Method 3 — Vanilla NN (stickman image → image)
Instead of raw coordinates, the skeleton is first rendered into a “stickman” image using `SkeToImageTransform`, then a CNN learns stickman → target image. [file:160][file:402]  
This intermediate representation is closer to the paper idea and generally helps learning compared to raw coordinates. [file:402]

### 6.4 Method 4 — GAN (WGAN-GP + L1)
`GenGAN` trains a generator and a PatchGAN-like critic/discriminator and uses **WGAN-GP** (gradient penalty) for stability. [file:402]  
Training uses multiple discriminator steps (`ncritic`) per generator step and adds an **L1 reconstruction loss** (weighted) to preserve structure. [file:402]  
At inference, the skeleton is converted to stickman input and the output tensor is denormalized back to an OpenCV-displayable image. [file:402]

---

## 7) Training the networks (how to reproduce)
All trainings use the target dataset built from `data/taichi1.mp4`. [file:402]

### 7.1 (Optional) Recompute skeleton cache
If needed, delete `data/taichi1.pkl` and folder `data/taichi1/`, then run any training or demo once; it will rebuild the cache (or set `forceCompute=True` in `VideoSkeleton`). [file:404]

### 7.2 Train VanillaNN (26D → image or stickman → image)
Open `src/GenVanillaNN.py` and enable the training mode in the `__main__` part (the code contains a `train` flag + `nepoch`). [file:402]  
Then run from repo root:
- Linux/macOS:
```
PYTHONPATH=src python src/GenVanillaNN.py data/taichi1.mp4
```
- Windows PowerShell:
```
$env:PYTHONPATH="src"
python src\GenVanillaNN.py data\taichi1.mp4
```
The trained weights are saved under `data/Dance/` with filenames like `DanceGenVanillaFromSke26.pth` or `DanceGenVanillaFromSkeim.pth` depending on the option used. [file:402][image:1]

### 7.3 Train GAN (WGAN-GP + L1)
Run:
- Linux/macOS:
```
PYTHONPATH=src python src/GenGAN.py data/taichi1.mp4
```
- Windows PowerShell:
```
$env:PYTHONPATH="src"
python src\GenGAN.py data\taichi1.mp4
```
This trains for a configured number of epochs (e.g. 50 in the provided main) and saves a checkpoint to `data/Dance/DanceGenGAN.pth`. [file:402][image:1]

---

## 8) Demo video (2 minutes)
Record a short video showing: [file:402]
- Running `DanceDemo.py` from the repository root.
- Switching `GEN_TYPE` (at least one NN + GAN recommended).
- The real-time window: `SOURCE | SKELETON | GENERATION`, FPS counter, and quitting with `q`.

---

## 9) Credits / notes
Project based on the TP inspired by “Everybody Dance Now” (Chan et al., ICCV 2019), with skeleton extraction handled by MediaPipe in the starter code. [file:402]

Author: <YOUR NAME>
Date: <DATE>
```

Si tu veux, colle ici ton `environment.yml` (ou `requirements.txt`) et je te complète la partie “Installation” avec les **noms exacts** des dépendances + versions, et je peux aussi te proposer une section “Résultats” (captures + commentaires) sans faire un vrai rapport séparé.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/116778225/d12f1c34-8abe-4de0-bbbb-1e2dafef6d7d/image.jpg?AWSAccessKeyId=ASIA2F3EMEYEXVSNCUT7&Signature=h0UfGsStkLSdPb9HMy0Pz7NPWZ4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEKL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDsHHabwIx2hH9rijr7Q9KdHzfq57Qs8AIiZJ8HApD7HwIhAN3KZsSmCkI70aOEXQITtfMFQtolFTK94M09zVj9TK5MKvMECGoQARoMNjk5NzUzMzA5NzA1Igxk61jdoKXDxQtoGNkq0AQ3jaY4BcxNzyOGVJ6TW%2FsXSOniSrNK6aXRzNKfNuRU6J8golL90hrAf%2FbOgS9kABRxJgnAYrSoXD%2BCGfZVUqhvMU2yZF%2Boi3%2FHJkVMEDgvmAV1RGSzhEsNHvqjaE4%2BB0KoFzesF9CYBbanps%2FASN2Bupy80JV9XdUxzFVTqDaFjTXKXzl6%2FQSabDjrWlY2VZWCn1AQ6DVdMYAUnH%2BjSnofmcLwLoxwW2cvYjFSrQlwYtIYXyCMvplPomVFVbaRTEu9%2Fabb93T965mzViyvUaFrJ3tnZkweOTuLU3OLnhg1wRsy%2FqqeUD9k77KCv7c7v%2Bz3GkAO7NJvB5uCnhxye27phnyU%2BFOHVNg0Dfu9CLktdUddrCLJE6a10tox5rnkyUZOGpqe47qMDr1%2BJgPMaZW87%2B1nGc1Ii6pkLJfId3El88xL0xFhGhY8nxy4knHh3lpvxP66gg3po%2BqJxBhnCtuUxAFXcw%2F2V6rfK59jv0FeSyFbIC5cZvsH5GllFCncm2RixV5EaFZ9nWlJngJPfbpuNZYApmSzGpBuBwcAKAM3JJxH6%2FPk3HEwl%2FM7z%2B9jct%2Bt1NsPygByvXftnOw45LT3CMHCLQoH1BTcRcPky2T6HCOX8sGGdJZN5lQQu90c97%2FIqwiO%2FXiThC%2BDVNzc7D0jlFiPb6PH%2F3xKlEZD%2BDnLgc3wqRZnY1L9u8mURLOA1Z6YdATY75qbzJMzMLB49yuyyA4FIYW6%2BIjIkuJ5vf1fmZ9%2BDPWTIZNbSR%2F1tCz8uQegCyzYeKxQQzcpAjhZhA1WMLWihsoGOpcB%2BL73dBktEtbG%2FWxQ%2FgHCtHlybgM9X14pQ2DVhwfvUxBpHAdvv6jtPJe3g2664Oq%2Fj9730aa7H%2FETpNXzCnsqmlHZpst8Ngxq%2F9r4hxfJ9bsMXD8m%2FD54sKWUJD6EDgoILKFEBtV1UKvLuFIEK694fj9K5njS9SDqvebvyi9qDBjxbN8VFAXSkOVTh8VgiMqMUdb%2BWTE6Kw%3D%3D&Expires=1765905956)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/b4e19712-949c-42bf-8bbe-471c434f93d4/paste.txt)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/1f4f91a2-67d7-415b-a890-545876b95021/VideoReader.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/b6b2aed6-777b-42ef-aee1-f8849bce1594/GenGAN.py)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/ab3e074e-8f6e-416b-98bd-a02dd1caa529/VideoSkeleton.py)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/1e793566-8f29-4299-a151-523c7fe9465f/GenVanillaNN.py)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/64522271-4852-4d4c-a70d-54115c32afd7/Vec3.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/116778225/32c8773d-140a-4b73-ad82-cf3e113e96e1/DanceDemo.py)
