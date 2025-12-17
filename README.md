# DeepAnimDance — Posture-guided Person Image Synthesis (TP "Everybody Dance Now")

> README en format "mini-rapport" (à la place d'un rapport PDF), demandé pour l'évaluation.

---

## 📑 Table des matières

- [Contexte & objectif](#contexte--objectif)
  - [Livrables (évaluation)](#livrables-évaluation)
- [1) Structure du dépôt (projet principal)](#1-structure-du-dépôt-projet-principal)
  - [Arborescence (exemple)](#arborescence-exemple)
  - [Rôle de chaque fichier (src/)](#rôle-de-chaque-fichier-src)
- [2) Environnements (venv + conda)](#2-environnements-venv--conda)
  - [Option A — venv (pip)](#option-a--venv-pip)
  - [Option B — conda (environment.yml)](#option-b--conda-environmentyml)
- [3) Exécuter la démo (projet principal)](#3-exécuter-la-démo-projet-principal)
  - [3.1 Important : exécuter depuis la racine du dépôt + PYTHONPATH](#31-important--exécuter-depuis-la-racine-du-dépôt--pythonpath)
  - [3.2 Ce que la démo affiche](#32-ce-que-la-démo-affiche)
  - [3.3 Choisir le générateur (GEN_TYPE)](#33-choisir-le-générateur-gen_type)
- [4) Aperçu du pipeline (données → squelette → génération)](#4-aperçu-du-pipeline-données--squelette--génération)
  - [4.1 Construction et mise en cache de l'ensemble de données cible (VideoSkeleton)](#41-construction-et-mise-en-cache-de-lensemble-de-données-cible-videoskeleton)
  - [4.2 Extraction de squelette (MediaPipe Pose) et représentation](#42-extraction-de-squelette-mediapipe-pose-et-représentation)
  - [4.3 Recadrage et cas d'échec (robustesse de la démo)](#43-recadrage-et-cas-déchec-robustesse-de-la-démo)
- [5) Méthodes et concepts (étapes TP)](#5-méthodes-et-concepts-étapes-tp)
  - [5.1 Étape 1 — Baseline Nearest Neighbor (GenNeirest)](#51-étape-1--baseline-nearest-neighbor-genneirest)
  - [5.2 Étape 2 — Vanilla NN (vecteur 26D → image)](#52-étape-2--vanilla-nn-vecteur-26d--image--gennnske26toimage)
  - [5.3 Étape 3 — Vanilla NN (image stickman → image)](#53-étape-3--vanilla-nn-image-stickman--image--gennnskeimtoimage)
  - [5.4 Étape 4 — Raffinement GAN](#54-étape-4--raffinement-gan--gengan-wgangp--l1)
- [6) Entraînement (reproductibilité)](#6-entraînement-reproductibilité)
  - [6.1 Optionnel : reconstruire le cache](#61-optionnel--reconstruire-le-cache)
  - [6.2 Entraîner VanillaNN (26D ou stickman)](#62-entraîner-vanillann-26d-ou-stickman)
  - [6.3 Entraîner GAN](#63-entraîner-gan)
- [7) Vidéo de démonstration](#7-vidéo-de-démonstration)
- [8) Bonus — Application web Flask (exécution uniquement)](#8-bonus--application-web-flask-exécution-uniquement)
  - [8.1 Rôle (ce qu'elle ajoute)](#81-rôle-ce-quelle-ajoute)
  - [8.2 Installation & exécution (venv)](#82-installation--exécution-venv)
- [Limitations (observées / attendues)](#limitations-observées--attendues)
- [Crédits](#crédits)

---

## Contexte & objectif

Ce projet implémente la **synthèse d'images guidée par la pose** ("posture-guided image synthesis") : on transfère le mouvement (séquence de poses) d'une **vidéo source** vers une **identité cible** apprise à partir d'une vidéo cible.

Le pipeline global est : **Vidéo → Extraction squelette (MediaPipe) → Génération d'image (plusieurs méthodes) → Affichage/Démo**.

### Livrables (évaluation)

- Un ZIP incluant : tout le code, les données nécessaires, les poids entraînés (`.pth`) et une vidéo de démonstration (~2 min).
- Pas de rapport séparé : ce README contient les explications techniques.

---

## 1) Structure du dépôt (projet principal)

### Arborescence (exemple)

```
project-root/
├── src/
│   ├── DanceDemo.py
│   ├── VideoSkeleton.py
│   ├── VideoReader.py
│   ├── Skeleton.py
│   ├── Vec3.py
│   ├── GenNearest.py
│   ├── GenVanillaNN.py
│   └── GenGAN.py
│
├── data/
│   ├── Dance/                      # poids entraînés (.pth)
│   ├── taichi1.mp4                 # vidéo cible (dataset target)
│   ├── taichi1.pkl                 # cache squelette/dataset
│   ├── taichi1/                    # frames extraites/cropées
│   ├── taichi2.mp4                 # exemple vidéo source
│   └── karate1.mp4                 # autre exemple vidéo source
│
├── requirements.txt
├── environment.yml
└── README.md
```

### Rôle de chaque fichier (src/)

**`DanceDemo.py`**  
Lance la démo temps réel : lecture vidéo source, extraction pose, appel au générateur choisi, affichage "SOURCE | SQUELETTE | GENERATION" + FPS + touches (`q`, `n`).

**`VideoSkeleton.py`**  
Construit le dataset cible à partir d'une vidéo (squelettes + images cropées), gère la mise en cache (`.pkl`) + sauvegarde frames sur disque, et fournit `cropAndSke` utilisé en démo.

**`VideoReader.py`**  
Lecture vidéo (OpenCV), accès total frames, lecture frame par frame et skip de N frames.

**`Skeleton.py`**  
Représentation du squelette : extraction MediaPipe Pose (33 landmarks), réduction à 13 joints (26D), distance entre squelettes, bounding box, dessin du squelette (stickman).

**`Vec3.py`**  
Vecteurs 3D + opérations utiles (support interne pour les landmarks (x,y,z)).

**`GenNearest.py`**  
Baseline "Nearest Neighbor" : renvoie l'image target correspondant au squelette le plus proche.

**`GenVanillaNN.py`**  
Deux versions supervisées :
1. vecteur 26D → image (décodeur par upsampling)
2. stickman image → image (encoder-decoder / U-Net simplifié + améliorations)

**`GenGAN.py`**  
Raffinement GAN : générateur image→image + critic/discriminator, entraînement WGAN-GP + L1, sauvegarde/chargement de checkpoint.

---

## 2) Environnements (venv + conda)

Deux environnements ont été utilisés selon les membres du groupe : l'un via **venv/pip**, l'autre via **conda** (`environment.yml`).

### Option A — venv (pip)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Option B — conda (environment.yml)

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```

---

## 3) Exécuter la démo (projet principal)

### 3.1 Important : exécuter depuis la racine du dépôt + PYTHONPATH

Le projet charge les modèles via des chemins relatifs (ex: `data/Dance/...`) et les imports Python sont écrits sous la forme `from VideoSkeleton import ...`.

Il faut donc lancer depuis la racine et ajouter `src/` au `PYTHONPATH`.

**Linux / macOS**

```bash
PYTHONPATH=src python src/DanceDemo.py
```

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH="src"
python src\DanceDemo.py
```

### 3.2 Ce que la démo affiche

La fenêtre de démo affiche 3 panneaux :

- **SOURCE VIDEO** : frame source recadrée
- **SQUELETTE** : stickman (squelette dessiné)
- **GENERATION** : image générée (target animé par la pose source)

**Contrôles clavier :**
- `q` : quitter
- `n` : sauter ~100 frames

### 3.3 Choisir le générateur (GEN_TYPE)

Dans `DanceDemo.py`, régler `GEN_TYPE` :

| GEN_TYPE | Méthode | Résumé |
|:--------:|---------|--------|
| **1** | Nearest Neighbor | Baseline sans apprentissage. |
| **2** | Vanilla NN (26D) | Vecteur squelette 26D → image. |
| **3** | Vanilla NN (stickman) | Image stickman → image (meilleure structure). |
| **4** | GAN | WGAN-GP + L1 (améliore le réalisme). |

**Exemple** (changer la vidéo source si besoin) :

```python
GEN_TYPE = 4
ddemo = DanceDemo("data/taichi2.mp4", GEN_TYPE)
ddemo.draw()
```

---

## 4) Aperçu du pipeline (données → squelette → génération)

### 4.1 Construction et mise en cache de l'ensemble de données cible (VideoSkeleton)

`VideoSkeleton` construit un dataset de paires (image target, squelette) à partir de la vidéo cible `taichi1.mp4`.

Pour éviter de recalculer à chaque exécution, on met en cache :
- `data/taichi1.pkl` (métadonnées + squelettes)
- `data/taichi1/` (frames cropées sauvegardées sur disque)

### 4.2 Extraction de squelette (MediaPipe Pose) et représentation

- **Extraction** par **MediaPipe Pose** : 33 landmarks (x, y, z).
- **Représentation réduite** utilisée pour l'apprentissage : 13 joints sélectionnés, uniquement (x, y) → vecteur **26D**.
- **Représentation "image"** : squelette dessiné sur une image (stickman), utilisé pour les approches image→image.

### 4.3 Recadrage et cas d'échec (robustesse de la démo)

- Le recadrage est basé sur la bounding box du squelette (centre + cropRatio), avec padding si nécessaire.
- Si aucun squelette n'est détecté, la démo affiche un panneau d'erreur (rouge) et skip la génération sur cette frame.
- Pour le temps réel, la démo peut calculer une frame sur N (ex: 1/5) afin d'augmenter les FPS.

---

## 5) Méthodes et concepts (étapes TP)

Le projet suit une progression pédagogique : **baseline → apprentissage supervisé → image-to-image → GAN**.

### 5.1 Étape 1 — Baseline Nearest Neighbor (GenNeirest)

**Idée :**  
Pour chaque pose source, chercher dans le dataset cible la pose la plus proche, puis renvoyer l'image correspondante (aucune génération).

**Distance :**  
Somme des distances joint-par-joint entre deux squelettes (utilise la représentation squelette du code).

**Limites :**  
Pas fluide, limité aux poses existantes, et coût de recherche linéaire.

---

### 5.2 Étape 2 — Vanilla NN (vecteur 26D → image) — GenNNSke26ToImage

**Entrée :**  
Vecteur 26D = (x,y) de 13 articulations.

**Objectif :**  
Apprendre une fonction G : 26D → image target 64×64.

**Concept d'architecture (décodeur) :**

- Projection (MLP) du 26D vers un tenseur compact (ex: 4×4×C).
- Upsampling progressif par convolutions transposées (ConvTranspose2d) pour atteindre 64×64.
- Normalisation (BatchNorm) + activations (souvent LeakyReLU) pour stabiliser.
- Sortie normalisée via `Tanh` ([-1,1]) puis dé-normalisation pour affichage.

**Training (supervisé) :**

Loss pixel-wise (MSE) : simple et stable, mais peut lisser les textures (effet "moyenne").

---

### 5.3 Étape 3 — Vanilla NN (image stickman → image) — GenNNSkeImToImage

**Entrée :**  
Image stickman (squelette dessiné) 64×64.

**Objectif :**  
Apprendre G : stickman → image target (image-to-image).

**Concept d'architecture (encoder-decoder / U-Net simplifié) :**

- **Encodeur** : convs qui compressent l'image pour extraire les features de pose.
- **Décodeur** : convTranspose pour reconstruire l'image.
- **Skip connections** (type U-Net) : concaténation des features encodeur→décodeur pour conserver les détails spatiaux.
- **Modules d'amélioration possibles** (dans votre version finale) : blocs résiduels + attention (self-attention) pour mieux modéliser la structure globale.

**Training :**

Toujours supervisé (MSE/L1 possibles). L'image est plus structurée qu'un vecteur, donc souvent plus facile à apprendre.

---

### 5.4 Étape 4 — Raffinement GAN — GenGAN (WGAN-GP + L1)

**Motivation :**  
Les approches MSE ont tendance à produire du flou. Un GAN force des textures plus réalistes.

**Principe :**

- **Générateur G** : stickman → image.
- **Discriminateur/Critic D** : distingue (réel vs généré). PatchGAN possible : sortie en carte de patches.

**WGAN-GP + reconstruction :**

- **WGAN** : utilise un critic (pas de sigmoid) pour une optimisation plus stable.
- **Gradient Penalty (GP)** : stabilise l'entraînement (évite l'explosion/instabilité).
- **Perte hybride** : adversarial (réalisme) + L1 (respect de la structure/pose).

---

## 6) Entraînement (reproductibilité)

L'entraînement utilise l'ensemble de données cible construit à partir de `data/taichi1.mp4`.

### 6.1 Optionnel : reconstruire le cache

Si nécessaire, supprimez le cache et le répertoire d'images puis réexécutez l'entraînement/démo pour que `VideoSkeleton` les recalcule :

```bash
rm -f data/taichi1.pkl
rm -rf data/taichi1/
```

(Les utilisateurs Windows peuvent supprimer manuellement depuis l'Explorateur de fichiers.)

### 6.2 Entraîner VanillaNN (26D ou stickman)

Dans `GenVanillaNN.py`, configurez :

- `optSkeOrImage = 1` pour vecteur(26D)→image
- `optSkeOrImage = 2` pour stickman→image

et activez l'entraînement dans la section `__main__`.

**Exécutez :**

```bash
PYTHONPATH=src python src/GenVanillaNN.py data/taichi1.mp4
```

**Poids enregistrés** (selon le mode) :
- `data/Dance/DanceGenVanillaFromSke26.pth`
- `data/Dance/DanceGenVanillaFromSkeim.pth`

### 6.3 Entraîner GAN

Dans `GenGAN.py`, définissez `train = True` et exécutez :

```bash
PYTHONPATH=src python src/GenGAN.py data/taichi1.mp4
```

**Sortie :**
- `data/Dance/DanceGenGAN.pth`

---

## 7) Vidéo de démonstration

Une vidéo (~2 minutes) est fournie ici :
- `demo.mp4` (dans le ZIP / dans le dépôt) **ou** lien : `<PUT_LINK_HERE>`

Elle montre :
- L'exécution de `DanceDemo.py` depuis la racine du dépôt (avec modèles entraînés).
- Au moins deux modes (ex: `GEN_TYPE=1` baseline puis `GEN_TYPE=4` GAN).
- L'interface 3 panneaux, l'affichage FPS, et la sortie via `q`.

---

## 8) Bonus — Application web Flask (exécution uniquement)

**Dépôt bonus :** https://github.com/infoelouarroudi-stack/DemoDaanceWEB

### 8.1 Rôle (ce qu'elle ajoute)

Une petite interface web Flask a été ajoutée en bonus pour exécuter/visualiser la démo via un navigateur, en s'appuyant sur :
- `app.py` (serveur Flask)
- `templates/index.html` (page d'accueil)
- `templates/viewer.html` (page de visualisation)
- `static/` (assets)

### 8.2 Installation & exécution (venv)

Depuis la racine du dépôt Flask (là où se trouvent `app.py` et `requirements.txt`) :

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Puis ouvrir : **http://localhost:5000/**

---

## Limitations (observées / attendues)

- **Cohérence temporelle limitée** (pas de contrainte explicite inter-frames), donc flickering possible.
- **Résolution limitée** (64×64 au niveau des réseaux) : la qualité dépend fortement du dataset et du modèle.
- **La baseline Nearest Neighbor** est limitée au contenu exact de la vidéo cible.

---

## Crédits

- Inspiré du concept **"Everybody Dance Now"** (Chan et al., ICCV 2019) dans le cadre du TP.
- Extraction de pose via **MediaPipe Pose** (starter code / pipeline du projet).
