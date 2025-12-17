# DeepAnimDance — Posture‑Guided Person Image Synthesis (TP "Everybody Dance Now")

*(README au format rapport comme requis : pas de rapport PDF séparé)*

---

## 📑 Table des matières

- [1) Structure du dépôt (projet principal)](#1-structure-du-dépôt-projet-principal)
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
  - [5.3 Étape 3 — Vanilla NN (image stickman → image)](#53-étape-3--vanilla-nn-image-stickman--image--gennnskeimltoimage)
  - [5.4 Étape 4 — Raffinement GAN](#54-étape-4--raffinement-gan--gengan-wgangp--l1)
- [6) Entraînement (reproductibilité)](#6-entraînement-reproductibilité)
  - [6.1 Optionnel : reconstruire le cache](#61-optionnel--reconstruire-le-cache)
  - [6.2 Entraîner VanillaNN (26D ou stickman)](#62-entraîner-vanillaNN-26d-ou-stickman)
  - [6.3 Entraîner GAN](#63-entraîner-gan)
- [7) Vidéo de démonstration](#7-vidéo-de-démonstration)
- [8) Bonus — Application web Flask](#9-bonus--application-web-flask-exécution-uniquement)
  - [8.1 Rôle (ce qu'elle ajoute)](#91-rôle-ce-quelle-ajoute)
  - [8.2 Installation & exécution (venv)](#92-installation--exécution-venv)

---
## 1) Structure du dépôt (projet principal)

Structure typique (racine) :
- `src/` : code source (démo, construction de l'ensemble de données, classe squelette, générateurs).
- `data/` : vidéos + cache + poids entraînés.
  - `data/taichi1.mp4` : vidéo **cible** utilisée pour construire l'ensemble de données.
  - `data/taichi1.pkl` + `data/taichi1/` : squelettes/images en cache produits par `VideoSkeleton`.
  - `data/Dance/` : réseaux entraînés (`.pth`) chargés par les générateurs.

---

## 2) Environnements (venv + conda)

Deux configurations ont été utilisées selon les membres de l'équipe :
- Un membre a utilisé `venv` (pip).
- Un autre membre a utilisé `conda` via `environment.yml`.

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

Les chemins vers les poids sont codés en dur comme des chemins relatifs comme `data/Dance/...`, et les imports sont écrits comme `from VideoSkeleton import ...`, donc exécutez depuis la racine du dépôt avec `src/` dans `PYTHONPATH`.

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

`DanceDemo.py` affiche 3 panneaux : **VIDÉO SOURCE | SQUELETTE | GÉNÉRATION**, et un overlay FPS.

Contrôles : `q` quitte, `n` saute ~100 images.

### 3.3 Choisir le générateur (GEN_TYPE)

Dans `DanceDemo.py`, sélectionnez le générateur avec `GEN_TYPE` (et une vidéo source comme `data/taichi2.mp4`).

| GEN_TYPE | Méthode | Description |
|---:|---|---|
| 1 | Nearest Neighbor | Baseline `GenNeirest` (pas d'apprentissage). |
| 2 | Vanilla NN (26D) | Vecteur de squelette réduit (26D) → image. |
| 3 | Vanilla NN (stickman) | Image stickman → image (encodeur‑décodeur + skips). |
| 4 | GAN | WGAN‑GP + L1 (stickman → image). |

---

## 4) Aperçu du pipeline (données → squelette → génération)

### 4.1 Construction et mise en cache de l'ensemble de données cible (VideoSkeleton)

`VideoSkeleton` construit un ensemble de données cible de paires (squelette, image) à partir de la vidéo cible.

Pour accélérer les exécutions répétées, il utilise la mise en cache : il stocke les squelettes et les métadonnées dans un `.pkl` et stocke les images extraites sur le disque ; les exécutions suivantes rechargent les données en cache si disponibles.

### 4.2 Extraction de squelette (MediaPipe Pose) et représentation

Les squelettes sont extraits avec MediaPipe Pose (33 points de repère avec x, y, z).

Pour l'apprentissage, un **squelette réduit** est utilisé : 13 articulations avec des coordonnées (x, y) → 26 valeurs (entrée de dimension inférieure, apprentissage plus facile).

### 4.3 Recadrage et cas d'échec (robustesse de la démo)

Pour chaque image source, la démo utilise la logique de recadrage de l'ensemble de données cible (`cropAndSke`) pour recadrer autour de la pose détectée avant de dessiner/générer.

Si aucun squelette n'est détecté, la démo affiche un panneau d'erreur rouge et saute la génération pour cette image.

Pour maintenir une vitesse en temps réel, la démo calcule une image sur 5 par défaut (modifiable).

---

## 5) Méthodes et concepts (étapes TP)

Ce projet suit une approche progressive : baseline → apprentissage supervisé → image‑à‑image → raffinement GAN pour le réalisme.

### 5.1 Étape 1 — Baseline Nearest Neighbor (GenNeirest)

**Concept :** pour chaque pose source, trouver la pose la plus proche dans l'ensemble de données cible et sortir l'image cible réelle correspondante (pas de synthèse).

**Implémentation (votre code) :**
- Itérer sur les squelettes cibles, calculer `ske.distance(target_ske)` et sélectionner le minimum.
- Retourner l'image cible correspondante convertie en float dans [0,1] pour compatibilité d'affichage.

**Limitations attendues :** mouvement saccadé (pas de modèle temporel), limité aux poses existant dans l'ensemble de données cible, et recherche linéaire lente pour les grands ensembles de données.

### 5.2 Étape 2 — Vanilla NN (vecteur 26D → image) — `GenNNSke26ToImage`

**Concept :** apprendre une correspondance directe d'un vecteur de pose de 26 dimensions vers une image RGB cible 64×64.

**Architecture (votre code final) :**
- `Linear(26 → 256*4*4)` puis reshape en un tenseur 4×4×256.
- Suréchantillonnage avec des blocs `ConvTranspose2d` pour atteindre 64×64.
- Stabilisation/qualité : `BatchNorm2d`, `LeakyReLU`, et plusieurs `ResidualBlock`s.
- Sortie : `Tanh()` → valeurs dans [-1,1] (alignées avec la normalisation cible).

**Entraînement :** `GenVanillaNN.train()` utilise Adam et `MSELoss` (régression pixel par pixel), qui est simple mais conduit souvent au flou (effet de moyenne).

### 5.3 Étape 3 — Vanilla NN (image stickman → image) — `GenNNSkeImToImage`

**Concept :** convertir le squelette en une "image stickman" intermédiaire et résoudre la traduction image-à-image.

**Création du stickman :** `SkeToImageTransform(64)` crée une image noire, dessine le squelette (dessin BGR coloré), puis convertit en RGB avant `ToTensor()`.

**Architecture (votre code final) :**
- Encodeur : 4 couches de sous-échantillonnage `Conv2d` (64×64 → 4×4) pour extraire les caractéristiques de pose.
- Module ajouté : `SelfAttention(256)` au niveau de caractéristiques 8×8×256 pour capturer les dépendances à longue portée.
- Goulot d'étranglement : blocs résiduels empilés (`ResidualBlock(512)` répétés).
- Décodeur : suréchantillonnage `ConvTranspose2d` pour reconstruire 64×64.
- Connexions résiduelles : concaténations explicites `torch.cat([...])` (style U‑Net).
- Sortie : `Tanh()` dans [-1,1].

**Entraînement :** toujours supervisé avec `MSELoss`, donc la pose est respectée mais les textures peuvent rester lisses.

### 5.4 Étape 4 — Raffinement GAN — `GenGAN` (WGAN‑GP + L1)

**Motivation :** les pertes de pixels supervisées (en particulier MSE) produisent souvent des images floues ; l'entraînement GAN encourage des textures plus nettes et des sorties plus réalistes.

**Discriminateur/Critique :** CNN de style PatchGAN produisant une carte de patchs, sans sigmoïde (score du critique WGAN).

**Pertes implémentées (votre code) :**
- Critique (WGAN‑GP) : D(fake).mean - D(real).mean + λ_gp*GP avec pénalité de gradient calculée sur des échantillons interpolés.
- Générateur : terme adversarial -D(fake).mean + terme de reconstruction `lambda_l1 * L1(fake, real)` avec `lambda_l1 = 100`.

**Détails de la boucle d'entraînement (votre code) :**
- `n_critic = 5` mises à jour du critique par mise à jour du générateur.
- Adam avec `betas=(0.0, 0.9)`, `lr_g=1e-4`, `lr_d=4e-4`.
- Point de contrôle enregistré comme `{"netG": state_dict, "netD": state_dict}` dans `data/Dance/DanceGenGAN.pth`.

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

Exécutez :

```bash
PYTHONPATH=src python src/GenVanillaNN.py data/taichi1.mp4
```

Poids enregistrés (selon le mode) :
- `data/Dance/DanceGenVanillaFromSke26.pth`
- `data/Dance/DanceGenVanillaFromSkeim.pth`

### 6.3 Entraîner GAN

Dans `GenGAN.py`, définissez `train = True` et exécutez :

```bash
PYTHONPATH=src python src/GenGAN.py data/taichi1.mp4
```

Sortie :
- `data/Dance/DanceGenGAN.pth`

---

## 7) Vidéo de démonstration

Une vidéo de démonstration d'environ 2 minutes est fournie dans ce dépôt :

- `demo.mp4` (ou lien) : <PUT_LINK_HERE>

La vidéo montre :
- L'exécution de `src/DanceDemo.py` depuis la racine du dépôt avec les réseaux entraînés.
- Le passage entre au moins deux modes (par exemple, `GEN_TYPE=1` baseline et `GEN_TYPE=4` GAN).
- Les 3 panneaux (SOURCE | SQUELETTE | GÉNÉRATION), l'affichage FPS, et la sortie avec `q`.

---

## 9) Bonus — Application web Flask (exécution uniquement)

Dépôt GitHub : https://github.com/infoelouarroudi-stack/DemoDaanceWEB

### 9.1 Rôle (ce qu'elle ajoute)

Une petite interface Flask a été ajoutée en **bonus** pour exécuter/visualiser le projet depuis un navigateur (wrapper UI), tout en gardant le projet principal comme cœur noté.

Structure :
- `app.py` : point d'entrée du serveur Flask.
- `templates/index.html` : page d'accueil.
- `templates/viewer.html` : page de visualisation des résultats.
- `static/` : CSS/JS/assets.

### 9.2 Installation & exécution (venv)

Depuis la racine du dépôt de l'application web (où se trouvent `app.py` et `requirements.txt`) :

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Puis ouvrez (serveur de développement Flask par défaut) : http://localhost:5000/

---
