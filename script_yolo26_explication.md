# 🎬 Script Détaillé — YOLO26 NMS-Free : Architecture & Entraînement

> **Objectif** : Expliquer en profondeur le fonctionnement de YOLO26 NMS-Free, de l'architecture à l'entraînement, en répondant systématiquement à "Pourquoi ?" pour chaque décision technique.

---

## 📌 INTRODUCTION — Le Problème Posé

**[Slide d'ouverture]**

Bonjour à tous. Aujourd'hui, on va plonger dans **YOLO26**, la dernière évolution de la famille YOLO, et plus précisément dans son innovation majeure : le **NMS-Free**, c'est-à-dire la suppression complète du Non-Maximum Suppression.

**La question fondamentale qu'on va résoudre aujourd'hui :**

> *Comment faire pour qu'un réseau de neurones produise **exactement 1 prédiction par objet**, sans aucun post-traitement ?*

La réponse tient en une phrase : **on l'entraîne à le faire**. Mais le "comment" est fascinant et constitue le cœur de cette présentation.

---

## 🧩 PARTIE 1 — Pourquoi Supprimer le NMS ?

### 1.1 Rappel : Qu'est-ce que le NMS ?

Dans un détecteur classique comme YOLOv8 ou YOLO11 :
- Le réseau produit **8400 prédictions** (appelées ancres) par image
- Pour **un seul piéton**, environ **10 ancres** prédisent toutes le même objet avec des boîtes similaires
- Le **NMS** (Non-Maximum Suppression) supprime les doublons en gardant la boîte avec le meilleur score et en éliminant celles qui se chevauchent trop (IoU > seuil)

### 1.2 Les Problèmes du NMS

| Problème | Explication |
|----------|-------------|
| **Latence** | NMS a une complexité O(n²) — environ 4ms sur un GPU T4 |
| **Non-parallélisable** | C'est un algorithme séquentiel, mal adapté au GPU |
| **Hyperparamètres sensibles** | Il faut calibrer le seuil IoU et le seuil de confiance **par domaine** |
| **Export difficile** | Le NMS est une opération dynamique difficile à intégrer dans un graphe ONNX/TensorRT |
| **Objets denses** | Dans les foules, le NMS peut supprimer des vrais positifs par erreur |

### 1.3 L'Objectif de YOLO26

> Produire un détecteur dont le réseau lui-même n'active qu'**une seule ancre par objet**, rendant le NMS structurellement inutile.

---

## 🏗️ PARTIE 2 — L'Architecture Double Tête

### 2.1 Vue d'Ensemble

YOLO26 conserve l'architecture duale héritée de YOLOv10 :

```
Image 640×640
    │
    ▼
┌──────────────┐
│   BACKBONE   │  ← Extraction de features (C2f / RepVGG)
│  (partagé)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  NECK (FPN)  │  ← Fusion multi-échelle
│  (partagé)   │     P3 (80×80), P4 (40×40), P5 (20×20)
└──────┬───────┘     = 8400 ancres au total
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
┌──────────────┐                 ┌─────────────────┐
│  Tête O2M    │                 │   detach()       │ ← STOP GRADIENT !
│  (one-to-many)│                 │       │          │
│  topk=10     │                 │       ▼          │
│  cls + bbox  │                 │  Tête O2O        │
└──────┬───────┘                 │  (one-to-one)    │
       │                         │  topk=7→topk2=1  │
       ▼                         │  cls + bbox      │
   L_o2m                        └──────┬────────────┘
       │                               │
       │                               ▼
       │                           L_o2o
       │                               │
       └────────┐    ┌─────────────────┘
                ▼    ▼
         ┌─────────────────┐
         │    ProgLoss      │
         │ w_o2m·L₁ + w_o2o·L₂│
         └─────────────────┘
```

### 2.2 Les Trois Niveaux de Features

Le backbone et le neck produisent des feature maps à **3 échelles** :

| Niveau | Taille grille | Stride | Nb ancres | Détecte |
|--------|--------------|--------|-----------|----------|
| **P3** | 80 × 80 | 8 px | 6400 | Petits objets (piétons éloignés) |
| **P4** | 40 × 40 | 16 px | 1600 | Objets moyens |
| **P5** | 20 × 20 | 32 px | 400 | Grands objets (voitures proches) |
| **Total** | — | — | **8400** | Toutes tailles |

> **Pourquoi 3 niveaux ?** Un piéton à 100m occupe peut-être 10 pixels — seul P3 (stride=8) peut le détecter. Un bus de face occupe 400 pixels — P5 (stride=32) est plus adapté. Cette hiérarchie est essentielle pour détecter des objets de **toutes tailles**.

### 2.3 La Tête O2M (One-to-Many) — Signal Dense

La tête O2M est la tête classique :
- **Branche régression** : `cv2` = Conv 3×3 → Conv 3×3 → Conv2d 1×1 → **4 distances** (l, t, r, b)
- **Branche classification** : `cv3` = DWConv 3×3 → Conv 1×1 → DWConv 3×3 → Conv 1×1 → Conv2d 1×1 → **80 scores** de classe

Pendant l'entraînement, le TAL lui assigne **10 ancres positives par objet** → c'est le signal "dense" qui entraîne le backbone.

### 2.4 La Tête O2O (One-to-One) — Prédiction Unique

La tête O2O est une **copie profonde indépendante** (`deepcopy`) de la tête O2M :

```python
self.one2one_cv2 = copy.deepcopy(self.cv2)  # mêmes couches, poids DIFFÉRENTS
self.one2one_cv3 = copy.deepcopy(self.cv3)
```

> **Pourquoi deepcopy ?** Les deux têtes doivent avoir la **même architecture** mais des **poids indépendants**. O2O apprend à ne produire qu'UNE prédiction par objet ; O2M en produit 10. Si elles partageaient les mêmes poids, elles ne pourraient pas avoir des comportements différents.

Pendant l'entraînement, le TAL ne lui assigne qu'**1 seule ancre par objet** (via `topk2=1`).

### 2.5 Le `detach()` — La Ligne la Plus Importante

```python
# head.py — Detect.forward() — LIGNE CRUCIALE
x_detach = [xi.detach() for xi in x]  # ← COUPE LE GRADIENT !

preds_o2o = self.forward_head(x_detach,
    box_head=self.one2one_cv2,
    cls_head=self.one2one_cv3)
```

> **Pourquoi detach() ?** C'est **la clé de la stabilité de l'entraînement**. Sans `detach()`, la tête O2O enverrait des gradients au backbone qui **contredisent** ceux de O2M :
> - O2M veut activer **10 ancres** par objet
> - O2O veut n'en activer qu'**une seule**
> - Ces signaux opposés rendraient le backbone **instable et divergent**
>
> Avec `detach()`, seuls les poids de `one2one_cv2` et `one2one_cv3` sont mis à jour par la loss O2O. Le backbone est entraîné **UNIQUEMENT par O2M**.

---

## 🔄 PARTIE 3 — Le Pipeline d'Entraînement (Une Itération Complète)

### Étape 0 : Initialisation

```python
# Deux fonctions de perte avec des TAL DIFFÉRENTS
self.one2many = v8DetectionLoss(model, tal_topk=10)
self.one2one  = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)

# ProgLoss — poids initiaux
self.o2m = 0.8         # début : 80% pour O2M
self.o2o = 0.2         # début : 20% pour O2O
self.final_o2m = 0.1   # fin : 10% pour O2M
```

### Étape 1 : Chargement des Données

Les données sont des **annotations COCO standard** — aucune modification nécessaire pour le NMS-Free :

```python
batch = {
    "img": tensor[B, 3, 640, 640],   # B images RGB
    "batch_idx": tensor[N],           # index image de chaque annotation
    "cls": tensor[N, 1],              # classe (0-79)
    "bboxes": tensor[N, 4],           # boîtes normalisées [x,y,w,h]
}
```

> **Point clé** : Le "secret" du NMS-Free est dans le **label assignment**, PAS dans les données.

### Étape 2 : Forward — Backbone + Neck (partagés)

L'image traverse le backbone puis le neck pour produire les 3 niveaux de features (P3, P4, P5 = 8400 ancres).

### Étape 3 : Forward — Tête O2M

Les features passent par la tête O2M **avec gradient activé** :

```python
preds_o2m = self.forward_head(x, box_head=self.cv2, cls_head=self.cv3)
# → boxes: tensor[B, 4, 8400], scores: tensor[B, 80, 8400]
```

Les gradients de O2M remontent au backbone et au neck — c'est elle qui entraîne principalement le backbone grâce à son signal dense (10 ancres/objet).

### Étape 4 : Forward — DETACH + Tête O2O

Les features sont **détachées** puis passées à la tête O2O :

```python
x_detach = [xi.detach() for xi in x]  # COUPE LE GRADIENT
preds_o2o = self.forward_head(x_detach, box_head=self.one2one_cv2, ...)
```

### Étape 5 : Pré-traitement des Targets

Les annotations COCO sont converties en format interne :
1. Concaténation de `batch_idx + cls + bboxes`
2. Conversion `xywh → xyxy` et mise à l'échelle
3. Séparation labels / boîtes
4. Génération des **8400 points d'ancrage** via `make_anchors()`
5. Décodage des boîtes prédites (`dist2bbox` — régression directe, PAS de DFL)

> **Pourquoi pas de DFL ?** YOLO26 utilise `reg_max=1`, ce qui supprime la Distribution Focal Loss. Au lieu de prédire une distribution softmax sur 16 valeurs, le réseau prédit **directement 4 distances** (l, t, r, b). Cela simplifie l'export sur NPU/DSP.

### Étape 6 : Label Assignment — TAL (le cœur du NMS-Free)

C'est l'**étape la plus critique** — voir la Partie 4 pour les détails.

### Étape 7 : Calcul des 3 Pertes

Chaque tête calcule indépendamment :

| Composante | Formule | Rôle |
|-----------|---------|------|
| **① Classification (BCE)** | `BCE(pred_scores, target_scores)` | Scores de classe — targets **continus** [0,1] |
| **② Localisation (CIoU)** | `(1 - CIoU) × weight` | Qualité géométrique des boîtes |
| **③ Distance L1** | `L1(pred_dist, target_ltrb)` | Précision des distances aux bords |

> **Pourquoi des targets continus et pas binaires ?** Une ancre qui chevauche 90% du GT mérite un target de ~0.9, tandis qu'une ancre qui chevauche 30% mérite ~0.3. La BCE pénalise proportionnellement à la qualité réelle.

> **Pourquoi CIoU ET L1 ?** CIoU mesure la qualité géométrique globale (chevauchement + distance centres + ratio d'aspect). L1 mesure les erreurs coordonnée par coordonnée. Les deux sont **complémentaires**.

### Étape 8 : ProgLoss — Pondération Progressive

Les pertes O2M et O2O sont combinées avec des poids qui **évoluent au fil des époques** :

```python
L_total = loss_o2m × w_o2m + loss_o2o × w_o2o
```

| Époque | w_o2m | w_o2o | Phase dominante |
|--------|-------|-------|-----------------|
| 0 | **0.80** | 0.20 | Signal dense (O2M) — apprentissage des features |
| 250 | 0.45 | 0.55 | Transition — croisement |
| 499 | **0.10** | 0.90 | Alignement inférence (O2O) |

**Formule de décroissance linéaire :**
```
w_o2m(t) = max(1 − t/(epochs−1), 0) × (0.8 − 0.1) + 0.1
w_o2o(t) = 1.0 − w_o2m(t)
```

> **Pourquoi cette progression ?**
> - **Début (w_o2m=0.8)** : Le modèle ne sait rien. Il a besoin du signal dense de O2M pour apprendre les features de base rapidement
> - **Milieu** : Le backbone est déjà entraîné. On renforce O2O pour affiner sa sélection unique
> - **Fin (w_o2m=0.1)** : On veut que le modèle s'aligne au maximum avec l'inférence (O2O seule). Le poids O2M reste à 0.1 (pas 0) pour une régularisation légère

### Étape 9 : Backpropagation

```
L_o2m → cv2, cv3 → Neck → Backbone     ← GRADIENT COMPLET
L_o2o → one2one_cv2, one2one_cv3 → STOP  ← GRADIENT BLOQUÉ par detach()
```

L'optimiseur **MuSGD** (SGD + correction Muon) est utilisé :
- **Backbone** : SGD + correction Muon (stabilisation)
- **Têtes** : SGD standard (apprentissage rapide)

> **Pourquoi MuSGD ?** L'entraînement dual-head est plus instable qu'un entraînement classique. MuSGD ajoute une "correction de courbure" au backbone, inspirée de l'optimisation des LLM (Muon/Kimi K2), qui lisse les oscillations de gradient.

### Étape 10 : Inférence — O2O Uniquement

À l'inférence, la tête O2M est **complètement supprimée** (`self.cv2 = self.cv3 = None`).

```python
y = self._inference(preds["one2one"])
y = self.postprocess(y)  # top-300, PAS de NMS !
# → tensor[B, 300, 6] : x1,y1,x2,y2,score,class
```

> **Pourquoi ça marche sans NMS ?** Parce que pendant l'entraînement, la tête O2O a appris (grâce au TAL topk2=1) à n'activer qu'**UNE SEULE ancre par objet**. Les 8399 autres ancres ont été entraînées à prédire un score **FAIBLE**. En sélectionnant les top-300, on obtient naturellement ~1 prédiction par objet **sans aucun doublon**.

---

## 🎯 PARTIE 4 — Le TaskAlignedAssigner (TAL) en Détail

Le TAL est l'algorithme qui décide **quelle ancre est responsable de quel objet** — c'est **le cœur du NMS-Free**.

### 4.1 Étape A — Entrées

Le TAL reçoit (exécuté avec `@torch.no_grad()` — pas de gradient !) :
- `pd_scores` : scores de classification sigmoid — `tensor[B, 8400, 80]`
- `pd_bboxes` : boîtes prédites en pixels — `tensor[B, 8400, 4]`
- `anc_points` : centres des 8400 ancres — `tensor[8400, 2]`
- `gt_labels` + `gt_bboxes` + `mask_gt` : les annotations ground truth

### 4.2 Étape B — Filtrage Géométrique + STAL

On garde uniquement les ancres dont le centre est **à l'intérieur** d'une boîte GT :

```python
# 1. STAL : élargir les petites GT
gt_xywh = xyxy2xywh(gt_bboxes)
wh_mask = gt_xywh[..., 2:] < stride[0]  # objets plus petits que 8px
gt_xywh[..., 2:] = where(wh_mask, 16, gt_xywh[..., 2:])
# → Les objets <8px sont élargis à 16px pour la sélection

# 2. Test d'inclusion : le centre de l'ancre est-il dans la GT ?
deltas = cat(anc_points - lt, rb - anc_points)
mask_in_gts = deltas.amin(-1).gt_(0)
```

> **Pourquoi STAL ?** Un piéton à 100m vu depuis un drone occupe peut-être **5×10 pixels**. Avec stride=8, il n'y a **AUCUN centre d'ancre** dans une boîte si petite. STAL élargit artificiellement ces boîtes à 16px pour la sélection des candidats (pas pour le calcul de la perte). Sans STAL, ces petits objets n'auraient **zéro ancre positive** et seraient invisibles à l'entraînement.

### 4.3 Étape C — Métrique d'Alignement

Pour chaque paire (ancre candidate, GT), on calcule un score de qualité :

```
M = score^α × IoU^β
  = score^0.5 × IoU^6.0
```

> **Pourquoi β=6 >> α=0.5 ?** La localisation est **12× plus importante** que la classification. Pour un détecteur NMS-Free, il est **crucial** que l'ancre sélectionnée soit la mieux **localisée**, pas simplement la plus confiante. Il n'y aura **PAS de NMS** pour corriger une mauvaise sélection après coup.

**Exemple concret :**
- Ancre A : score_cls=0.9, IoU=0.3 → M = 0.95 × 0.0007 = **0.0007**
- Ancre B : score_cls=0.5, IoU=0.8 → M = 0.71 × 0.262 = **0.186**
- → L'ancre B (mieux localisée) gagne **largement**, même avec un score de classification inférieur

### 4.4 Étape D — Sélection Top-k

On garde les **k meilleures ancres** par objet selon la métrique M :
- **O2M** : `topk=10` → ~10 ancres positives par objet (signal dense)
- **O2O** : `topk=7` → ~7 ancres candidates (pour l'instant)

### 4.5 Étape E — Résolution des Conflits + Filtrage topk2

**Résolution des conflits** : Si une ancre est assignée à 2 objets, elle va à celui avec le meilleur IoU.

**Filtrage topk2 (O2O uniquement)** :

```python
if self.topk2 != self.topk:    # O2O: topk=7, topk2=1 → True
    align_metric = align_metric * mask_pos
    # Garder seulement le TOP-1 par GT
    best_idx = torch.topk(align_metric, self.topk2, dim=-1).indices
    mask_pos *= scatter(best_idx)  # 1 SEULE ancre/objet
```

> ⚡ **C'est ICI que le NMS-Free se produit !** Ce bloc `if self.topk2 != self.topk` est **LE mécanisme** qui rend YOLO26 NMS-Free. En forçant exactement 1 ancre positive par objet pendant l'entraînement, le réseau apprend que chaque objet ne doit activer qu'**UN SEUL point** de la grille.

> **Pourquoi topk=7 puis topk2=1 (et pas directement topk=1) ?** Si on mettait directement topk=1, le TAL choisirait la meilleure ancre sur la base d'une seule mesure. Avec topk=7, on explore d'abord 7 candidates, on calcule leurs métriques, on résout les conflits, PUIS on filtre à topk2=1. La sélection est **plus robuste et stable**.

### 4.6 Étape F — Normalisation des Scores Cibles

Les target scores sont normalisés pour être **continus** [0,1], pas binaires :

```python
target_scores = target_scores * norm
# Une ancre bien alignée (IoU élevé) → target_score ≈ 0.9
# Une ancre marginale → target_score ≈ 0.3
```

> **Pourquoi des targets continus ?** Si les targets étaient binaires {0,1}, la BCE traiterait toutes les ancres positives de la même façon. Avec des targets continus, le réseau apprend des scores **proportionnels à la qualité réelle** — essentiel pour O2O.

---

## 📊 PARTIE 5 — Comparaison NMS vs NMS-Free

| Critère | NMS-Based (YOLOv8/v11) | NMS-Free (YOLO26) |
|---------|------------------------|-------------------|
| **Stratégie d'assignation** | O2M uniquement (topk=10) | Double : O2M + O2O |
| **Têtes de détection** | Tête unique | Double tête + detach() |
| **Post-traitement** | ❌ NMS obligatoire — O(n²), ~4ms | ✅ Aucun — O(1), sortie directe |
| **Hyperparamètres inférence** | Seuil IoU + seuil confiance à calibrer | Aucun (juste max_det=300) |
| **Régression des coordonnées** | DFL (softmax → NPU-hostile) | Régression directe (export facile) |
| **Pondération de la perte** | Poids fixes | ProgLoss dynamique |
| **Petits objets** | Risque de 0 ancre positive | STAL → garanti |
| **Optimiseur** | SGD / AdamW | MuSGD hybride |
| **Export ONNX/TensorRT** | Difficile (NMS dans/hors graphe) | Natif — graphe pur |
| **mAP (nano, COCO)** | v8n: 37.3% / v11n: 39.5% | **26n: 40.9%** (+1.4 vs v11) |
| **Latence CPU ONNX (nano)** | v8n: ~80.4ms / v11n: ~56.1ms | **26n: 38.9ms** (−30% vs v11) |

---

## 🔗 PARTIE 6 — Graphe de Dépendances des Composants

Chaque composant est **nécessaire** — voici pourquoi et ce qui se passe sans :

| Composant | Quoi ? | Pourquoi ? | Sans ce composant ? |
|-----------|--------|------------|---------------------|
| **Tête O2O** | Assignation 1-vers-1 | Produit 1 prédiction/objet à l'inférence | ❌ Impossible d'être NMS-Free |
| **Tête O2M** | 10 ancres/objet (entraînement seul) | Signal dense pour le backbone | ❌ −2 à −3% mAP, backbone sous-entraîné |
| **detach()** | Coupe gradient O2O→backbone | Empêche gradients contradictoires | ❌ Entraînement instable, non-convergence |
| **TAL topk2=1** | 1 meilleure ancre/objet pour O2O | Force 1 point par objet | ❌ Doublons → NMS nécessaire |
| **ProgLoss** | Poids O2M: 0.8→0.1, O2O: 0.2→0.9 | Transition entraînement→inférence | ❌ Compromis sous-optimal, −0.3-0.5% mAP |
| **STAL** | Élargit GT < 8px à 16px | Ancres candidates pour très petits objets | ❌ Piétons éloignés invisibles |
| **Sans DFL** | reg_max=1, régression directe | Compatible NPU/DSP, pas de softmax | ⚠️ Export impossible sur certains edge devices |
| **MuSGD** | SGD + correction Muon | Stabilise l'entraînement dual-head | ⚠️ Convergence plus lente |
| **β=6.0** | Localisation 12× > classification dans TAL | Ancre choisie = mieux localisée | ❌ Détections décalées |

---

## 🔄 PARTIE 7 — La Chaîne de Causalité Complète (le "Pourquoi" de bout en bout)

```
PROBLÈME : NMS est lent, non-parallélisable, et difficile à exporter
    ↓ pourquoi ne pas simplement le retirer ?

PROBLÈME : Sans NMS, O2M produit ~10 doublons par objet
    ↓ pourquoi pas entraîner avec 1 ancre/objet directement ?

PROBLÈME : Avec topk=1, signal trop faible → backbone sous-entraîné
    ↓ comment avoir un signal dense ET une sortie unique ?

SOLUTION : Double tête O2M (dense) + O2O (unique)
    ↓ mais les gradients se contredisent...

SOLUTION : detach() les features pour O2O
    ↓ mais O2M domine au début, O2O n'apprend pas assez...

SOLUTION : ProgLoss — augmenter O2O progressivement
    ↓ mais comment choisir LA meilleure ancre pour O2O ?

SOLUTION : TAL avec topk=7→topk2=1 + beta=6 (favoriser localisation)
    ↓ mais les petits objets n'ont pas d'ancres candidates...

SOLUTION : STAL élargit les petites GT pour la sélection
    ↓ et pour le déploiement edge ?

SOLUTION : Supprimer DFL (pas de softmax) + MuSGD pour stabilité
    ↓
RÉSULTAT : YOLO26 NMS-Free ✓
```

---

## 📝 PARTIE 8 — Les Fichiers Clés du Code Source

### 8.1 `head.py` — Classe `Detect`

**Rôle** : Tête de détection duale

- `__init__()` : Crée `cv2` (box) + `cv3` (cls) pour O2M, puis `deepcopy` pour O2O, DFL remplacée par `nn.Identity()` quand `reg_max=1`
- `forward()` : O2M sur features normales, O2O sur features `detach()`. En inférence, seule O2O est utilisée
- `postprocess()` : Top-300 par score — **PAS de NMS**
- `fuse()` : Supprime `cv2` et `cv3` (O2M) pour l'inférence optimisée

### 8.2 `tal.py` — Classe `TaskAlignedAssigner`

**Rôle** : Label assignment (le cœur du NMS-Free)

- `__init__()` : `topk`, `topk2`, `alpha=0.5`, `beta=6.0`, `stride=[8,16,32]`
- `forward()` : Exécuté avec `@torch.no_grad()` — pas de gradient
- `select_candidates_in_gts()` : Filtrage géométrique + STAL
- `get_box_metrics()` : Calcul de `M = score^α × IoU^β`
- `select_topk_candidates()` : Sélection top-k
- `select_highest_overlaps()` : Résolution conflits + filtrage `topk2`

### 8.3 `loss.py` — Classes `E2ELoss` + `v8DetectionLoss`

**Rôle** : ProgLoss + perte duale

- `E2ELoss.__init__()` : Crée O2M (topk=10) et O2O (topk=7, topk2=1), poids initiaux 0.8/0.2
- `E2ELoss.__call__()` : `L_total = loss_o2m × w_o2m + loss_o2o × w_o2o`
- `E2ELoss.update()` : Mise à jour des poids ProgLoss à chaque époque
- `E2ELoss.decay()` : Décroissance linéaire de 0.8 → 0.1
- `v8DetectionLoss` : BCE (classification) + CIoU (localisation) + L1 (distances)

---

## 🎯 PARTIE 9 — Résumé pour l'Encadrant

Pour rendre **n'importe quel** détecteur NMS-Free, il faut :

1. ✅ **Un label assigner one-to-one** pendant l'entraînement (TAL topk2=1)
2. ✅ **Un signal auxiliaire dense** pour compenser le signal faible de O2O (tête O2M + topk=10)
3. ✅ **L'isolation des gradients** entre les deux objectifs (`detach()`)
4. ✅ **Un ordonnancement progressif** des poids (ProgLoss)
5. ✅ **Un mécanisme pour les petits objets** (STAL ou équivalent)
6. ✅ **Une assignation qui privilégie la localisation** (β >> α)

**Sans n'importe lequel de ces éléments**, soit le NMS-Free ne fonctionne pas du tout, soit les performances se dégradent significativement.

---

## 📊 PARTIE 10 — Résultats Comparatifs (MS-COCO, modèles Nano)

| Modèle | mAP₅₀₋₉₅ | Latence CPU ONNX | Post-traitement |
|--------|-----------|-------------------|-----------------|
| YOLOv8n | 37.3% | ~80.4 ms | NMS obligatoire |
| YOLOv10n | 38.5% | — | NMS-Free (v1) |
| YOLO11n | 39.5% | ~56.1 ms | NMS obligatoire |
| **YOLO26n** | **40.9%** | **~38.9 ms** | **NMS-Free** |

> YOLO26 nano gagne **+1.4% mAP** par rapport à YOLO11 tout en étant **30% plus rapide** sur CPU — et **sans aucun post-traitement NMS**.

---

## 🔚 CONCLUSION

YOLO26 NMS-Free n'est pas une simple suppression du NMS — c'est une **refonte architecturale complète** du pipeline d'entraînement :

1. **Double tête** (O2M + O2O) pour combiner signal dense et sortie unique
2. **detach()** pour isoler les gradients contradictoires
3. **TAL avec topk2=1** pour forcer l'assignation one-to-one
4. **ProgLoss** pour orchestrer la transition entraînement → inférence
5. **STAL** pour ne pas oublier les petits objets
6. **Suppression de DFL** pour un export natif sur edge devices
7. **MuSGD** pour stabiliser l'entraînement dual-head

Le résultat : un détecteur plus précis, plus rapide, et **nativement compatible avec tout pipeline d'export** — sans aucun hyperparamètre de post-traitement à calibrer.

---

*Sources : Wang et al. (YOLOv10, NeurIPS 2024) · Ultralytics YOLO26 · Chakrabarty — "YOLO26: An Analysis of NMS-Free End to End Framework" · Sapkota et al. · Hidayatullah et al.*
