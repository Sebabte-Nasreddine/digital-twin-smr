# Guide complet : L'optimisation appliquée au Jumeau Numérique SMR

---

## Table des matières

1. [Qu'est-ce que l'optimisation ?](#1-quest-ce-que-loptimisation)
2. [Les ingrédients d'un problème d'optimisation](#2-les-ingrédients-dun-problème-doptimisation)
3. [Types de problèmes](#3-types-de-problèmes)
4. [Les algorithmes : comment chercher le meilleur point ?](#4-les-algorithmes--comment-chercher-le-meilleur-point-)
5. [Notre problème SMR : modélisation complète](#5-notre-problème-smr--modélisation-complète)
6. [Le code expliqué ligne par ligne](#6-le-code-expliqué-ligne-par-ligne)
7. [Optimisation globale vs contrainte : la différence fondamentale](#7-optimisation-globale-vs-contrainte--la-différence-fondamentale)
8. [Pièges courants et comment les éviter](#8-pièges-courants-et-comment-les-éviter)
9. [Aller plus loin](#9-aller-plus-loin)

---

## 1. Qu'est-ce que l'optimisation ?

L'optimisation, c'est répondre à la question :

> **"Parmi toutes les solutions possibles, laquelle est la meilleure ?"**

### Un exemple de la vie quotidienne

Imaginez que vous voulez aller de Paris à Lyon en voiture. Vous avez plusieurs routes possibles. Chaque route a une durée et une consommation d'essence. Vous voulez aller le plus vite possible tout en dépensant le moins d'essence possible. C'est un problème d'optimisation.

- **Variables de décision** : quelle route choisir ?
- **Objectifs** : minimiser le temps, minimiser la consommation
- **Contraintes** : rester sur des routes légales, ne pas tomber en panne

### Dans notre réacteur SMR

On veut trouver la température et le temps de contact qui donnent les meilleures performances (conversion, rendement, dépôt de carbone). C'est exactement la même structure.

---

## 2. Les ingrédients d'un problème d'optimisation

Tout problème d'optimisation a **trois composants obligatoires** :

### 2.1 Les variables de décision (les leviers)

Ce sont les paramètres qu'on peut régler. On les note souvent **x**.

```
x = [x₁, x₂, ..., xₙ]
```

Dans notre cas :
- x₁ = Température (X1), entre 500°C et 800°C
- x₂ = Temps de contact (X2), entre 5.57 et 77.08 g·h/mol

On a donc **n = 2 variables**. C'est un espace de recherche 2D.

### 2.2 La fonction objectif (ce qu'on veut optimiser)

C'est la mesure de qualité d'une solution. On la note **f(x)**.

Par convention, on **minimise** toujours. Pour maximiser quelque chose, on minimise son négatif.

```
Minimiser  f(x)         ← problème standard
Maximiser  g(x)  ≡  Minimiser  -g(x)
```

### 2.3 Les contraintes (les limites à respecter)

Elles définissent l'**espace faisable** : l'ensemble des solutions légales.

```
Contraintes d'inégalité :  gᵢ(x) ≤ 0
Contraintes d'égalité   :  hⱼ(x) = 0
Bornes                  :  xₗₒ ≤ x ≤ xₕᵢ
```

### La formulation mathématique générale

```
Minimiser    f(x)
Sous :       gᵢ(x) ≤ 0    pour i = 1, ..., m
             hⱼ(x) = 0    pour j = 1, ..., p
             xₗₒ ≤ x ≤ xₕᵢ
```

---

## 3. Types de problèmes

### 3.1 Optimisation sans contraintes vs avec contraintes

**Sans contraintes** : on peut aller partout dans l'espace de recherche.

```python
# La solution est n'importe quel (T, X2) dans les bornes physiques
minimize f(x),  500 ≤ T ≤ 800,  5.57 ≤ X2 ≤ 77.08
```

**Avec contraintes** : certaines solutions sont interdites même si elles semblent bonnes.

```python
# En plus des bornes, des exigences opératoires
minimize f(x)
sous :  Y1(x) ≥ 85%      ← conversion minimale exigée
        Y2(x) ≥ 10%      ← rendement H₂ minimal
        Y4(x) ≤ 30%      ← dépôt de carbone maximal
```

### 3.2 Optimisation mono-objectif vs multi-objectif

**Mono-objectif** : un seul critère. La solution optimale est **unique**.

```
Minimiser  -Y1(x)     ← juste maximiser la conversion
```

**Multi-objectif** : plusieurs critères souvent en conflit. La solution est un **ensemble** appelé front de Pareto.

```
Minimiser  [-Y1(x),  -Y2(x),  +Y4(x)]     ← trois objectifs simultanément
```

### 3.3 Pondération : transformer le multi-objectif en mono-objectif

La technique la plus simple : additionner les objectifs avec des poids.

```
f(x) = -w₁·Y1(x) - w₂·Y2(x) - w₃·Y3(x) + |w₄|·Y4(x)
```

- Si w₁ = 3.0 et w₂ = 0.0 → on ne se soucie que de la conversion
- Les poids expriment les **préférences** de l'ingénieur

**Attention** : avant d'additionner, il faut normaliser les objectifs pour qu'ils soient sur la même échelle.

```
Y_normalisé = (Y - Y_min) / (Y_max - Y_min)    ← entre 0 et 1
```

---

## 4. Les algorithmes : comment chercher le meilleur point ?

### 4.1 Recherche exhaustive sur grille (Grid Search)

**Principe** : on divise l'espace en une grille régulière et on évalue f(x) en chaque nœud.

```
T    : [500, 504, 508, ..., 796, 800]   ← 80 valeurs
X2   : [5.57, 6.48, ..., 76.18, 77.08] ← 80 valeurs
Total: 80 × 80 = 6400 points évalués
```

**Avantages** :
- Simple à comprendre et à implémenter
- Garantit de couvrir tout l'espace
- Facile à vectoriser (évaluation en batch)

**Inconvénients** :
- Explose avec le nombre de dimensions (malédiction de la dimensionnalité)
- Ne garantit pas de trouver l'exact optimum (la grille peut le rater)
- Avec 10 variables et 80 points chacune : 80¹⁰ ≈ 10¹⁹ points → impossible

```python
# Implémentation naïve (LENTE — appels un par un)
for x1 in np.linspace(500, 800, 80):
    for x2 in np.linspace(5.57, 77.08, 80):
        score = f(x1, x2)     # 6400 appels séparés

# Implémentation vectorisée (RAPIDE — un seul appel batch)
df = predict_grid((500, 800), (5.57, 77.08), n_points=80)   # 1 seul appel
scores = -w1*df["Y1"] - w2*df["Y2"] + w4*df["Y4"]           # calcul numpy
```

### 4.2 Optimisation locale par gradient (L-BFGS-B)

**Principe** : partir d'un point de départ, calculer le gradient (la pente), descendre dans la direction opposée, répéter.

```
x_nouveau = x_ancien - α · ∇f(x_ancien)

où α = pas de gradient
   ∇f = gradient (direction de montée de f)
```

**L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds) est une version avancée qui :
- Estime le gradient numériquement (pas besoin de formule analytique)
- Respecte les bornes xₗₒ ≤ x ≤ xₕᵢ
- Utilise peu de mémoire

**Avantages** :
- Très précis une fois proche de l'optimum
- Rapide (quelques dizaines d'itérations)

**Inconvénients** :
- Peut rester bloqué dans un optimum **local** (un creux local, pas le plus profond)
- Dépend fortement du point de départ

```
Mauvais point de départ → optimum local (pas forcément le global)
Bon point de départ     → optimum global (si la fonction est "gentille")
```

### 4.3 Stratégie combinée : Grille + Affinement local

C'est ce qu'on utilise dans notre code. C'est le meilleur compromis :

```
Étape 1 : Grille vectorisée (80×80)
          → trouver les 5 meilleurs points de départ candidats

Étape 2 : L-BFGS-B depuis chacun des 5 points
          → affiner précisément chaque candidat

Étape 3 : Garder le meilleur résultat final
```

**Pourquoi 5 points de départ ?** Pour réduire le risque de rester dans un optimum local. On explore plusieurs "bassins d'attraction" différents.

```
          f(x)
           │
    ───────┤           ← optimum global (on veut celui-là)
           │     ╭──╮
           │  ╭──╯  ╰──╮
           │──╯         ╰────
           └──────────────────► x
           ↑                ↑
     optimum local    point de départ B
          ↑
    point de départ A
    (bloqué ici)
```

---

## 5. Notre problème SMR : modélisation complète

### 5.1 Données du problème

```
Variables de décision :
  x₁ = X1 = Température (°C)         ∈ [500, 800]
  x₂ = X2 = Temps de contact (g·h/mol) ∈ [5.57, 77.08]

Sorties prédites par les modèles ML (cascade RandomForest) :
  Y1(x) = Conversion CH₄ (%)         ∈ [14.23, 100]
  Y2(x) = Rendement H₂ (%)           ∈ [0.05, 35.36]
  Y3(x) = Sélectivité CO₂ (%)        ∈ [0.26, 15.89]
  Y4(x) = Dépôt de carbone (%)       ∈ [-0.16, 99.88]
```

### 5.2 Cascade de modèles (important à comprendre)

Les sorties ne sont pas indépendantes. Elles forment une **cascade** :

```
X1, X2  ──→  Modèle Y1  ──→  Y1
X1, X2, Y1 ──→  Modèle Y2  ──→  Y2
X1, X2, Y2 ──→  Modèle Y3  ──→  Y3
X1, X2, Y2 ──→  Modèle Y4  ──→  Y4
```

Conséquence : on ne peut pas optimiser Y4 sans passer par Y1 et Y2. Tout est lié.

### 5.3 Formulation de l'optimisation globale (pondérée)

```
Minimiser  f(x) = -w₁·Ŷ1(x) - w₂·Ŷ2(x) - w₃·Ŷ3(x) - w₄·Ŷ4(x)

où  Ŷₖ(x) = (Yₖ(x) - Yₖ_min) / (Yₖ_max - Yₖ_min)   ← normalisé entre 0 et 1

Sous : 500 ≤ x₁ ≤ 800
       5.57 ≤ x₂ ≤ 77.08
```

**Note** : w₄ est négatif dans l'interface (pénalité). Dans la formule, on minimise donc +|w₄|·Ŷ4(x).

### 5.4 Formulation de l'optimisation contrainte

```
Minimiser  f(x) = -(Y1(x) + Y2(x) - 0.5·Y4(x))

Sous : 500 ≤ x₁ ≤ 800
       5.57 ≤ x₂ ≤ 77.08
       Y1(x) ≥ min_Y1      ← contrainte d'inégalité (g₁)
       Y2(x) ≥ min_Y2      ← contrainte d'inégalité (g₂)
       Y4(x) ≤ max_Y4      ← contrainte d'inégalité (g₃)
```

---

## 6. Le code expliqué ligne par ligne

### 6.1 Fichier `src/optimize.py`

#### La fonction objectif (globale)

```python
def run_single_objective(weights=None):

    # 1. Charger les bornes depuis la config
    bounds = _get_bounds()
    # bounds = ((500, 800), (5.57, 77.08))

    # 2. Récupérer les plages min/max de chaque sortie (pour normaliser)
    y_ranges = {k: (v["min"], v["max"]) for k, v in cfg["outputs"].items()}
    # y_ranges = {"Y1": (14.23, 100), "Y2": (0.05, 35.36), ...}

    # 3. Évaluer TOUTE la grille en un seul appel vectorisé
    df = predict_grid((500, 800), (5.57, 77.08), n_points=80)
    # df a 6400 lignes, colonnes : X1, X2, Y1, Y2, Y3, Y4

    # 4. Calculer le score pour chaque point de la grille
    scores = np.zeros(len(df))
    for k, w in weights.items():
        lo, hi = y_ranges[k]
        p_norm = (df[k].values - lo) / (hi - lo)   # normalisation 0→1
        scores -= w * p_norm                         # on minimise donc on soustrait
    # scores[i] = score du point i (plus bas = meilleur)

    # 5. Prendre les 5 meilleurs points de la grille
    top_idx = np.argsort(scores)[:5]
    top_pts = df[["X1", "X2"]].values[top_idx]
    # top_pts = [[T₁, X2₁], [T₂, X2₂], ..., [T₅, X2₅]]

    # 6. Définir la fonction à minimiser point par point (pour L-BFGS-B)
    def _f(x):
        p = predict(float(x[0]), float(x[1]))   # un seul point
        s = 0.0
        for k, w in weights.items():
            lo, hi = y_ranges[k]
            p_norm = (p[k] - lo) / (hi - lo)
            s -= w * p_norm
        return s

    # 7. Affiner depuis chacun des 5 meilleurs points
    best_x, best_score = None, np.inf
    for pt in top_pts:
        res = minimize(
            _f,                    # fonction à minimiser
            x0=pt,                 # point de départ
            method="L-BFGS-B",    # algorithme
            bounds=bounds,         # respecter les bornes
            options={"maxiter": 200, "ftol": 1e-9}
        )
        if res.fun < best_score:   # garder le meilleur
            best_score = res.fun
            best_x = res.x

    # 8. Prédire les sorties au point optimal
    preds = predict(float(best_x[0]), float(best_x[1]))
    return {"X1": best_x[0], "X2": best_x[1], **preds}
```

#### La fonction de recherche contrainte

```python
def find_optimal(min_Y1=90, min_Y2=10, max_Y4=40):

    # 1. Évaluer toute la grille
    df = predict_grid((500, 800), (5.57, 77.08), n_points=80)

    # 2. Filtrer : garder uniquement les points qui respectent TOUTES les contraintes
    mask = (df["Y1"] >= min_Y1) & (df["Y2"] >= min_Y2) & (df["Y4"] <= max_Y4)
    feasible = df[mask]
    # feasible = sous-ensemble de df où tout est respecté

    if feasible.empty:
        # Diagnostic : quelle contrainte bloque ?
        partial = df[(df["Y1"] >= min_Y1) & (df["Y2"] >= min_Y2)]
        min_y4_achievable = partial["Y4"].min()
        return {"infeasible": True, "min_y4_achievable": min_y4_achievable, ...}

    # 3. Parmi les points faisables, calculer un score et prendre le meilleur
    scores = feasible["Y1"] + feasible["Y2"] - 0.5 * feasible["Y4"]
    best_row = feasible.iloc[scores.argmax()]
    x0 = [best_row["X1"], best_row["X2"]]

    # 4. Affiner avec L-BFGS-B en rejetant les points hors contraintes
    def _neg_score(x):
        p = predict(float(x[0]), float(x[1]))
        if p["Y1"] < min_Y1 or p["Y2"] < min_Y2 or p["Y4"] > max_Y4:
            return 1e6    # pénalité infinie = solution interdite
        return -(p["Y1"] + p["Y2"] - 0.5 * p["Y4"])

    res = minimize(_neg_score, x0=x0, method="L-BFGS-B", bounds=bounds)
    return {"X1": res.x[0], "X2": res.x[1], ...}
```

### 6.2 Pourquoi `predict_grid` et pas une boucle ?

```python
# ❌ LENT : 6400 appels séparés au modèle RandomForest
for pt in grid_pts:
    score = f(pt)       # chaque appel charge et exécute le modèle

# ✅ RAPIDE : 1 seul appel vectorisé (numpy/sklearn en batch)
df = predict_grid(...)  # sklearn prédit 6400 points en une seule opération
scores = calcul_numpy   # opérations matricielles ultra-rapides
```

**Rapport de vitesse** : typiquement 100× à 1000× plus rapide selon le modèle.

---

## 7. Optimisation globale vs contrainte : la différence fondamentale

### Ce que cherche chacune

```
Optimisation GLOBALE  :  meilleur x selon f(x) uniquement
                         → "le plus beau point du paysage"

Optimisation CONTRAINTE : meilleur x selon f(x) ET respectant g(x)
                          → "le plus beau point du paysage ACCESSIBLE"
```

### Représentation visuelle

```
Espace de toutes les solutions possibles
┌──────────────────────────────────────┐
│                                      │
│    ★ ← Optimum global               │
│    (meilleur score, mais Y4=99%)     │
│                                      │
│   ╔══════════════════╗               │
│   ║  Zone faisable   ║               │
│   ║  (contraintes    ║               │
│   ║   respectées)    ║               │
│   ║                ● ← Optimum       │
│   ║                  ║  contraint    │
│   ╚══════════════════╝               │
│                                      │
└──────────────────────────────────────┘
```

L'optimum global ★ est **hors de la zone faisable** → il est interdit.
L'optimum contraint ● est le **meilleur point dans la zone verte**.

### Exemple numérique dans notre SMR

**Contexte** : poids Y1=0, Y2=0, Y3=3, Y4=0 (maximiser sélectivité CO₂ uniquement)

| | Optimum global | Optimum contraint |
|---|---|---|
| Température | 500°C (basse) | 760°C (haute) |
| Temps de contact | 77 g·h/mol (max) | 15 g·h/mol |
| Y1 Conversion | 78% | 92% ✓ |
| Y2 Rendement H₂ | 0.2% | 24% ✓ |
| Y3 Sélectivité CO₂ | **15.8%** (max!) | 11.2% |
| Y4 Dépôt carbone | **99.5%** ← catastrophe | 18% ✓ |

L'optimum global maximise Y3 mais le réacteur serait inutilisable (99% de carbone).
L'optimum contraint est opérable mais Y3 est réduit.

---

## 8. Pièges courants et comment les éviter

### Piège 1 : Contraintes trop strictes → espace faisable vide

```
Y4 ≤ 14%  →  require Y2 ≥ 32.7%  →  require T ≥ 795°C
              (car Y4 ≈ 99.7 - 2.63×Y2 dans ce modèle)

Si en plus Y1 ≥ 95% et Y2 ≥ 30% → l'intersection peut être vide !
```

**Solution** : relâcher une contrainte à la fois. Notre code dit maintenant exactement quelle contrainte bloque et quelle valeur minimale est réellement atteignable.

### Piège 2 : Optimum local

```
L-BFGS-B depuis un mauvais point de départ :

   f(x)
    │   ╭──╮           ╭──────
    │ ╭─╯  ╰──╮     ╭──╯
    │─╯        ╰─────╯
    └──────────────────────► x
         ↑         ↑
     L-BFGS-B   Vrai optimum
     s'arrête    global
     ici !
```

**Solution** : toujours partir de plusieurs points de départ (on utilise les top-5 de la grille).

### Piège 3 : Objectifs non normalisés

```python
# ❌ MAUVAIS : Y1 va de 14 à 100, Y4 va de 0 à 99
f = -w1*Y1 - w4*Y4
# Si w1=w4=1 : Y1 contribue 86 unités max, Y4 contribue 99 unités max
# → Y4 domine artificiellement même avec w1=w4 !

# ✅ BON : tout entre 0 et 1
Y1_norm = (Y1 - 14.23) / (100 - 14.23)   # contribue au max 1 unité
Y4_norm = (Y4 - (-0.16)) / (99.88 - (-0.16))  # contribue au max 1 unité
f = -w1*Y1_norm + w4*Y4_norm              # les poids ont maintenant le même sens
```

### Piège 4 : Cache Python (spécifique à Streamlit)

Streamlit recharge `app.py` à chaque interaction mais **conserve les modules importés** en mémoire. Si vous modifiez `optimize.py`, Streamlit ne le recharge pas automatiquement.

```python
# Solution : forcer le rechargement avant chaque utilisation
import sys
sys.modules.pop("optimize", None)    # effacer du cache
from optimize import find_optimal    # reimporter depuis le disque
```

---

## 9. Aller plus loin

### 9.1 Optimisation multi-objectif et front de Pareto

Quand on ne veut pas choisir a priori les poids, on calcule le **front de Pareto** : l'ensemble de toutes les solutions telles qu'aucune n'est meilleure sur tous les critères à la fois.

```
         Y4 (dépôt, à minimiser)
         │
    100% ┤ ●
         │   ●
         │     ●
     50% ┤       ●
         │         ●
     14% ┤           ★ ← Front de Pareto
         └────────────────────► Y1 (conversion, à maximiser)
                           100%
```

Chaque point ● sur le front est "Pareto-optimal" : augmenter Y1 augmente forcément Y4.

### 9.2 Algorithmes plus puissants pour des espaces plus complexes

| Algorithme | Quand l'utiliser |
|---|---|
| Grid + L-BFGS-B (notre choix) | 2-5 variables, fonction continue |
| Évolution différentielle (`differential_evolution`) | Fonctions non-convexes, nombreux optima locaux |
| Algorithmes génétiques (NSGA-II) | Multi-objectif, grands espaces |
| Optimisation bayésienne | Évaluations très coûteuses (simulations CFD) |

### 9.3 Pourquoi un jumeau numérique facilite l'optimisation

Sans jumeau numérique, optimiser un réacteur nécessiterait des centaines d'expériences physiques (cher, long, dangereux). Avec le jumeau numérique :

```
Expérience physique  : 1 point = plusieurs heures + coût du réactif
Modèle ML (surrogate): 6400 points = quelques millisecondes

→ On peut explorer l'espace en entier, trouver l'optimum, PUIS faire
  l'expérience physique uniquement pour valider la solution optimale.
```

C'est le principe du **surrogate-based optimization** (optimisation basée sur modèle de substitution).

---

## Résumé en une page

```
PROBLÈME D'OPTIMISATION
═══════════════════════

1. VARIABLES       x = [T, X2]                   ← ce qu'on règle

2. MODÈLE          Y = RandomForest(x)            ← prédit les sorties
   (cascade)       Y1 = f(T, X2)
                   Y2 = f(T, X2, Y1)
                   Y4 = f(T, X2, Y2)

3. OBJECTIF        f(x) = -w1·Y1̂ - w2·Y2̂ - w3·Y3̂ + |w4|·Y4̂   ← à minimiser
   (global)        Ŷk = (Yk - min) / (max - min)              ← normalisé

4. CONTRAINTES     Y1(x) ≥ min_Y1                ← zone faisable
   (opératoires)   Y2(x) ≥ min_Y2
                   Y4(x) ≤ max_Y4

5. ALGORITHME
   Étape 1 : predict_grid(80×80) → 6400 prédictions vectorisées
   Étape 2 : trier, prendre top-5
   Étape 3 : L-BFGS-B depuis chacun des 5 points
   Étape 4 : retourner le meilleur résultat

6. RÉSULTAT        X1* = 798°C,  X2* = 5.57 g·h/mol
                   Y1* = 97.3%,  Y2* = 32.8%,  Y4* = 13.6%
```
