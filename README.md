# Hedging with Signatures

Couverture (hedging) d'options sous modèles à volatilité stochastique
(**Heston**, **shifted Bergomi rough** et **équation à retard**), par trois
familles de méthodes :

- **Deep Hedging** — réseaux de neurones (VNN, SNN, RNN) entraînés à minimiser
  la variance du P&L de couverture.
- **Méthodes de Fourier** — pricing et calcul du delta optimal par inversion de la
  fonction caractéristique, y compris l'approche **signature-Fourier** pour les
  modèles non-markoviens (Bergomi, équation à retard).
- **Régresseurs linéaires** — la signature du chemin **lead-lag** du prix comme
  features d'une régression linéaire : **Linear BCS** (régression du P&L empirique)
  et **Linear REG** (signature espérée seule, forme quadratique shuffle).

Le code provient de la réorganisation de notebooks de recherche
(`stage00.ipynb`, `signatures_as_linear_regressors.ipynb`) en une bibliothèque
Python modulaire accompagnée de notebooks de démonstration.

> **Référence principale :** Abi Jaber, E. & Gérard, L. (2025).
> *Hedging with Memory: Shallow and Deep Learning with Signatures* /
> *Signature Volatility Models: Pricing and Hedging with Fourier*.

---

## Architecture

Le projet sépare le **code réutilisable** (packages Python) de son
**utilisation** (notebooks). Les modules contiennent les fonctions ; les
notebooks ne font que les importer et les appeler.

```
hedging_signatures/
├── models/             # Simulation des trajectoires et payoffs
│   ├── simulation.py       # simulate_gbm, simulate_heston, simulate_shifted_bergomi
│   ├── delayed_equation.py # simulate_DE, simulate_S_given_DE (volatilité à mémoire)
│   ├── augmented_paths.py  # build_W_hat (t,W), build_Y_hat (t,Σ) pour les signatures
│   └── payoffs.py          # européen, asiatique (géométrique), lookback, polynomial
├── signatures/         # Algèbre tensorielle et signatures de chemins
│   ├── tensor_algebra.py   # mots, produit shuffle, exp, projection (repr. dense)
│   ├── tensor_dict.py      # même algèbre en repr. creuse (dict mot→coef) : half-shuffle,
│   │                       #   concaténation, inverse de Neumann, exponentielles
│   ├── volatility_models.py# σ EXACT : compute_sigma_CIR (Heston), build_sigma_DE (retard)
│   ├── calibration.py      # σ APPRIS (Algorithme 2) : E[Ŵ] de Fawcett, ell^v, calibrate_sigma
│   ├── features.py         # compute_signatures (via iisignature)
│   ├── bergomi_kernel.py   # compute_sigma_t (noyau Bergomi)
│   └── normalization.py    # normalize_psi / normalize_sig
├── deep_hedging/       # Réseaux de neurones et boucles train/eval
│   ├── networks.py         # VNN, SNN, RNN
│   ├── training.py         # get_optimizer, training_loop, training_rnn
│   └── evaluation.py       # evaluate, evaluate_rnn
├── linear_hedging/     # Signatures comme régresseurs linéaires (Linear BCS / REG)
│   ├── lead_lag.py         # transformée de Hoff, build_path_LL, signature_transform
│   ├── black_scholes.py    # prix/delta/P&L BS en forme close (payoff polynomial)
│   ├── linear_regressors.py # Linear BCS (lstsq) et Linear REG (E[S^LL], shuffle Q/b)
│   └── payoff_xi.py        # linéarisation ξ du payoff par régression (call, asiatique)
├── fourier/            # Pricing et hedging par transformée de Fourier
│   ├── black_scholes.py        # fonctions caractéristiques BS de référence
│   ├── black_scholes_asian.py  # benchmark BS naïf de l'asiatique géométrique
│   ├── heston.py               # CF Heston, X0, delta (européen/asiatique)
│   ├── signature_volatility.py # signature-Fourier, σ_t variable dans le temps (Bergomi)
│   └── signature_constant_vol.py # signature-Fourier, σ tenseur constant (Heston, retard)
├── utils/
│   └── plotting.py         # distributions de P&L, comparaison de sigma
├── notebooks/          # Démonstrations 01 à 09 (importent les modules)
├── requirements.txt
└── README.md
```

**Principe :** une formule corrigée dans un module bénéficie automatiquement à
tous les notebooks.

---

## Installation (depuis GitHub)

### Prérequis

- **Python 3.10** (recommandé). La librairie `iisignature` n'a pas de paquet
  précompilé pour Python 3.13 et doit être compilée ; Python 3.10 est le choix
  testé et le plus fiable.
- **Microsoft C++ Build Tools** (Windows) — nécessaires pour compiler
  `iisignature`. Téléchargeables gratuitement
  ([Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).
- (Optionnel) un **GPU NVIDIA** + drivers récents pour accélérer le deep hedging.

### 1. Cloner le dépôt

```bash
git clone <URL_DU_DEPOT>
cd hedging_signatures
```

### 2. Créer et activer un environnement virtuel Python 3.10

```powershell
# Windows (PowerShell)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3.10 -m venv .venv
source .venv/bin/activate
```

> PowerShell : si l'activation est bloquée par la politique d'exécution, lancer
> une fois `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### 3. Installer les dépendances

`iisignature` doit être installé **après** numpy et **sans isolation de build**
(son `setup.py` importe numpy à la compilation). Procéder dans cet ordre :

```bash
python -m pip install --upgrade pip
pip install numpy scipy matplotlib torch tqdm scikit-learn pandas setuptools wheel
pip install --no-build-isolation iisignature
```

Vérifier l'installation :

```bash
python -c "import torch, iisignature, numpy; print('OK', torch.__version__, numpy.__version__)"
```

### 4. (Optionnel) Activer le GPU NVIDIA (CUDA)

Par défaut `pip install torch` installe la version **CPU**. Pour la version CUDA
(beaucoup plus rapide pour l'entraînement) :

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Vérifier :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> `cu128` = CUDA 12.8. Adapter selon le driver NVIDIA (`nvidia-smi`). Les
> notebooks détectent automatiquement le GPU via
> `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.

### 5. Sélectionner l'interpréteur dans l'éditeur

Dans VS Code : `Ctrl+Shift+P` → **Python: Select Interpreter** →
`.\.venv\Scripts\python.exe`. Pour un notebook : « Select Kernel » → `.venv`.

---

## Exécution des notebooks

```bash
jupyter lab        # ou jupyter notebook, ou directement dans VS Code
```

Chaque notebook commence par une cellule d'amorçage qui ajoute la racine du
projet au `sys.path`, de sorte que les modules sont importables depuis le
sous-dossier `notebooks/`. Aucune configuration supplémentaire n'est requise :
exécuter les cellules de haut en bas.

### Notebooks disponibles

Les notebooks sont **autonomes et indépendants** : chacun re-simule ses propres
données. L'ordre 01 → 07 est pédagogique.

| Notebook | Contenu |
|----------|---------|
| `01_simulation_models`        | Simulations Heston / Bergomi et payoffs de base |
| `02_signature_algebra_tests`  | Algèbre tensorielle et calcul de signatures |
| `03_deep_hedging_heston`      | Deep hedging Heston : **VNN, SNN, RNN** × options **européenne, asiatique, lookback**. Sauvegarde les P&L de test → `results/heston_pnl.npz` |
| `04_deep_hedging_bergomi`     | Deep hedging shifted Bergomi : **VNN, SNN, RNN** × **européenne, asiatique, lookback**. Sauvegarde → `results/bergomi_pnl.npz` |
| `05_fourier_heston`           | **Comparaison** Heston : superpose VNN/SNN/RNN (chargés depuis `results/`) + benchmark Fourier (eur/asian) ou BS naïf (lookback) |
| `06_fourier_signature_bergomi`| Signature-Fourier Bergomi : **européenne** (σ rescalée + σ apprise) et **asiatique** (σ apprise) |
| `07_comparison_results`       | Tableau de comparaison final → `comparison_results.csv` |
| `08_signature_linear_regressors` | Signatures comme régresseurs linéaires sous **Black-Scholes** (payoff polynomial) : benchmark BS + **Linear BCS** (régression du P&L) + **Linear REG** (signature espérée, forme quadratique shuffle). Reproduit le Tableau 5 du papier |
| `09_signature_heston_de`      | Volatilité stochastique **représentée par une signature**. §5.2 **Heston**, call européen ATM (Tableau 6) et §5.3 **équation à retard**, call asiatique géométrique (Tableau 7). Compare à chaque fois **Linear BCS / Linear REG / Fourier BCS / Fourier REG** + une référence (Fourier exact sous Heston, BS naïf sous retard) |

> **Dépendance NB03 → NB05.** Le notebook 05 ne réentraîne pas les réseaux : il
> **charge** les P&L sauvegardés par le notebook 03 (mêmes paramètres, même jeu
> test `seed=123`, donc courbes comparables). **Exécuter NB03 avant NB05.** Si
> `results/heston_pnl.npz` est absent, NB05 affiche un message explicite.

> Le **lookback** n'a pas de formule signature-Fourier : il est couvert par le
> deep hedging (NB04) et, sous Heston, par un benchmark Black-Scholes (NB05).

### Tailles de calcul

Les notebooks 03 à 09 utilisent les **tailles complètes** du stage
(`I_train=10000`, `I_test=20000`, `J=126`, signatures d'ordre 4–5). Conséquences :

- **Deep hedging (NB03/04)** : `num_epochs=64`, batch 64. Rapide sur **GPU**,
  lent sur CPU.
- **Signature-Fourier (NB06/NB09)** : les deltas analytiques sont **longs**
  (plusieurs minutes par option sur 20 000 trajectoires).
- **Linear REG (NB08/NB09)** : la signature espérée `E[S^LL]` est calculée à
  l'ordre `2M = 8`, soit 9840 colonnes (~800 Mo pour 10 000 trajectoires).
- **NB09** est le plus lourd (~25 min) : deux sections × (E[S^LL] à l'ordre 8 +
  deltas Fourier + une calibration par gradient **numérique** sur 30 paramètres).

Pour un test rapide, réduire dans les cellules de paramètres : `NUM_EPOCHS`,
`I_train`/`I_test`, `n_laguerre`, `N_steps`, `depth`/`M`.

### À propos de la forme des signatures `(I, J, 120)`

`compute_signatures(states, depth=4)` renvoie un tableau de forme
`(I, J, 120)` — c'est **normal et attendu** :

- **I** = nombre de trajectoires (ex. 10000 en train, 20000 en test) ;
- **J = 126** = nombre de pas de temps (mode *stream* : une signature par pas,
  de `t_1` à `t_J`) ;
- **120** = dimension de la signature tronquée d'un chemin à **3 canaux**
  `(t, S, σ)` à l'ordre 4 : `3 + 3² + 3³ + 3⁴ = 3 + 9 + 27 + 81 = 120`.

C'est exactement le vecteur d'entrée du **SNN** (`input_features=120`).

---

## Modèles et conventions

- **Heston** : variance stochastique avec retour à la moyenne ; CF européenne en
  forme close, CF asiatique via un système de Riccati résolu par RK4.
- **Shifted Bergomi (rough)** : volatilité dirigée par un noyau fractionnaire
  (paramètre de Hurst `H`) ; couverture via les **signatures** du mouvement
  brownien augmenté `Ŵ = (t, W)` et une équation de Riccati dans l'algèbre
  tensorielle. Modèle **non-markovien** → le SNN (mémoire du chemin) y est
  attendu meilleur que le VNN/RNN.
- **Équation à retard** : `dΣ = κ(θ−Σ)dt + (η+αΣ+ςZ)dW`, `dZ = (Σ−λZ)dt`. `Z` est
  une moyenne exponentiellement pondérée du passé de `Σ` : c'est elle qui crée la
  **mémoire**. ⚠️ `Σ` n'y est **pas** contraint positif (min ≈ −0.07) : tout
  dénominateur en `Σ_t` exige un plancher **signé** (`fourier.signed_floor`), et
  non `np.maximum`, qui inverserait le signe des trajectoires négatives.
- Les méthodes de Fourier utilisent une quadrature de **Gauss-Laguerre** et un
  schéma de type Carr-Madan avec terme de contrôle Black-Scholes.
- Le P&L de couverture vaut `X0 + Σⱼ αⱼ·dSⱼ − payoff`, où `X0` est le prix
  initial et `αⱼ` la stratégie (delta) à chaque pas.

### Volatilité représentée par une signature (NB09)

`Σ_t = ⟨σ, S(Ŵ)_{0,t}⟩` avec `σ` **déterministe** : toute la dépendance-chemin est
portée par la signature, `σ` ne fait que la pondérer. D'où la distinction :

- **Fourier BCS** — `σ` **exact**, en forme close (`signatures.volatility_models`) ;
- **Fourier REG** — `σ` **calibré** sur les seuls moments (`signatures.calibration`,
  Algorithme 2), sans jamais voir la forme close.

Le niveau de base `σ[∅]` n'est **pas** calibré et doit être restauré avant injection
dans le Riccati : le mot `(2,)` de la signature de `Ŷ = (t, Σ)` mesure l'**incrément**
`Σ_t − Σ_0`, pas le niveau.

### Convention du mot vide

`signatures.generate_words(M, d)` **inclut** le mot vide en position 0, alors que
`iisignature.sig` ne l'émet pas. Conséquence :

- pour indexer les **colonnes d'une signature** : `linear_hedging.sig_words(M, d)`
  (= `generate_words(M, d)[1:]`) ;
- pour les **tenseurs de Fourier** (`words` / `word_idx`) : `generate_words(M, 2)`
  tel quel, le mot vide y est attendu à l'index 0.

Se tromper de convention **décale silencieusement toutes les colonnes** au lieu de
lever une erreur.

---

## Reproductibilité

Pour figer les versions exactes de l'environnement testé :

```bash
pip freeze > requirements.lock.txt
```

Environnement de référence : Python 3.10.11, numpy 2.2.6, torch 2.11.0+cu128
(CUDA 12.8), iisignature 0.24, scikit-learn 1.7, scipy 1.15.
