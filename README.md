# Hedging with Signatures

Couverture (hedging) d'options sous modèles à volatilité stochastique
(**Heston** et **shifted Bergomi rough**), par deux familles de méthodes :

- **Deep Hedging** — réseaux de neurones (VNN, SNN, RNN) entraînés à minimiser
  la variance du P&L de couverture.
- **Méthodes de Fourier** — pricing et calcul du delta par inversion de la
  fonction caractéristique, y compris l'approche **signature-Fourier** pour le
  modèle Bergomi.

Le code provient de la réorganisation d'un notebook de recherche unique
(`stage00.ipynb`) en une bibliothèque Python modulaire accompagnée de notebooks
de démonstration.

## Architecture

Le projet sépare le **code réutilisable** (packages Python) de son
**utilisation** (notebooks). Les modules contiennent les fonctions ; les
notebooks ne font que les importer et les appeler.

```
hedging_signatures/
├── models/             # Simulation des trajectoires et payoffs
│   ├── simulation.py       # simulate_heston, simulate_shifted_bergomi
│   └── payoffs.py          # européen, asiatique, lookback
├── signatures/         # Algèbre tensorielle et signatures de chemins
│   ├── tensor_algebra.py   # mots, produit shuffle, exp, projection
│   ├── features.py         # compute_signatures (via iisignature)
│   ├── bergomi_kernel.py   # compute_sigma_t (noyau Bergomi)
│   └── normalization.py    # normalize_psi / normalize_sig
├── deep_hedging/       # Réseaux de neurones et boucles train/eval
│   ├── networks.py         # VNN, SNN, RNN
│   ├── training.py         # get_optimizer, training_loop, training_rnn
│   └── evaluation.py       # evaluate, evaluate_rnn
├── fourier/            # Pricing et hedging par transformée de Fourier
│   ├── black_scholes.py        # fonctions caractéristiques BS de référence
│   ├── heston.py               # CF Heston, X0, delta (européen/asiatique)
│   └── signature_volatility.py # approche signature-Fourier (Bergomi)
├── utils/
│   └── plotting.py         # distributions de P&L, comparaison de sigma
├── notebooks/          # Démonstrations 01 à 07 (importent les modules)
└── requirements.txt
```

**Principe :** une formule corrigée dans un module bénéficie automatiquement à
tous les notebooks.

## Installation

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

> **Note Windows — `iisignature`.** Ce package (utilisé par
> `signatures/features.py`) nécessite un compilateur C++. Si l'installation
> échoue, installe d'abord les **Microsoft C++ Build Tools**, puis relance la
> commande. Toutes les autres dépendances (numpy, scipy, torch, pandas,
> scikit-learn, tqdm, matplotlib) s'installent sans configuration.

## Exécution des notebooks

```bash
jupyter lab        # ou jupyter notebook, ou directement dans VS Code
```

Chaque notebook commence par une cellule d'amorçage qui ajoute la racine du
projet au `sys.path`, de sorte que les modules sont importables depuis le
sous-dossier `notebooks/` :

```python
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Aucune configuration supplémentaire n'est requise : exécute les cellules de haut
en bas.

### Les notebooks sont autonomes et indépendants

Chaque notebook **re-simule ses propres données** en **tailles réduites** (pour
une exécution de contrôle rapide). Tu peux donc lancer n'importe lequel
directement, sans exécuter les précédents. L'ordre 01 → 07 est purement
pédagogique.

| Notebook | Rôle |
|----------|------|
| `01_simulation_models`        | Simulations Heston / Bergomi et payoffs de base |
| `02_signature_algebra_tests`  | Algèbre tensorielle et calcul de signatures |
| `03_deep_hedging_heston`      | Deep hedging sous Heston (VNN / SNN / RNN) |
| `04_deep_hedging_bergomi`     | Deep hedging sous shifted Bergomi |
| `05_fourier_heston`           | Pricing et hedging par Fourier sous Heston |
| `06_fourier_signature_bergomi`| Hedging par signature-Fourier sous Bergomi |
| `07_comparison_results`       | Tableau de comparaison final → `comparison_results.csv` |

Il n'y a **pas de fichiers de données partagés** entre notebooks ; seul le
notebook 07 écrit un CSV de synthèse.

### Passer à l'échelle « grandeur réelle »

Les notebooks utilisent volontairement de petites tailles. Pour des résultats
complets, augmente les paramètres dans les cellules dédiées, par exemple :

- nombre de trajectoires : `I_train`, `I_test`, `I`
- entraînement : `num_epochs`
- quadrature de Fourier : `n_laguerre`, `N_steps`
- ordre de troncature des signatures : `depth`, `M`

## Modèles et conventions

- **Heston** : variance stochastique avec retour à la moyenne ; CF européenne en
  forme close, CF asiatique via un système de Riccati résolu par RK4.
- **Shifted Bergomi (rough)** : volatilité dirigée par un noyau fractionnaire
  (paramètre de Hurst `H`) ; couverture via les **signatures** du mouvement
  brownien sous-jacent et une équation de Riccati dans l'algèbre tensorielle.
- Les méthodes de Fourier utilisent une quadrature de **Gauss-Laguerre** et un
  schéma de type Carr-Madan avec terme de contrôle Black-Scholes.
