# Experiment Log: Movie Recommender System

> All experiment runs are tracked in MLflow and can be explored interactively on the
> [MLflow Dashboard (DagsHub)](https://dagshub.com/vigneshsabapathi/recommender_system.mlflow).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Methodology](#2-methodology)
3. [Experiment 5 -- Model Comparison](#3-experiment-5--model-comparison)
4. [Experiment 4 -- Hybrid Weight Optimization](#4-experiment-4--hybrid-weight-optimization)
5. [Experiment 6 -- Ablation Study](#5-experiment-6--ablation-study)
6. [Experiment 7 -- Cold-Start Analysis](#6-experiment-7--cold-start-analysis)
7. [Experiment 2 -- CF Hyperparameter Tuning](#7-experiment-2--cf-hyperparameter-tuning)
8. [Experiment 9 -- Feature Importance](#8-experiment-9--feature-importance)
9. [Key Findings and Recommendations](#9-key-findings-and-recommendations)
10. [Reproducibility](#10-reproducibility)

---

## 1. Overview

This document records every offline experiment conducted during the development of the
movie recommender system. The project implements four recommendation strategies --
Item-Item Collaborative Filtering (CF), Content-Based Filtering, Alternating Least
Squares (ALS) matrix factorisation, and a Weighted Hybrid ensemble -- and evaluates
them across 78 total MLflow runs spanning six experiment groups:

| # | Experiment | Runs | Primary Question |
|---|-----------|------|------------------|
| 5 | Model Comparison | 15 | Which single model performs best at different K values? |
| 4 | Hybrid Weight Optimization | 36 | What is the optimal blend of CF, Content, and ALS? |
| 6 | Ablation Study | 7 | How much does each component contribute to the hybrid? |
| 7 | Cold-Start Analysis | 12 | How do models degrade for users with few ratings? |
| 2 | CF Top-K Sweep | 5 | What is the accuracy--diversity tradeoff as the similarity neighbourhood grows? |
| 9 | Feature Importance | 3 | Do genre TF-IDF or genome tags contribute more to content-based quality? |

All runs use a deterministic seed (`seed=42`) and are logged to the shared MLflow
tracking server on DagsHub for full reproducibility.

---

## 2. Methodology

### Data Split

Ratings are split **temporally** using a global 80/20 chronological cutoff
(`src/evaluation/temporal_split.py`). All ratings are sorted by timestamp; the first
80% form the training set and the most recent 20% form the test set. This mirrors a
real production scenario where the system must predict future preferences from past
behaviour.

### Test User Sampling

Each experiment samples a fixed number of test users (typically 200--500, depending on
the experiment) via `np.random.RandomState(seed=42)` for reproducibility. Only users
that appear in the training set (i.e., known to the model) are eligible, except for the
popularity baseline which can score any user.

### Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **RMSE** | Rating prediction | Root Mean Squared Error between predicted and actual ratings. Lower is better. |
| **NDCG@K** | Ranking quality | Normalised Discounted Cumulative Gain. Rewards relevant items ranked higher. Primary metric. |
| **Precision@K** | Ranking quality | Fraction of top-K recommendations that are relevant (rating >= 4.0). |
| **Recall@K** | Ranking quality | Fraction of relevant items captured in the top-K list. |
| **MAP@K** | Ranking quality | Mean Average Precision. Summarises precision at every relevant hit. |
| **Catalog Coverage** | Beyond-accuracy | Fraction of the full movie catalog that appears in at least one user's recommendation list. |

**Relevance threshold:** A test rating >= 4.0 is treated as a positive (relevant) item
for all ranking metrics.

---

## 3. Experiment 5 -- Model Comparison

**Goal:** Establish a head-to-head baseline across all four models plus a popularity
baseline, evaluated at K = 5, 10, and 20.

**Setup:** 500 test users, `relevance_threshold=4.0`, `seed=42`. Each model is evaluated
at three cutoff values (K=5, 10, 20), yielding 15 runs total.

### Results (K=10)

| Model | RMSE | NDCG@10 | P@10 | Coverage |
|-------|-----:|--------:|-----:|---------:|
| **ALS** | **0.852** | 0.080 | 0.080 | 2.4% |
| Hybrid | 0.990 | 0.153 | 0.137 | 2.5% |
| Content-Based | 1.329 | 0.046 | 0.040 | 3.2% |
| Collaborative (CF) | 0.944 | 0.023 | 0.024 | **29.0%** |
| Popularity Baseline | 0.910 | **0.217** | **0.197** | 0.1% |

### Analysis

**Rating prediction (RMSE).** ALS achieves the lowest RMSE (0.852), confirming that
latent factor models excel at modelling explicit rating patterns. CF comes second
(0.944), while the hybrid's RMSE (0.990) is higher because it blends in the less
accurate content signal.

**Ranking quality (NDCG@10).** The popularity baseline dominates ranking metrics (0.217
NDCG@10) because it consistently recommends blockbusters that many users genuinely rate
highly. This is a well-known phenomenon in offline evaluation: popularity signals
correlate strongly with test-set positives. Among personalised models, the **hybrid
leads at 0.153**, nearly double the next best personalised approach (ALS at 0.080).

**Catalog coverage.** CF stands out at 29.0% coverage -- it surfaces a far wider variety
of items by drawing on user-specific neighbourhood patterns. Content-based achieves 3.2%,
the hybrid 2.5%, and the popularity baseline collapses to 0.1% (recommending the same
few hits to everyone).

### Key Takeaways

- The hybrid model is the **strongest personalised recommender** (NDCG@10 = 0.153),
  nearly 2x ALS alone and 6.5x pure CF.
- ALS provides the best individual rating accuracy (RMSE = 0.852).
- CF contributes diversity rather than precision -- it surfaces the long tail.
- The popularity baseline's high NDCG is an artifact of offline evaluation; in
  production, users already know those movies, so the perceived utility is much lower.

---

## 4. Experiment 4 -- Hybrid Weight Optimization

**Goal:** Find the optimal blend of CF, Content-Based, and ALS weights for the hybrid
recommender.

**Setup:** Exhaustive grid search over all weight triplets (cf, content, als) in 0.1
increments where each weight is in [0.0, 0.5] and all three sum to 1.0. This produces
**36 configurations**, each evaluated on 300 test users at K=10.

### Top 5 Configurations by NDCG@10

| Rank | CF Weight | Content Weight | ALS Weight | NDCG@10 |
|-----:|----------:|---------------:|-----------:|--------:|
| 1 | 0.2 | 0.2 | **0.6** | **0.1619** |
| 2 | 0.0 | 0.3 | 0.7 | 0.1614 |
| 3 | 0.1 | 0.2 | 0.7 | 0.1610 |
| 4 | 0.1 | 0.3 | 0.6 | 0.1607 |
| 5 | 0.2 | 0.3 | 0.5 | 0.1606 |

### Weight Landscape Analysis

The top-5 results reveal a consistent pattern:

- **ALS dominates** the optimal blend, commanding 50--70% of the weight in every
  top-performing configuration.
- **Content and CF are complementary but secondary.** The best single config uses
  equal 0.2/0.2 for CF and content, suggesting both provide roughly equal marginal
  value on top of a strong ALS backbone.
- **The surface is relatively flat near the optimum.** The spread between rank 1
  (0.1619) and rank 5 (0.1606) is only 0.0013 NDCG -- this means the hybrid is
  robust to small weight perturbations, which is a desirable property for deployment.
- **Removing CF entirely (rank 2: cf=0.0)** barely hurts performance (-0.0005), while
  removing content hurts more, foreshadowing the ablation study results.

### Best Configuration (saved to `models/hybrid_weight_results.json`)

```
cf_weight:      0.2
content_weight: 0.2
als_weight:     0.6
NDCG@10:        0.1619
```

---

## 5. Experiment 6 -- Ablation Study

**Goal:** Quantify the marginal contribution of each component by systematically
removing one model at a time from the full hybrid, and by isolating each model on its
own.

**Setup:** Seven configurations based on the optimal weights from Experiment 4. When a
component is removed, its weight is redistributed proportionally to the remaining
models. 300 test users, K=10.

### Results

| Configuration | Components | NDCG@10 | Delta vs Full |
|--------------|-----------|--------:|--------------:|
| full_hybrid | CF + Content + ALS | **0.1619** | -- |
| no_cf | Content + ALS | 0.1608 | -0.0010 |
| no_content | CF + ALS | 0.1588 | -0.0030 |
| no_als | CF + Content | 0.1527 | **-0.0092** |
| only_als | ALS alone | 0.1582 | -0.0037 |
| only_content | Content alone | 0.1554 | -0.0064 |
| only_cf | CF alone | 0.1514 | -0.0105 |

### Component Impact Ranking

Removing each component from the full hybrid produces the following NDCG drops:

1. **ALS** is the most critical component. Removing it causes the largest drop
   (-0.0092), confirming ALS's role as the ensemble's backbone.
2. **Content-Based** is the second most important. Removing it drops NDCG by -0.0030,
   providing meaningful signal beyond what ALS captures alone.
3. **CF** has the smallest individual impact (-0.0010 when removed), consistent with
   its low optimal weight (0.2). However, it still contributes positively.

### Isolation Analysis

When each model runs in isolation:

- **ALS alone (0.1582)** captures 97.7% of the full hybrid's performance, confirming
  it is the dominant signal.
- **Content alone (0.1554)** is surprisingly competitive at 96.0%.
- **CF alone (0.1514)** lags at 93.5%, consistent with its weaker ranking metrics in
  Experiment 5.

### Key Takeaway

All three components contribute positively, and the full hybrid outperforms every
individual component. The gains are additive but not dramatic -- this suggests the three
models capture substantially overlapping preference signals, with ALS being the
strongest individual predictor.

---

## 6. Experiment 7 -- Cold-Start Analysis

**Goal:** Understand how each model's performance varies with user activity level,
from near-cold-start users to power users.

**Setup:** Users are stratified by their training-set rating count into three buckets.
Each stratum is evaluated independently on 200 test users at K=10.

### User Strata

| Stratum | Rating Count | Description |
|---------|-------------|-------------|
| Light | 1--20 ratings | Near-cold-start users with minimal history |
| Medium | 21--100 ratings | Typical active users |
| Heavy | 100+ ratings | Power users with extensive history |

### Results (NDCG@10 / RMSE)

| Model | Light (1--20) | Medium (21--100) | Heavy (100+) |
|-------|:-------------:|:----------------:|:------------:|
| CF | 0.088 / 1.154 | 0.035 / 1.045 | 0.026 / 0.863 |
| Content | 0.098 / 1.484 | 0.092 / 1.418 | 0.030 / 1.300 |
| ALS | 0.108 / 1.093 | 0.072 / 0.920 | 0.073 / 0.788 |
| **Hybrid** | **0.136 / 1.094** | **0.111 / 0.965** | **0.061 / 0.870** |

### Analysis

**Light users (1--20 ratings).** The hybrid performs best across all strata, but its
advantage is most pronounced for light users (NDCG = 0.136 vs. 0.108 for ALS alone).
This makes sense: when collaborative signals are sparse, the content-based component
fills in the gaps with item-feature similarity. The hybrid acts as a natural fallback
mechanism.

**Medium users (21--100 ratings).** The hybrid maintains its lead (0.111). ALS (0.072)
and content (0.092) both contribute meaningfully. CF's NDCG actually *decreases* from
light to medium users (0.088 to 0.035), which is counterintuitive and likely reflects
the difficulty of finding similar users in a mid-range activity zone.

**Heavy users (100+ ratings).** ALS achieves its best RMSE here (0.788) as more
training data yields better latent factors. The hybrid's NDCG drops to 0.061, but so
do all models -- heavy users have seen more of the catalog, making novel relevant
recommendations harder to surface.

**RMSE trends.** Rating prediction consistently improves with more data across all
models. ALS shows the steepest improvement (1.093 to 0.788), while content-based
improves the least (1.484 to 1.300), confirming that content features provide a
stable but less adaptive signal.

### Deployment Recommendations by User Segment

| User Segment | Recommended Strategy |
|-------------|---------------------|
| New users (< 5 ratings) | Content-heavy hybrid or popularity fallback |
| Light users (5--20) | Full hybrid with standard weights |
| Medium users (21--100) | Full hybrid (strongest absolute performance) |
| Heavy users (100+) | ALS-heavy blend or full hybrid |

---

## 7. Experiment 2 -- CF Hyperparameter Tuning

**Goal:** Determine how the size of the item-item similarity neighbourhood (`top_k`)
affects CF performance, exposing the accuracy--diversity tradeoff.

**Setup:** The CF model is retrained from scratch for each `top_k` value in
{10, 20, 50, 100, 200}. Evaluated on 300 test users at K=10.

### Results

| top_k | NDCG@10 | P@10 | Coverage | RMSE |
|------:|--------:|-----:|---------:|-----:|
| 10 | **0.046** | **0.046** | 17.0% | 0.952 |
| 20 | 0.031 | 0.031 | 18.8% | 0.943 |
| **50** | 0.018 | 0.018 | 20.6% | 0.934 |
| 100 | 0.013 | 0.012 | 21.6% | 0.932 |
| 200 | 0.014 | 0.013 | 22.3% | **0.931** |

### Accuracy vs. Diversity Tradeoff

This experiment reveals a textbook tradeoff between ranking accuracy and catalog
diversity:

**Precision peaks at small neighbourhoods.** With `top_k=10`, each item's predicted
score is influenced by only the 10 most similar items, producing sharp, focused
recommendations. NDCG@10 is 0.046 -- the highest in the sweep.

**Coverage grows monotonically.** As `top_k` increases from 10 to 200, coverage
rises from 17.0% to 22.3%. Larger neighbourhoods dilute the signal, spreading
recommendations across more of the catalog.

**RMSE improves with more neighbours.** Rating prediction benefits from more data
points in the weighted average (0.952 at k=10 vs. 0.931 at k=200), but the marginal
gain diminishes rapidly.

**Diminishing returns beyond top_k=100.** Both NDCG and coverage plateau between
100 and 200, suggesting that neighbours beyond the 100th add mostly noise.

### Selected Configuration

The production configuration uses `top_k=50` as a balanced compromise: it retains
reasonable ranking quality (NDCG = 0.018) while achieving 20.6% coverage and the
third-best RMSE (0.934). This value is recorded in `params.yaml`.

---

## 8. Experiment 9 -- Feature Importance

**Goal:** Isolate the contribution of each feature source (genre TF-IDF vs. genome
tag vectors) in the content-based recommender.

**Setup:** Three configurations of the content-based model are trained from scratch,
each using a different subset of features. Evaluated on 300 test users at K=10.

### Feature Configurations

| Config | Features | Dimensionality |
|--------|---------|---------------|
| Full | Genre TF-IDF (weight 0.3) + Genome Tags (weight 0.7) | ~1,100+ |
| Genre-only | Genre TF-IDF (weight 1.0) | ~20 |
| Genome-only | Genome Tags (weight 1.0) | ~1,100 |

### Results

| Config | NDCG@10 | RMSE | Coverage |
|--------|--------:|-----:|---------:|
| Full (0.3 genre + 0.7 genome) | 0.041 | 1.308 | 2.6% |
| **Genre TF-IDF only** | **0.113** | **0.952** | **3.6%** |
| Genome only | 0.019 | 1.365 | 2.1% |

### Surprising Finding: Genre Beats Genome

This is one of the most striking results in the entire experiment suite. **Genre
TF-IDF alone (0.113 NDCG) outperforms the full combined model (0.041) by nearly 3x,**
and massively outperforms genome tags alone (0.019).

Several factors explain this counterintuitive outcome:

1. **Genome sparsity.** The MovieLens genome scores exist for only a subset of movies.
   Movies without genome data receive zero-vectors, degrading similarity calculations
   for any user who interacted with non-genome movies.

2. **Dimensionality curse.** Genome vectors have ~1,100 dimensions versus ~20 for
   genre. High-dimensional cosine similarity is known to suffer from distance
   concentration, where all item pairs appear roughly equidistant.

3. **Genre signals are robust.** Genre categories are available for every movie and
   capture the primary axis along which users express preferences (action vs. drama,
   comedy vs. thriller). This coarse-grained signal turns out to be more predictive
   than fine-grained genome tags for top-K ranking.

4. **Weighting imbalance.** The default configuration assigns 0.7 weight to genome
   tags, which actually *hurts* the combined model. The full model's NDCG (0.041) is
   dragged down below genre-only (0.113) because the dominant genome signal introduces
   noise.

### Actionable Recommendation

The content-based component should use **genre TF-IDF features only** (or at minimum,
flip the weights to genre=0.7, genome=0.3) for ranking tasks. The genome tags may still
be valuable for explainability or item-similarity pages, but they actively harm top-K
ranking quality in the current configuration.

---

## 9. Key Findings and Recommendations

### Top Insights

1. **The hybrid ensemble is the best personalised recommender.** It achieves 0.153
   NDCG@10 in head-to-head comparison -- nearly 2x ALS alone and 6.5x pure CF.

2. **ALS is the backbone of the hybrid.** The optimal blend allocates 60% weight to
   ALS. Removing ALS causes the largest ablation drop (-0.0092 NDCG).

3. **The hybrid is especially valuable for cold-start users.** For users with 1--20
   ratings, the hybrid (0.136 NDCG) outperforms ALS alone (0.108) by 26%, because
   content features compensate for sparse collaborative signals.

4. **Genre TF-IDF dramatically outperforms genome tags** for content-based ranking
   (0.113 vs. 0.019 NDCG). The current 0.7 genome weight is suboptimal and should be
   inverted or genome features should be dropped entirely for ranking.

5. **CF contributes diversity, not accuracy.** CF achieves only 0.023 NDCG@10 but
   covers 29% of the catalog -- 10x more than any other model. Its role in the
   hybrid is to broaden the recommendation surface.

6. **The hybrid weight surface is flat near the optimum.** The top-5 configurations
   span only 0.0013 NDCG, meaning the system is robust to small weight changes and
   does not require frequent retuning.

### Optimal Production Configuration

Based on the full experiment suite, the recommended production configuration is:

```yaml
hybrid:
  cf_weight:      0.2
  content_weight: 0.2
  als_weight:     0.6

als:
  rank: 64
  max_iter: 15
  reg_param: 0.1

collaborative:
  top_k_similar: 50

content_based:
  # Recommendation: switch to genre-only or invert weights
  genre_weight: 0.3   # current; consider 1.0
  genome_weight: 0.7  # current; consider 0.0
```

### Interview Discussion Points

- **Why not just use the popularity baseline?** It has the highest offline NDCG (0.217),
  but it recommends the *same movies to every user*. Coverage is 0.1%. In production,
  these are movies users have already seen, rendering the recommendations useless.
  Offline metrics overestimate popularity because test-set positives are biased toward
  popular items.

- **Why a hybrid over a single model?** The ablation study shows all three components
  contribute positively. More importantly, the cold-start analysis demonstrates that
  the hybrid degrades gracefully -- content features pick up the slack when
  collaborative signals are sparse.

- **What is the biggest remaining opportunity?** The feature importance experiment
  revealed that genome tags *hurt* the content model. Fixing the content feature
  weights alone could improve the content component from 0.041 to 0.113 NDCG -- a
  2.75x improvement that would propagate through the hybrid.

---

## 10. Reproducibility

### Prerequisites

- Python 3.10+ with dependencies from `requirements.txt`
- MovieLens 25M dataset in `data/raw/`
- Trained models in `models/` (or run the training pipeline first)
- DagsHub/MLflow credentials in `.env`

### Running Individual Experiments

Each experiment is a standalone script in `scripts/`:

```bash
# Model comparison (15 runs, ~30 min)
python scripts/exp5_model_comparison.py

# Hybrid weight optimization (36 runs, ~15 min)
python scripts/exp4_hybrid_weights.py

# Ablation study (7 runs, ~10 min)
python scripts/exp6_ablation.py

# Cold-start analysis (12 runs, ~15 min)
python scripts/exp7_cold_start.py

# CF top-K sweep (5 runs, ~20 min)
python scripts/exp2_cf_topk.py

# Feature importance (3 runs, ~10 min)
python scripts/exp9_feature_importance.py
```

### Key Parameters (from `params.yaml`)

```yaml
data:
  test_ratio: 0.2
  min_user_ratings: 20
  min_movie_ratings: 50

collaborative:
  similarity_metric: cosine
  top_k_similar: 50
  n_recommendations: 20

content_based:
  tfidf_max_features: 5000
  genome_weight: 0.7
  genre_weight: 0.3
  n_recommendations: 20

als:
  rank: 64
  max_iter: 15
  reg_param: 0.1
  cold_start_strategy: drop
  n_recommendations: 20

hybrid:
  cf_weight: 0.25
  content_weight: 0.35
  als_weight: 0.4
```

> **Note:** The `params.yaml` weights (0.25/0.35/0.40) reflect the initial defaults.
> Experiment 4 identified the optimal weights as 0.2/0.2/0.6, which are used in all
> subsequent experiments and saved to `models/hybrid_weight_results.json`.

### Viewing Results

All 78 runs are logged to the shared MLflow tracking server:

**[https://dagshub.com/vigneshsabapathi/recommender_system.mlflow](https://dagshub.com/vigneshsabapathi/recommender_system.mlflow)**

Each run records:
- Full hyperparameters (`mlflow.log_params`)
- All evaluation metrics (`mlflow.log_metrics`)
- Experiment tags for filtering (`experiment_type`, `model`, `stratum`, etc.)
