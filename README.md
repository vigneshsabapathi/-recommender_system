# Movie Recommender System

A production-grade movie recommendation engine built on the MovieLens 20M dataset, combining collaborative filtering, content-based filtering, and ALS matrix factorisation into a weighted hybrid model -- served through a FastAPI backend and a Netflix-inspired Next.js frontend.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-06B6D4?logo=tailwindcss&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-E25A1C?logo=apachespark&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.58-13ADC7?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

```
                          +--------------------+
                          |   Next.js Frontend |
                          |  (Netflix-style UI)|
                          +---------+----------+
                                    |
                                    | REST API
                                    v
                          +--------------------+
                          |   FastAPI Backend   |
                          |  /api/v1/recommend |
                          |  /api/v1/similar   |
                          |  /api/v1/movies    |
                          +---------+----------+
                                    |
                     +--------------+--------------+
                     |              |              |
              +------+---+  +------+---+  +-------+--+
              |  Item-Item|  | Content  |  |   ALS    |
              |    CF     |  |  Based   |  | (Spark)  |
              +------+----+  +-----+---+  +----+-----+
                     |             |            |
                     +------+------+------+-----+
                            |             |
                     +------+---+  +------+---+
                     | Weighted |  | SQLite   |
                     |  Hybrid  |  | (movies) |
                     +----------+  +----------+

  Data Pipeline (DVC):
  ┌────────┐   ┌────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────┐
  │ Ingest │-->│ Preprocess │-->│ Feature Engineer │-->│ Train (x3)     │-->│ Evaluate │
  └────────┘   └────────────┘   └──────────────────┘   └────────────────┘   └──────────┘
```

---

## Features

**Machine Learning**
- Item-item collaborative filtering with cosine similarity (top-K truncated)
- Content-based filtering with TF-IDF + MovieLens genome tags (1,128 dimensions)
- ALS matrix factorisation via PySpark for scalable latent-factor learning
- Weighted hybrid model blending all three approaches
- Temporal train/test split respecting chronological order
- Comprehensive evaluation: RMSE, Precision@K, Recall@K, NDCG@K, MAP@K, catalog coverage, intra-list diversity
- Cosine-Euclidean mathematical verification of similarity computations

**Backend**
- FastAPI REST API with auto-generated OpenAPI docs
- Endpoints for recommendations, similar movies, movie search, and health checks
- SQLite database seeded from processed MovieLens data
- TMDb API integration for movie posters and metadata
- Docker-ready with Render deployment config

**Frontend**
- Netflix-inspired dark UI built with Next.js 15 and Tailwind CSS
- Horizontal carousels for recommendation categories
- Movie detail modals with poster art
- Model switcher to compare recommendation approaches in real time
- Responsive design for desktop and mobile

**MLOps**
- DVC pipeline with 7 reproducible stages
- MLflow experiment tracking via DagsHub
- Parameterised hyperparameters in `params.yaml`
- Makefile for one-command workflows

---

## Screenshots

> **Home Page** -- Netflix-style carousels showing top picks, trending, and genre-based recommendations.

> **Movie Detail** -- Modal overlay with poster, synopsis, genres, and personalised "More Like This" suggestions.

> **Model Comparison** -- Toggle between collaborative, content-based, ALS, and hybrid outputs to see how each approach differs.

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Java 8+ (for PySpark ALS)

### 1. Install dependencies

```bash
git clone https://github.com/vigneshsabapathi/recommender_system.git
cd recommender_system
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Run the ML pipeline

```bash
# Download MovieLens 20M, preprocess, train all models, evaluate
make all
```

Or step by step:

```bash
make data       # Download + preprocess + feature engineering
make train      # Train collaborative, content-based, ALS
make evaluate   # Compute all metrics
make seed-db    # Seed SQLite for the API
```

### 3. Launch the application

```bash
# Terminal 1: Backend
make serve      # http://localhost:8000/docs

# Terminal 2: Frontend
make frontend   # http://localhost:3000
```

---

## ML Pipeline (DVC)

The pipeline is defined in `dvc.yaml` and can be reproduced with a single command:

```bash
dvc repro
```

### Stages

| Stage | Command | Inputs | Outputs |
|-------|---------|--------|---------|
| **ingest** | `python -m src.data.ingest` | `ingest.py` | `data/raw/*.csv` (6 files) |
| **preprocess** | `python -m src.data.preprocess` | Raw CSVs, params | Train/test splits, clean ratings, metadata |
| **feature_engineering** | `python -m src.data.feature_engineering` | Train ratings, metadata, genome | User-item matrix, content features, ID maps |
| **train_collaborative** | `python -m src.pipeline.train collaborative` | User-item matrix, ID maps | `models/collaborative/` |
| **train_content_based** | `python -m src.pipeline.train content_based` | Content features, user-item matrix | `models/content_based/` |
| **train_als** | `python -m src.pipeline.train als` | Train ratings | `models/als/` |
| **evaluate** | `python -m src.pipeline.evaluate` | Test ratings, all models | `models/evaluation_summary.json` |

### Pipeline DAG

```
  ingest --> preprocess --> feature_engineering --> train_collaborative ─┐
                                                ├-> train_content_based ├-> evaluate
                                                └-> train_als ──────────┘
```

Changing a hyperparameter in `params.yaml` automatically triggers only the affected downstream stages.

---

## Model Performance

Results are written to `models/evaluation_summary.json` after running the evaluation stage.

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Coverage |
|-------|------|-------------|-----------|---------|--------|----------|
| Collaborative (item-item CF) | -- | -- | -- | -- | -- | -- |
| Content-Based (TF-IDF + genome) | -- | -- | -- | -- | -- | -- |
| ALS (matrix factorisation) | -- | -- | -- | -- | -- | -- |
| Hybrid (weighted blend) | -- | -- | -- | -- | -- | -- |

> Values are populated after running `make evaluate`. See `notebooks/06_evaluation_comparison.ipynb` for visualisations.

---

## API Documentation

The FastAPI backend auto-generates interactive docs at `/docs` (Swagger) and `/redoc`.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check and loaded models |
| `GET` | `/api/v1/recommendations/{user_id}` | Get personalised recommendations |
| `GET` | `/api/v1/similar/{movie_id}` | Find movies similar to a given movie |
| `GET` | `/api/v1/movies` | Search and browse movies |
| `GET` | `/api/v1/movies/{movie_id}` | Get movie details |

### Query Parameters

- `model` -- Choose model: `collaborative`, `content_based`, `als`, `hybrid` (default)
- `n` -- Number of recommendations (default: 20)
- `page`, `page_size` -- Pagination for movie listings

### Example

```bash
# Get 10 recommendations for user 42 using the hybrid model
curl "http://localhost:8000/api/v1/recommendations/42?model=hybrid&n=10"
```

---

## Project Structure

```
recommender_system/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # App entry point and lifespan
│   │   ├── config.py          # Pydantic settings
│   │   ├── dependencies.py    # Service singletons
│   │   ├── db/
│   │   │   ├── database.py    # SQLite setup
│   │   │   └── seed.py        # Seed from processed data
│   │   ├── routers/
│   │   │   ├── health.py      # GET /health
│   │   │   ├── movies.py      # GET /movies
│   │   │   ├── recommendations.py
│   │   │   └── similar.py     # GET /similar/{id}
│   │   ├── schemas/           # Pydantic request/response models
│   │   └── services/          # Business logic layer
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # Next.js 15 frontend
│   ├── src/
│   │   ├── app/               # App router pages
│   │   ├── components/        # React components
│   │   ├── hooks/             # Custom hooks
│   │   ├── lib/               # API client, utilities
│   │   └── stores/            # State management
│   ├── package.json
│   └── vercel.json            # Vercel deployment config
├── src/                        # ML source code
│   ├── data/
│   │   ├── ingest.py          # Download MovieLens 20M
│   │   ├── preprocess.py      # Clean, filter, temporal split
│   │   └── feature_engineering.py  # User-item matrix, content features
│   ├── models/
│   │   ├── base.py            # Abstract base recommender
│   │   ├── collaborative.py   # Item-item CF
│   │   ├── content_based.py   # TF-IDF + genome content
│   │   ├── als_model.py       # PySpark ALS
│   │   └── hybrid.py          # Weighted hybrid
│   ├── pipeline/
│   │   ├── train.py           # Training entry point
│   │   └── evaluate.py        # Evaluation entry point
│   ├── evaluation/
│   │   ├── metrics.py         # RMSE, Precision, Recall, NDCG, MAP, etc.
│   │   ├── temporal_split.py  # Temporal split logic
│   │   └── verification.py    # Cosine-Euclidean verification
│   └── utils/
│       ├── config.py          # Pydantic settings + params loader
│       └── logger.py          # Logging setup
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_collaborative_filtering.ipynb
│   ├── 04_content_based.ipynb
│   ├── 05_matrix_factorization.ipynb
│   └── 06_evaluation_comparison.ipynb
├── data/
│   ├── raw/                   # Downloaded CSVs (DVC tracked)
│   └── processed/             # Cleaned data, feature matrices
├── models/                    # Trained model artefacts
│   ├── collaborative/
│   ├── content_based/
│   ├── als/
│   ├── hybrid/
│   └── evaluation_summary.json
├── tests/                     # Test suite
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Hyperparameters
├── Makefile                   # Convenience commands
├── render.yaml                # Render deployment config
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Deployment

### Backend (Render)

The `render.yaml` defines a free-tier web service using the Docker runtime:

```bash
# Deploy via Render dashboard or CLI
# The service builds from backend/Dockerfile
# Set TMDB_API_KEY in the Render environment variables
```

### Frontend (Vercel)

The Next.js frontend deploys to Vercel with zero configuration:

```bash
cd frontend
vercel --prod
```

Set the `NEXT_PUBLIC_API_URL` environment variable to point to the Render backend URL.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | EDA on MovieLens 20M: distributions, sparsity, temporal patterns, genres, genome tags |
| `02_preprocessing.ipynb` | Step-by-step preprocessing walkthrough with before/after visualisations |
| `03_collaborative_filtering.ipynb` | Item-item CF: similarity computation, "similar to Toy Story", sample recommendations |
| `04_content_based.ipynb` | TF-IDF + genome features, genre-only vs genome-enhanced comparison |
| `05_matrix_factorization.ipynb` | ALS training, latent factor inspection, t-SNE visualisation, convergence analysis |
| `06_evaluation_comparison.ipynb` | Cross-model metrics comparison, radar charts, cosine-Euclidean verification |

---

## Dataset Attribution

This project uses the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) by GroupLens Research:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19.
> https://doi.org/10.1145/2827872

Movie poster images and additional metadata are sourced from [The Movie Database (TMDb)](https://www.themoviedb.org/). This product uses the TMDb API but is not endorsed or certified by TMDb.

---

## License

This project is released under the [MIT License](LICENSE).
