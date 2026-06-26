# Network Intrusion Detection System (NIDS)

A classical machine learning pipeline for detecting network intrusions, built on
the **NSL-KDD** benchmark dataset. Trains and compares **Random Forest**, **SVM**,
and **KNN** classifiers to distinguish normal traffic from attacks (DoS, Probe,
R2L, U2R).

Built as part of an internship project. Full design rationale, literature review,
and methodology are in [`docs/PDR_Network_Intrusion_Detection.pdf`](docs/PDR_Network_Intrusion_Detection.pdf).

## Features

- 📥 Automatic NSL-KDD download (or use your own local copy)
- 🧹 Full preprocessing pipeline: cleaning, categorical encoding, scaling, label mapping
- 🤖 Three classical ML models out of the box, easy to extend
- 📊 Evaluation with accuracy, precision, recall, F1, false-positive rate, and confusion matrices
- 🔌 Simple CLI for training and running predictions on new traffic
- ✅ Test suite covering preprocessing, training, and evaluation
- ⚙️ GitHub Actions CI that runs tests on every push

## Project Structure

```
network-intrusion-detection/
├── main.py                  # CLI entry point (train / predict)
├── src/
│   ├── constants.py         # Column defs, attack-category mapping, dataset URLs
│   ├── data_loader.py       # Download/load NSL-KDD, synthetic sample data
│   ├── preprocessing.py     # Cleaning, encoding, scaling
│   ├── train.py             # Model registry + training
│   ├── evaluate.py          # Metrics + plots
│   └── predict.py           # Inference on new traffic
├── tests/                   # pytest test suite
├── data/
│   ├── raw/                 # Downloaded NSL-KDD files land here
│   └── processed/           # (optional) cached processed data
├── models/                  # Saved trained models + preprocessing artifacts
├── reports/
│   └── figures/             # Confusion matrices, comparison charts
├── notebooks/                # Exploratory analysis (optional)
├── docs/
│   ├── PDR_Network_Intrusion_Detection.pdf
│   └── ARCHITECTURE.md
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

## Quick Start

```bash
git clone https://github.com/<your-username>/network-intrusion-detection.git
cd network-intrusion-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train a model

```bash
# Trains Random Forest by default; downloads NSL-KDD automatically into data/raw/
python main.py train

# Train and compare multiple models
python main.py train --models random_forest svm knn

# Multiclass (normal/dos/probe/r2l/u2r) instead of binary normal-vs-attack
python main.py train --models random_forest --no-binary

# No internet access? Smoke-test the pipeline with synthetic data
python main.py train --models random_forest knn --sample-data
```

This will:
1. Load and preprocess the NSL-KDD dataset
2. Train the selected model(s)
3. Print a classification report to the console
4. Save trained models to `models/`
5. Save confusion matrices and a comparison chart to `reports/figures/`
6. Save all metrics to `reports/metrics.json`

### Run predictions on new traffic

```bash
python main.py predict --input path/to/traffic.csv --output predictions.csv --model random_forest
```

The input CSV should contain the same feature columns as NSL-KDD (see
`src/constants.py:NSL_KDD_COLUMNS`, excluding `label` and `difficulty`).

### Run tests

```bash
pytest tests/ -v
```

## Dataset

This project uses **[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)**, an
improved version of the original KDD Cup 1999 dataset with duplicate records
removed and a more balanced difficulty distribution. `src/data_loader.py`
downloads it automatically; if you have no internet access in your environment,
download `KDDTrain+.txt` and `KDDTest+.txt` manually and place them in `data/raw/`.

| Dataset | Records | Features | Notes |
|---|---|---|---|
| NSL-KDD | ~150,000 | 41 | Used in this repo |
| CICIDS2017 | ~2.8M | 80+ | Possible future extension |
| UNSW-NB15 | ~2.5M | 49 | Possible future extension |

## Models

| Model | Notes |
|---|---|
| Random Forest | Default; robust baseline with `class_weight="balanced"` |
| SVM (RBF kernel) | Good for high-dimensional feature spaces; slower to train |
| KNN | Simple baseline for comparison |

## Evaluation Metrics

Because intrusion datasets are heavily imbalanced (far more normal traffic than
attacks), accuracy alone is misleading. This pipeline reports:

- **Precision** — of predicted attacks, how many were real attacks
- **Recall** — of real attacks, how many were caught
- **F1-score** — harmonic mean of precision and recall
- **False Positive Rate** — critical for real-world usability (binary mode)
- **Confusion Matrix** — full breakdown, saved as a PNG per model

## Roadmap

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for ideas — deep learning models (LSTM /
autoencoder), SMOTE oversampling, a live dashboard, and support for
CICIDS2017/UNSW-NB15 are natural next steps documented in the PDR.

## License

MIT — see [LICENSE](LICENSE).
