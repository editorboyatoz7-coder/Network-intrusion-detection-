# Contributing

Contributions, issues, and suggestions are welcome.

## Setup

```bash
git clone https://github.com/<your-username>/network-intrusion-detection.git
cd network-intrusion-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

- Keep functions small and documented with docstrings (NumPy style).
- Add a test in `tests/` for any new function in `src/`.
- Run `python main.py train --sample-data` before opening a PR to confirm the
  pipeline still runs end-to-end without requiring a network download.

## Suggested contribution ideas

- Add a deep learning model (LSTM / autoencoder) alongside the classical models.
- Add SMOTE-based oversampling as an optional preprocessing step.
- Add a Streamlit dashboard for live predictions.
- Extend `src/data_loader.py` to support CICIDS2017 or UNSW-NB15.
