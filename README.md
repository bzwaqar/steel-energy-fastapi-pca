# Steel Energy Week 3 — FastAPI Dashboard

## Setup

**Recommended: use the exact versions from Colab, not guessed pins.**

`week3_pca.ipynb` has a cell (right after `joblib.dump(...)`) that writes
`requirements_colab.txt` with the *exact* `scikit-learn`, `xgboost`, `lightgbm`, `joblib`,
`pandas`, `numpy`, and `scipy` versions used to train and save `model.joblib`. Download
that file from Colab along with `model.joblib` and `feature_engineering.py`, place it in
this folder, and install from it directly:

```bash
pip install -r requirements_colab.txt
pip install fastapi "uvicorn[standard]" jinja2 python-multipart
```

This avoids version-mismatch errors at `joblib.load()` time (`AttributeError`,
`InconsistentVersionWarning`, or `XGBoostError: input stream corrupted`) far more
reliably than manually guessing/pinning versions in `requirements.txt`.

If you don't have `requirements_colab.txt` yet, the fallback `requirements.txt` below has
best-guess pins — but check them against Colab's actual versions first:

```python
import sklearn, xgboost, lightgbm
print(sklearn.__version__, xgboost.__version__, lightgbm.__version__)
```

```bash
pip install -r requirements.txt
```

1. Copy `model.joblib` (saved by `week3_pca.ipynb`) into this same folder, next to `main.py`.
2. Generate the 3 dashboard PNGs (see the comment block at the top of the `/dashboard`
   route in `main.py`) and place them in `static/`:
   - `static/energy_by_hour.png`
   - `static/energy_by_load_type.png`
   - `static/correlation_heatmap.png`

## Run

```bash
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page with nav bar |
| `/dashboard` | GET | Static EDA charts |
| `/predict` | GET | Prediction form |
| `/predict` | POST | Runs form input through `model.joblib`, returns `Usage_kWh` |

## Notes

- The form collects **raw** values only (no manual one-hot encoding or scaling on the
  frontend/backend) — the loaded pipeline does all of that internally, exactly as it was
  fit in `week3_pca.ipynb`.
- `Load_Type`, `WeekStatus`, and `Day_of_week` dropdown option lists in `main.py`
  (`LOAD_TYPE_OPTIONS`, etc.) must match the exact string values present in the original
  training data. Adjust them if your dataset uses different labels.
- If `model.joblib` isn't found at startup, the app still runs — `/predict` will show a
  clear on-page error instead of crashing, so you can still browse `/` and `/dashboard`.
