# README — Keras Inference API (FastAPI)

Simple FastAPI service that loads a saved Keras model, a scaler and a label encoder, and serves inference endpoints.

---

## What this is

A lightweight HTTP API to perform predictions using your trained Keras CNN+LSTM model. It:

* Loads artifacts at startup: model, scaler, and label encoder.
* Provides endpoints for single-sample and batch predictions.
* Includes validation to ensure each sample has exactly 14 features (shape used during training).

Files (single-folder layout)

* `main.py` — app entrypoint & startup hook
* `model_loader.py` — load and expose model/scaler/encoder globals
* `schemas.py` — Pydantic request/response schemas and validators
* `utils.py` — preprocessing and prediction helper functions
* `routes.py` — API routes (`/`, `/predict`, `/predict/batch`)
* `requirements.txt` — Python dependencies
* `.gitignore` — recommended ignores
* `model/` — suggested folder for saved artifacts (not included in repo)

  * `my_model_balanced_keras.keras`
  * `scaler.pkl` (or `.joblib`)
  * `label_encoder.pkl` (or `.joblib`)

---

## Requirements

Create and activate a virtualenv, then install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` should include (examples):

```
fastapi>=0.95
uvicorn[standard]>=0.22
tensorflow>=2.12
scikit-learn>=1.2
numpy>=1.24
pandas>=2.0
joblib>=1.2
requests
```

Notes:

* If you trained artifacts with a specific scikit-learn version (e.g. 1.6.1), consider matching that version to avoid `InconsistentVersionWarning` when unpickling.
* For GPU execution, install an appropriate GPU build of TensorFlow and set up CUDA/cuDNN.

---

## Configuration / Environment variables

Default artifact paths (relative to your project directory):

* `model/my_model_balanced_keras.keras`
* `model/scaler.pkl`
* `model/label_encoder.pkl`

You can override by exporting environment variables before running uvicorn:

```bash
export MODEL_PATH="/full/path/to/my_model_balanced_keras.keras"
export SCALER_PATH="/full/path/to/scaler.pkl"
export ENCODER_PATH="/full/path/to/label_encoder.pkl"

uvicorn main:app --reload
```

---

## Run the server

From the project folder (where `main.py` resides):

```bash
uvicorn main:app --reload
```

Server will be available at `http://127.0.0.1:8000`.

Interactive docs:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

---

## API Endpoints

### Health

`GET /`
Response:

```json
{ "status": "ok", "model_loaded": true }
```

### Single prediction

`POST /predict`
Request (flat list of 14 numbers):

```json
{
  "features": [0.5, 1.2, -0.7, 3.3, 0.0, 2.1, 4.5, 1.0, -2.3, 0.9, 1.4, -0.6, 2.7, 0.8]
}
```

Response:

```json
{
  "predicted_label": "Fail",
  "predicted_class_index": 2,
  "probabilities": [0.03, 0.04, 0.88, 0.05]
}
```

### Batch prediction

`POST /predict/batch`
Request (list of rows; each row must have 14 numbers):

```json
{
  "features": [
    [0.5, 1.2, ..., 0.8],
    [1.1, 0.4, ..., 0.7]
  ]
}
```

Response:

```json
{
  "predictions": [
    { "predicted_label": "...", "predicted_class_index": 2, "probabilities": [...] },
    ...
  ]
}
```

---

## Test examples

Curl (single):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[0.5,1.2,-0.7,3.3,0.0,2.1,4.5,1.0,-2.3,0.9,1.4,-0.6,2.7,0.8]}'
```

Curl (batch):

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"features":[[0.5,1.2,...,0.8],[1.1,0.4,...,0.7]]}'
```

Python quick test:

```python
import requests, json
single = {"features":[0.5,1.2,-0.7,3.3,0.0,2.1,4.5,1.0,-2.3,0.9,1.4,-0.6,2.7,0.8]}
r = requests.post("http://127.0.0.1:8000/predict", json=single)
print(r.status_code, r.json())
```

---
