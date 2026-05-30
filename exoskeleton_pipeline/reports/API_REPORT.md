# API Validation Report

**Service:** `app.py` (FastAPI + uvicorn)  
**Model:** `best_regressor_model.pkl` (RandomForest, 77 features, 11 channels)  
**Date:** 2026-05-26

---

## Channel alignment

| Component | Channels |
|-----------|----------|
| Training (`config.EMG_CHANNELS`) | 11 Camargo muscles |
| API `EMGPipeline` | Same 11 channels |
| Request shape | `(N_samples, 11)`, N ≥ 200 @ 1000 Hz |

`example_config_data.csv` (4 scalar examples) is **not** used by the API.

---

## Endpoints

### GET `/health`

| Field | Expected |
|-------|----------|
| `status` | `OK` when model loaded |
| `loaded_model` | `best_regressor_model` |

**Result:** `OK` — All systems operational.

### POST `/predict`

**Payload:** `{"emg_window": [[ch0..ch10], ...]}` — minimum 200 rows.

**Response fields:** `predicted_knee_angle`, `latency_ms`, `features_extracted`

---

## Test results (`test_client.py`)

| Test | Result |
|------|--------|
| Health check | Pass |
| Predict × 5 | 5/5 success |
| Avg round-trip latency | ~46 ms (first call ~153 ms warm-up) |
| Server inference (steady) | ~12 ms |

**Note:** Synthetic random EMG produces similar feature vectors → similar angle (~16°). Use real Camargo windows for meaningful angle spread.

---

## DSP consistency

API `EMGPipeline` uses `overlap=0.5` (matches training). For a single 200 ms raw window, decimation yields exactly one 20-sample envelope window — overlap does not change feature count.

---

## How to run

```bash
cd exoskeleton_pipeline
uvicorn app:app --host 127.0.0.1 --port 8000
python test_client.py
```

---

## Status: **PASS**
