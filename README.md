

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-success)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Task](https://img.shields.io/badge/Task-Predictive%20Maintenance-brightgreen)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

# predictive-maintenance
Binary classification project: predict whether an engine will fail within the next **N cycles** using sensor time-series data (NASA C-MAPSS). The model will be served via **FastAPI** and packaged with **Docker**.

## Overview
This repository will include:
- dataset download and preprocessing (NASA C-MAPSS)
- feature engineering using rolling time-window statistics
- model training and evaluation (ROC-AUC, F1, Recall)
- inference API (`/predict`) and containerized deployment

## API
### Endpoints
- `GET /health`
- `POST /predict`

Example:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"engine_id": 1, "features": {"s2_mean_30": 0.12, "s3_std_30": 0.03}}'
```


### Run (Docker)
```bash
docker build -t pm-api .
docker run -p 8000:8000 pm-api
```
