.PHONY: help data features train api demo clean

help:
	@echo "Available commands:"
	@echo "  make data      - Download dataset and build labeled data"
	@echo "  make features  - Build rolling-window features"
	@echo "  make train     - Train baseline model and save artifacts"
	@echo "  make api       - Run FastAPI server"
	@echo "  make demo      - Full pipeline + API"
	@echo "  make clean     - Remove generated artifacts"

data:
	python src/data/download_cmapps.py
	python src/data/make_dataset.py

features:
	python src/features/build_features.py

train:
	python src/models/train.py

api:
	uvicorn app.main:app --reload

demo: data features train
	@echo "Model trained. Starting API..."
	uvicorn app.main:app --reload

clean:
	rm -rf data/processed artifacts/*.joblib artifacts/*.json
