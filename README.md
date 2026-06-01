# Skin Cancer Detection Backend

This backend provides FastAPI endpoints for skin lesion classification using a MobileNetV2-based PyTorch model.

## Features
- `POST /api/predict` for single image inference
- `POST /api/predict/batch` for multiple image inference
- `GET /api/classes` for class metadata
- `GET /api/health`
- CORS configured for `http://localhost:3003`
- Input validation for image format and size
- Logging and error handling
- Docker support

## Project Structure

```text
skin-cancer-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   └── ml/
├── models/                  # Model checkpoints (.pth)
├── scripts/                 # train.py, preprocess.py
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

## Local development

```bash
cd skin-cancer-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

## Docker

```bash
cd skin-cancer-backend
docker build -t skin-cancer-backend .
docker run -p 5000:5000 --env-file .env skin-cancer-backend
```

## Training

1. Download the HAM10000 dataset.
2. Prepare metadata CSV and image directories.
3. Run:

```bash
python scripts/train.py --data_dir /path/to/ham10000 --csv_path /path/to/HAM10000_metadata.csv --output models/mobilenetv2_skin.pt
```

If you do not have a trained checkpoint yet, the backend will load a fresh MobileNetV2 classifier and still run inference, but predictions will be random until training is complete.

## Important Disclaimer

This backend is for research and demonstration. It is NOT a medical diagnosis. Users must consult a dermatologist for professional evaluation.
# Skin_cancer_api
