# AidRenal — End-to-End CNN Kidney CT Classifier

**Author:** Okereke Kelechi Collins  
**Repository:** https://github.com/MicroPyscho/renal_tumour_project  
**Experiment Tracking:** https://dagshub.com/MicroPyscho/renal_tumour_project.mlflow  
**Live Dashboard:** Deployed via Docker on AWS ECR / App Runner  

> *Inspired by and built using Krish Naik's end-to-end ML project structure as a reference and guiding framework.*

---

## What This Project Does

AidRenal is a production-grade, end-to-end deep learning pipeline that classifies renal pathologies from CT scan images into four categories:

- **Cyst** — fluid-filled sacs on or in the kidney
- **Normal** — healthy renal tissue
- **Stone** — nephrolithiasis (kidney stones)
- **Tumour** — malignant or suspicious renal masses

Four CNN architectures were trained, tracked, and compared under identical conditions:

| Model | Val Accuracy | Val Loss | Macro F1 | Best Epoch |
|---|---|---|---|---|
| DenseNet121 | 100% | 0.0063 | 1.0000 | ~42 |
| ResNet50V2 | 100% | 0.0007 | 1.0000 | ~43 |
| EfficientNetV2B3 | 99.40% | 0.0198 | 0.9937 | ~47 |
| VGG16 (frozen) | 94.16% | 0.2403 | 0.9284 | ~60+ |

All models, training scripts, experiment logs, and inference pipelines are open-source and freely available. No GPU, cloud account, or proprietary licence is required to run inference.

---

## Project Architecture

```
renal_tumour_project/
├── app.py                          # Flask web server (port 8080)
├── main.py                         # Pipeline entry point
├── dvc.yaml                        # DVC pipeline stages
├── params.yaml                     # MODEL_NAME + hyperparameters
├── config/config.yaml              # Paths configuration
├── requirements.txt
├── Dockerfile
├── .github/workflows/              # CI/CD → AWS ECR
├── checkpoints/                    # Saved model weights (.keras)
│   ├── best_DenseNet121.keras
│   ├── best_ResNet50v2.keras
│   └── best_EfficientNetV2B3.keras
├── artifacts/                      # DVC-tracked outputs
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   ├── training/
│   └── evaluation/
├── src/cnnClassifier/
│   ├── components/                 # data ingestion, model prep, training, evaluation
│   ├── pipeline/                   # prediction.py, vis_prediction.py
│   ├── config/                     # configuration manager
│   └── utils/
├── templates/index.html            # AidRenal dashboard (CT Classifier, Grad-CAM, etc.)
└── research/                       # Jupyter notebooks for each stage
```

---

## Pipeline Stages (DVC)

```
data_ingestion → prepare_base_model → training → evaluation
```

Each stage is tracked by DVC. Running `dvc repro` re-executes any stage whose inputs have changed.

---

## Workflows (Development Order)

When adding a new feature or model:

1. Update `config/config.yaml`
2. Update `secrets.yaml` (optional — not committed)
3. Update `params.yaml`
4. Update the entity (`src/cnnClassifier/entity/`)
5. Update the configuration manager (`src/cnnClassifier/config/`)
6. Update the component (`src/cnnClassifier/components/`)
7. Update the pipeline (`src/cnnClassifier/pipeline/`)
8. Update `main.py`
9. Update `dvc.yaml`
10. Update `app.py` if endpoint changes

---

## How to Run This Project

### Clone the Repository

```bash
git clone https://github.com/MicroPyscho/renal_tumour_project
cd renal_tumour_project
```

### STEP 1 — Create and Activate Conda Environment

```bash
conda create -n ml python=3.11 -y
conda activate ml
```

> Use Python 3.11. Python 3.13+ is not yet fully supported by TensorFlow and open-clip-torch.

### STEP 2 — Install Requirements

```bash
pip install -r requirements.txt
```

### STEP 3 — Run the Flask App

```bash
python app.py
```

Visit `http://localhost:8080` in your browser.

---

## MLflow Experiment Tracking

### Run MLflow UI locally

```bash
mlflow ui
```

Visit `http://localhost:5000` to browse experiment runs.

### Connect to DagHub (Remote Tracking)

```bash
pip install dagshub
```

```python
import dagshub
import mlflow

dagshub.init(repo_owner='MicroPyscho', repo_name='renal_tumour_project', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('model', 'DenseNet121')
    mlflow.log_metric('val_accuracy', 1.0)
```

All experiment runs, confusion matrices, and model artefacts are logged to:  
**https://dagshub.com/MicroPyscho/renal_tumour_project.mlflow**

### MLflow Run IDs

| Model | MLflow Run ID |
|---|---|
| EfficientNetV2B3 | c359d65d |
| ResNet50V2 | f7abac8d |
| DenseNet121 | 99732add |
| VGG16 | 9366b9ca26e844e8b48c3f0c05240c30 |

---

## DVC Pipeline

```bash
dvc init
dvc repro       # Run the full pipeline
dvc dag         # Visualise the pipeline graph
```

> **Do not run `dvc repro` to retrain a single model** — it will overwrite all stage outputs. Instead, change `MODEL_NAME` in `params.yaml` and run `python main.py` directly.

### Switch Model

In `params.yaml`:
```yaml
MODEL_NAME: DenseNet121   # options: DenseNet121 | ResNet50V2 | EfficientNetV2B3 | VGG16
```

---

## BiomedCLIP Gate

Before any CT scan reaches a CNN classifier, it passes through a **BiomedCLIP gate** — a biomedical vision-language model that validates the image is a legitimate CT scan. Non-CT images (photos, X-rays, documents) are rejected with an HTTP 400 error before inference.

### Setup BiomedCLIP

```bash
pip install open-clip-torch
```

BiomedCLIP downloads automatically on first run and caches to:
- **Mac/Linux:** `~/.cache/huggingface/`
- **Windows:** `C:\Users\YourName\.cache\huggingface\`

The model is approximately 1–2 GB. If transferring between machines, copy the cache folder rather than re-downloading.

---

## Docker

### Build the Image

```bash
docker build -t aidrenal:latest .
```

> **Important:** Use `python:3.11-slim-bookworm` as your base image in the Dockerfile. `slim-buster` is EOL for Python 3.11+ and `python:3.14.x` does not yet have a slim-buster variant.

```dockerfile
FROM python:3.11-slim-bookworm
```

### Run the Container

```bash
docker run -p 8080:8080 aidrenal:latest
```

### Push to Docker Hub

```bash
docker login
docker tag aidrenal:latest yourdockerhub/aidrenal:latest
docker push yourdockerhub/aidrenal:latest
```

---

## AWS Deployment (ECR + App Runner)

### ECR Repository

```
141927126255.dkr.ecr.eu-north-1.amazonaws.com/kidney_ct_scan
```

### Step 1 — Configure AWS CLI

```bash
aws configure
```

Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `eu-north-1`
- Output format: `json`

### Step 2 — Authenticate Docker to ECR

```bash
aws ecr get-login-password --region eu-north-1 | \
  docker login --username AWS --password-stdin \
  141927126255.dkr.ecr.eu-north-1.amazonaws.com
```

### Step 3 — Build, Tag, and Push

```bash
docker build -t kidney_ct_scan .

docker tag kidney_ct_scan:latest \
  141927126255.dkr.ecr.eu-north-1.amazonaws.com/kidney_ct_scan:latest

docker push \
  141927126255.dkr.ecr.eu-north-1.amazonaws.com/kidney_ct_scan:latest
```

### Step 4 — Deploy via App Runner

In the AWS Console, go to **App Runner → Create Service**:
- Source: Amazon ECR
- Image URI: `141927126255.dkr.ecr.eu-north-1.amazonaws.com/kidney_ct_scan:latest`
- Port: `8080`

App Runner handles HTTPS, auto-scaling, and zero-downtime deploys.

### CI/CD via GitHub Actions

The `.github/workflows/` directory contains the CI/CD pipeline. On every push to `main`, GitHub Actions:

1. Builds the Docker image
2. Pushes to ECR
3. Triggers a new App Runner deployment

Required GitHub Secrets:
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION=eu-north-1
ECR_REPOSITORY_URI=141927126255.dkr.ecr.eu-north-1.amazonaws.com/kidney_ct_scan
```

---

## Hugging Face Spaces (Alternative Free Deployment)

The project can also be deployed to Hugging Face Spaces using the existing Dockerfile:

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Docker** as the SDK
3. Connect your GitHub repository (`MicroPyscho/renal_tumour_project`)
4. Set port to `8080` in the Space settings

Hugging Face Spaces provides a permanent public HTTPS URL at no cost for public spaces and is well-suited for ML model demos.

---

## Quick Inference (No App Required)

```python
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

pipeline = PredictionPipeline(filename="path/to/ct_scan.jpg")
result = pipeline.predict()
print(result)  # [{'image': 'ct_scan.jpg', 'prediction': 'Normal'}]
```

---

## Dataset

**CT-KIDNEY-DATASET (Normal-Cyst-Tumor-Stone)**  
Source: Kaggle — https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone  
Curated from the Department of Urology and Nephrology, Dhaka, Bangladesh.

| Class | Total | Train (80%) | Val (20%) |
|---|---|---|---|
| Cyst | 3,709 | 2,967 | 742 |
| Normal | 5,077 | 4,062 | 1,015 |
| Stone | 1,377 | 1,102 | 275 |
| Tumour | 2,283 | 1,826 | 457 |
| **Total** | **12,446** | **9,955** | **2,491** |

---

## Training Configuration

All four models were trained under identical conditions:

| Hyperparameter | Value |
|---|---|
| Image Size | 224 × 224 × 3 |
| Batch Size | 24 |
| Base Learning Rate | 0.0001 |
| Max Epochs | 100 |
| Early Stopping Patience | 10 (min_delta=0.001) |
| LR Reduce Patience | 5 (factor=0.5, min_lr=1e-7) |
| Augmentation | Rotation, flip, shift, shear, zoom |
| Class Weighting | sklearn balanced |
| Loss Function | SparseCategoricalCrossentropy |
| ImageNet Weights | Yes |
| Train/Val Split | 80/20 (seed=42) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Experiment Tracking | MLflow + DagHub |
| Pipeline Orchestration | DVC |
| CT Gate | BiomedCLIP (open-clip-torch) |
| Web Framework | Flask |
| Frontend Dashboard | HTML / CSS / Vanilla JS |
| Containerisation | Docker |
| Cloud (Production) | AWS ECR + App Runner |
| Cloud (Demo) | Hugging Face Spaces |
| Version Control | Git + GitHub |

---

## Acknowledgements

This project was built using [Krish Naik](https://github.com/krishnaik06)'s end-to-end ML project repository structure as a guide and inspiration for the modular MLOps pipeline architecture.

---

## Next Steps

The next phase of AidRenal development will focus on **out-of-distribution generalisation**: training the models to correctly infer on arbitrary CT scan images they have never seen before — including full abdominal scans, images with different windowing levels, varied slice thicknesses, and scanner-manufacturer variation — and produce accurate pathology predictions. This will involve preprocessing normalisation, kidney region localisation, and external validation on datasets beyond the CT-KIDNEY-DATASET benchmark.

---

## Citation

If you use AidRenal in your research, please cite:

```
Okereke, K. C. (2025). AidRenal: A Comparative Deep Learning Framework for 
Automated Classification of Renal Pathologies in CT Imaging. 
GitHub: https://github.com/MicroPyscho/renal_tumour_project
```

---

## License

Open-source. Free for use by clinicians, researchers, and developers.  
**https://github.com/MicroPyscho/renal_tumour_project**