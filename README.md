# 🥭 Mango Analysis System: AI-Powered Disease & Pesticide Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-FF6F00.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive full-stack web application for automated mango quality analysis using deep learning and machine learning. This system provides dual-task analysis capabilities with intelligent hybrid model selection for optimal accuracy.

**🎯 Quick Stats:**
- ✅ **81.5% accuracy** on disease detection (5 classes)
- ✅ **97.1% accuracy** on pesticide detection (binary)
- ✅ **Hybrid AI** combining CNN + SVM
- ✅ **Production-ready** React + Flask stack
- ✅ **Real-time predictions** with confidence scores

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Model Training](#-model-training)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Screenshots](#-screenshots)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This intelligent system supports **two distinct analysis tasks**:

### 1. 🦠 Disease Detection (Multi-class Classification)
Classifies mangoes into 5 disease categories:
- **Alternaria** - Fungal disease causing dark spots
- **Anthracnose** - Common fungal infection
- **Black Mould Rot** - Post-harvest decay
- **Healthy** - No disease detected
- **Stem end Rot** - Stem infection

### 2. 🧪 Pesticide Detection (Binary Classification)
Determines food safety:
- **Organic** - No pesticide residue detected
- **Pesticide** - Pesticide treatment identified

### How It Works

1. **User uploads** mango image via web interface
2. **Backend processes** image through dual AI models:
   - **CNN** extracts deep visual features
   - **SVM** classifies using CNN features
3. **Hybrid selection** automatically chooses best model
4. **Results displayed** with confidence scores and per-class probabilities

---

## ✨ Features

### Core Functionality
✅ **Dual-Task Analysis** - Switch between disease and pesticide detection  
✅ **Hybrid Model Selection** - Auto-selects best performing model (CNN or SVM)  
✅ **Real-time Predictions** - Fast inference (< 2 seconds)  
✅ **Confidence Scores** - Percentage confidence for each prediction  
✅ **Per-Class Probabilities** - Detailed breakdown of all class likelihoods  
✅ **Image Validation** - Rejects non-mango images with low confidence  

### User Experience
🎨 **Beautiful UI** - Modern glassmorphism design with Tailwind CSS  
📊 **Interactive Charts** - Model comparison visualizations with Chart.js  
📱 **Responsive Design** - Works on desktop, tablet, and mobile  
🔄 **Live Status** - Real-time backend health monitoring  
🖼️ **Drag & Drop** - Easy image upload with preview  
🌈 **Smooth Animations** - Polished transitions with Framer Motion  

### Technical Features
🚀 **RESTful API** - Clean endpoints with proper error handling  
💾 **Model Caching** - Intelligent memory management  
📈 **Data Augmentation** - Training-time image transformations  
🎯 **Multi-task Architecture** - Independent model artifacts per task  
🔒 **Input Validation** - File type, size, and content checks  

---

## 🛠 Technology Stack

### Backend Stack (Python)

| Technology | Version | Purpose |
|------------|---------|---------|
| **Flask** | 3.0.0 | Lightweight web framework for REST API |
| **TensorFlow** | 2.18.1 | Deep learning framework for CNN training |
| **Keras** | Built-in | High-level neural network API |
| **Scikit-learn** | 1.3.2 | SVM classifier, preprocessing, metrics |
| **NumPy** | 1.26.2 | Numerical computing and array operations |
| **Pillow (PIL)** | 10.1.0 | Image loading, preprocessing, resizing |
| **Flask-CORS** | 4.0.0 | Cross-Origin Resource Sharing support |
| **Joblib** | 1.3.2 | Model serialization and persistence |
| **Matplotlib** | 3.8.2 | Training curve and metric visualization |
| **Seaborn** | 0.13.0 | Statistical data visualization |
| **Gunicorn** | 21.2.0 | Production WSGI HTTP server |

### Frontend Stack (JavaScript/React)

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.2.0 | Component-based UI framework |
| **React DOM** | 18.2.0 | React rendering for web |
| **React Router** | 6.26.2 | Client-side routing and navigation |
| **Tailwind CSS** | 3.4.14 | Utility-first CSS framework |
| **Chart.js** | 4.4.4 | Canvas-based chart library |
| **react-chartjs-2** | 5.2.0 | React wrapper for Chart.js |
| **Framer Motion** | 11.3.24 | Animation library for React |
| **PostCSS** | 8.4.47 | CSS transformation tool |
| **Autoprefixer** | 10.4.20 | Vendor prefix automation |

### Development Tools
- **Git** - Version control
- **npm** - Frontend package management
- **pip** - Python package management
- **VS Code** - Recommended IDE with Python and ESLint extensions

---

## 🏗 Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          React Frontend (Port 3000)                    │ │
│  │  • Tailwind CSS styling                                │ │
│  │  • React Router for SPA navigation                     │ │
│  │  • Chart.js for data visualization                     │ │
│  │  • Framer Motion for animations                        │ │
│  │  • Fetch API for HTTP requests                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP POST /api/detect
                            │ (multipart/form-data)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVER LAYER                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Flask Backend (Port 5000)                    │ │
│  │  • CORS enabled for cross-origin requests             │ │
│  │  • Blueprint-based route organization                  │ │
│  │  • Image validation & preprocessing                    │ │
│  │  • Model caching & lazy loading                        │ │
│  │  • Error handling & validation                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Load Models & Predict
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE LAYER                         │
│  ┌────────────────────────┐      ┌─────────────────────────┐│
│  │  CNN Feature Extractor │──────│  SVM Classifier         ││
│  │  (TensorFlow/Keras)    │      │  (Scikit-learn)         ││
│  │                        │      │                         ││
│  │  • 4 Conv2D blocks     │      │  • RBF kernel           ││
│  │  • BatchNormalization  │      │  • StandardScaler       ││
│  │  • MaxPooling2D        │      │  • Probability output   ││
│  │  • Dropout layers      │      │  • Trained on CNN       ││
│  │  • Dense(128) bottleneck      │    features (128-dim)   ││
│  │  • Softmax output      │      │                         ││
│  └────────────────────────┘      └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Read/Write
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ARTIFACTS STORAGE                          │
│  ┌───────────────────────┐     ┌───────────────────────┐   │
│  │  Disease Task         │     │  Pesticide Task       │   │
│  │  • mango_model.h5     │     │  • mango_model.h5     │   │
│  │  • svm.pkl            │     │  • svm.pkl            │   │
│  │  • class_indices.json │     │  • class_indices.json │   │
│  │  • history.json       │     │  • history.json       │   │
│  │  • metrics/           │     │  • metrics/           │   │
│  │    - model_comparison │     │    - model_comparison │   │
│  └───────────────────────┘     └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

1. **User Upload**: Image selected via drag-drop or file picker
2. **Client Validation**: Check file type (JPG/PNG/WEBP) and size (< 5MB)
3. **API Request**: POST to `/api/detect` with image + task parameter
4. **Server Validation**: Validate image format and content
5. **Preprocessing**: 
   - Convert to RGB (3 channels)
   - Resize to 224×224 pixels
   - Normalize pixel values to [0, 1]
   - Add batch dimension (1, 224, 224, 3)
6. **Model Loading**: Lazy load from cache or disk based on task
7. **CNN Inference**: 
   - Forward pass through convolutional layers
   - Extract 128-dimensional feature vector
   - Generate class probabilities
8. **SVM Inference**:
   - Use CNN features as input
   - StandardScaler normalization
   - RBF kernel classification
   - Generate probability estimates
9. **Model Selection**: Choose model with highest validation accuracy
10. **Response Formation**: JSON with prediction, confidence, probabilities
11. **Visualization**: Frontend renders results with charts and animations

---

## 📁 Project Structure

```
mango-pesticide-detector/
│
├── 📄 README.md                      # This comprehensive documentation
├── 📄 project_summary.txt            # Technical project summary
├── 📄 .gitignore                     # Git ignore rules
│
├── 📂 client/                        # React Frontend Application
│   ├── 📂 public/
│   │   └── index.html                # HTML template
│   │
│   ├── 📂 src/
│   │   ├── App.jsx                   # Main app with routing
│   │   ├── index.js                  # React entry point
│   │   │
│   │   ├── 📂 components/            # Reusable components
│   │   │   ├── ClassBadge.jsx        # Disease/pesticide badge
│   │   │   ├── HealthBadge.jsx       # API status indicator
│   │   │   ├── ModelComparisonChart.jsx  # Comparison bar chart
│   │   │   ├── Navbar.jsx            # Navigation bar
│   │   │   ├── PerClassBars.jsx      # Confidence visualization
│   │   │   └── Upload.jsx            # Image upload UI
│   │   │
│   │   ├── 📂 config/
│   │   │   └── api.js                # API endpoints config
│   │   │
│   │   ├── 📂 hooks/
│   │   │   └── useBackendHealth.js   # Health check hook
│   │   │
│   │   ├── 📂 pages/                 # Route pages
│   │   │   ├── About.jsx             # About page
│   │   │   ├── Compare.jsx           # Model comparison
│   │   │   ├── Home.jsx              # Landing page
│   │   │   └── Insights.jsx          # Metrics & matrices
│   │   │
│   │   └── 📂 styles/
│   │       ├── App.css               # Component styles
│   │       └── index.css             # Global Tailwind styles
│   │
│   ├── package.json                  # npm dependencies
│   ├── tailwind.config.js            # Tailwind configuration
│   └── postcss.config.js             # PostCSS setup
│
├── 📂 server/                        # Flask Backend Application
│   ├── app.py                        # Main Flask app entry
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── 📂 routes/
│   │   ├── __init__.py
│   │   └── detect.py                 # API endpoints
│   │
│   └── 📂 model/                     # ML training & artifacts
│       ├── __init__.py
│       ├── model_trainer.py          # Standard CNN training
│       ├── model_trainer_advanced.py # Transfer learning
│       ├── compare_models.py         # SVM training & comparison
│       ├── evaluate.py               # Evaluation utilities
│       ├── generate_dummy_dataset.py # Synthetic data
│       └── update_report.py          # Report generation
│       │
│       └── 📂 artifacts/             # Model storage
│           ├── 📂 disease/
│           │   ├── mango_model.h5    # CNN (81.5% acc)
│           │   ├── best_model.json   # Model selector
│           │   ├── class_indices.json
│           │   ├── history.json
│           │   ├── training_curves.png
│           │   ├── 📂 models/
│           │   │   └── svm.pkl       # SVM classifier
│           │   └── 📂 metrics/
│           │       └── model_comparison.json
│           │
│           └── 📂 pesticide/
│               ├── mango_model.h5    # CNN (97.1% acc)
│               ├── best_model.json
│               ├── class_indices.json
│               ├── history.json
│               ├── training_curves.png
│               ├── 📂 models/
│               │   └── svm.pkl
│               └── 📂 metrics/
│                   └── model_comparison.json
│
└── 📂 datasets/                      # Training data (not in repo)
    ├── 📂 disease/
    │   └── MangoFruitDDS/
    │       └── SenMangoFruitDDS_original/
    │           ├── Alternaria/       # ~1000 images
    │           ├── Anthracnose/      # ~1000 images
    │           ├── Black Mould Rot/  # ~1000 images
    │           ├── Healthy/          # ~1000 images
    │           └── Stem end Rot/     # ~1000 images
    │
    └── 📂 pesticide/
        ├── organic/                  # Organic samples
        └── pesticide/                # Treated samples
```

---

## 🚀 Installation

### Prerequisites

Before starting, ensure you have:

- ✅ **Python 3.11+** ([Download](https://www.python.org/downloads/))
- ✅ **Node.js 16+** and **npm** ([Download](https://nodejs.org/))
- ✅ **Git** ([Download](https://git-scm.com/))
- ✅ **8GB+ RAM** (recommended for model training)
- ✅ **GPU with CUDA** (optional, speeds up training 4-5x)

### Step 1: Clone Repository

```bash
git clone https://github.com/rakeshjayanna/final-year-project-.git
cd mango-pesticide-detector
```

### Step 2: Backend Setup

```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**What gets installed:**
- Flask 3.0.0 - Web framework
- TensorFlow 2.18.1 - Deep learning (700MB+)
- NumPy 1.26.2 - Array operations
- Pillow 10.1.0 - Image processing
- Scikit-learn 1.3.2 - ML algorithms
- Flask-CORS 4.0.0 - CORS support
- Matplotlib 3.8.2 - Visualization
- Seaborn 0.13.0 - Statistical plots
- Joblib 1.3.2 - Model persistence
- Gunicorn 21.2.0 - Production server

### Step 3: Frontend Setup

```bash
# Navigate to client directory (from project root)
cd client

# Install Node.js dependencies
npm install
```

**What gets installed:**
- React & React DOM 18.2.0 - UI framework
- React Router DOM 6.26.2 - Routing
- Tailwind CSS 3.4.14 - Styling
- Chart.js 4.4.4 - Charting
- react-chartjs-2 5.2.0 - React charts
- Framer Motion 11.3.24 - Animations
- PostCSS & Autoprefixer - CSS processing

**Installation time:** ~5-10 minutes depending on internet speed

---

## 🎓 Model Training

### Dataset Preparation

#### Disease Detection Dataset
- **Location**: `datasets/disease/MangoFruitDDS/SenMangoFruitDDS_original/`
- **Structure**: 5 subdirectories (one per class)
- **Total images**: 5000 (1000 per class)
- **Format**: JPG/PNG
- **Recommended size**: 224×224 or larger

#### Pesticide Detection Dataset
- **Location**: `datasets/pesticide/`
- **Structure**: 2 subdirectories (`organic/` and `pesticide/`)
- **Total images**: 1035+ recommended
- **Format**: JPG/PNG
- **Balance**: Roughly equal samples per class

### Training Commands

**Option 1: Standard CNN Training (Faster, ~15 minutes)**

```bash
# Disease detection (from project root)
python server/model/model_trainer.py --task disease --epochs 15 --batch-size 32

# Pesticide detection
python server/model/model_trainer.py --task pesticide --epochs 15 --batch-size 32
```

**Option 2: Transfer Learning (Higher Accuracy, ~30 minutes)**

```bash
# Disease with MobileNetV2
python server/model/model_trainer_advanced.py --task disease --epochs 30

# Pesticide with MobileNetV2
python server/model/model_trainer_advanced.py --task pesticide --epochs 30
```

**Training Parameters:**
- `--task`: Either `disease` or `pesticide` (required)
- `--epochs`: Number of training iterations (15-30 recommended)
- `--batch-size`: Images per batch (32 for 8GB RAM, 16 for 4GB)
- `--img-size`: Input dimensions (default: 224 224)
- `--learning-rate`: Initial learning rate (default: 1e-3)
- `--data-dir`: Custom dataset path (optional, auto-detected)

### Model Comparison & SVM Training

After training CNN, generate metrics:

```bash
# Disease task
python server/model/compare_models.py --task disease

# Pesticide task
python server/model/compare_models.py --task pesticide
```

**What this does:**
1. Loads trained CNN model
2. Extracts 128-dimensional features from Dense layer
3. Trains SVM classifier on these features
4. Evaluates both models on validation set (20% of data)
5. Compares accuracies and selects best model
6. Saves comparison metrics to JSON
7. Saves trained SVM to `.pkl` file

### Training Output

After successful training:

```
server/model/artifacts/<task>/
├── mango_model.h5              # Trained CNN (~40-50 MB)
├── best_model.json             # {"best_model": "svm"}
├── class_indices.json          # Label mappings
├── history.json                # Training metrics per epoch
├── training_curves.png         # Loss/accuracy plots
├── models/
│   └── svm.pkl                 # Trained SVM (~1-5 MB)
└── metrics/
    └── model_comparison.json   # Detailed comparison
```

**Expected training times (CPU):**
- Disease: ~12-15 minutes (5000 images, 15 epochs)
- Pesticide: ~8-10 minutes (1035 images, 15 epochs)
- With GPU: 3-4x faster

---

## 🎮 Usage

### Development Mode

**Terminal 1 - Backend:**
```bash
cd server
# Activate venv if not already
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

python app.py
# ✓ Server runs on http://localhost:5000
```

**Terminal 2 - Frontend:**
```bash
cd client
npm start
# ✓ Opens browser at http://localhost:3000
```

### Production Mode

**Backend (Gunicorn):**
```bash
cd server
gunicorn --workers 4 --bind 0.0.0.0:5000 --timeout 120 app:app
```

**Frontend (Build & Serve):**
```bash
cd client
npm run build
# Serve build/ directory with nginx, Apache, or other web server
```

### Using the Application

1. **Open** http://localhost:3000 in your browser
2. **Select Analysis Type**:
   - Click "🦠 Disease Detection" for disease classification
   - Click "🧪 Pesticide Detection" for pesticide analysis
3. **Upload Image**:
   - Click "Upload Mango Image" button
   - OR drag & drop image onto preview area
   - Supported: JPG, PNG, WEBP (max 5MB)
4. **View Results**:
   - Predicted class badge
   - Confidence percentage (0-100%)
   - Per-class probability distribution
   - Model used (SVM or CNN)
5. **Explore More**:
   - Click "Compare" to see model comparison charts
   - Click "Insights" for confusion matrices and metrics
   - Click "About" for project information

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health?task=disease
```

**Query Parameters:**
- `task` (optional): `disease` or `pesticide` (default: `disease`)

**Response (200 OK):**
```json
{
  "status": "ok",
  "task": "disease",
  "model_present": true,
  "best_model": "svm"
}
```

#### 2. Detect/Predict
```http
POST /api/detect
Content-Type: multipart/form-data
```

**Form Data:**
- `image`: Image file (required)
- `task`: `disease` or `pesticide` (required)

**Success Response (200 OK):**
```json
{
  "label": "Healthy",
  "confidence": 91.5,
  "model_used": "svm",
  "task": "disease",
  "models": {
    "cnn": {
      "label": "Healthy",
      "confidence": 89.2,
      "probs": {
        "Alternaria": 2.1,
        "Anthracnose": 3.4,
        "Black Mould Rot": 1.8,
        "Healthy": 89.2,
        "Stem end Rot": 3.5
      }
    },
    "svm": {
      "label": "Healthy",
      "confidence": 91.5,
      "probs": {
        "Alternaria": 1.5,
        "Anthracnose": 2.8,
        "Black Mould Rot": 1.2,
        "Healthy": 91.5,
        "Stem end Rot": 3.0
      }
    }
  },
  "selection": {
    "model": "svm",
    "reason": "highest validation accuracy",
    "detail": {
      "cnn_acc": 0.81,
      "svm_acc": 0.815
    }
  }
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Low confidence (45.2%) - image may not be a mango",
  "is_mango": false
}
```

#### 3. Compare Image
```http
POST /api/compare-image
Content-Type: multipart/form-data
```

**Form Data:** Same as `/detect`

**Response:** Includes full comparison details without "final" selection

#### 4. Model Comparison Metrics
```http
GET /api/models/comparison?task=disease
```

**Response (200 OK):**
```json
{
  "models": {
    "cnn": {
      "accuracy": 0.81,
      "report": {
        "0": {
          "precision": 0.647,
          "recall": 0.564,
          "f1-score": 0.603,
          "support": 179
        },
        ...
      },
      "confusion_matrix": [[101, 2, 66, 10, 0], ...]
    },
    "svm": {
      "accuracy": 0.815,
      "report": {...},
      "confusion_matrix": [...]
    },
    "class_names": ["Alternaria", "Anthracnose", ...]
  },
  "best": {
    "name": "svm",
    "accuracy": 0.815
  }
}
```

#### 5. Reload Models
```http
POST /api/reload
Content-Type: application/json
```

**Body (optional):**
```json
{
  "task": "disease"
}
```

**Response (200 OK):**
```json
{
  "status": "reloaded",
  "task": "disease",
  "model_present": true
}
```

---

## 📊 Performance Metrics

### Disease Detection Results

**Model Comparison:**
| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|--------|----------|----------------|
| CNN | 81.0% | 81.5% | 81.0% | 81.0% | ~150ms |
| **SVM** | **81.5%** | **82.1%** | **81.5%** | **81.7%** | ~50ms |

**Per-Class Performance (SVM - Best Model):**
| Disease Class | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Alternaria | 60.3% | 65.4% | 62.7% | 179 |
| Anthracnose | 95.9% | 83.8% | 89.4% | 197 |
| Black Mould Rot | 66.2% | 66.2% | 66.2% | 201 |
| Healthy | 89.4% | 97.1% | 93.1% | 208 |
| Stem end Rot | 95.7% | 92.1% | 93.8% | 215 |

**Key Insights:**
- ✅ **Healthy** mangoes: 97.1% recall (very few missed)
- ✅ **Stem end Rot**: 95.7% precision (reliable diagnosis)
- ⚠️ **Alternaria** vs **Black Mould Rot**: Similar visual features cause confusion
- ✅ Overall: 815 correct predictions out of 1000 test images

### Pesticide Detection Results

**Model Comparison:**
| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|--------|----------|----------------|
| CNN | 45.4% | 45.4% | 45.4% | 45.4% | ~150ms |
| **SVM** | **97.1%** | **97.3%** | **97.1%** | **97.1%** | ~50ms |

**Per-Class Performance (SVM):**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Organic | 100.0% | 94.7% | 97.3% | 113 |
| Pesticide | 94.0% | 100.0% | 96.9% | 94 |

**Key Insights:**
- ✅ **SVM dominates**: 97.1% vs 45.4% CNN accuracy
- ✅ **Zero false negatives** for pesticide detection (critical for food safety)
- ✅ **High precision** on both classes
- ℹ️ CNN struggles with binary classification on this dataset
- ✅ SVM leverages CNN features effectively

### Training Performance

**Disease Model (15 epochs):**
- Training accuracy: 83.3%
- Validation accuracy: 81.5%
- Training time: ~12 min (CPU) / ~3 min (GPU)
- Best epoch: 11 (early stopping)
- Parameters: 423,877 trainable

**Pesticide Model (15 epochs):**
- Training accuracy: 97.7%
- Validation accuracy: 98.1%
- Training time: ~8 min (CPU) / ~2 min (GPU)
- Best epoch: 2 (rapid convergence)
- Parameters: 423,361 trainable

---

## 📸 Screenshots

### Home Page with Upload
![Home Page](https://via.placeholder.com/1000x600/F59E0B/FFFFFF?text=Home+Page+-+Task+Selection+%26+Upload)

**Features shown:**
- Task selection cards (Disease / Pesticide)
- Drag & drop upload zone
- Image preview
- Live backend status

### Prediction Results
![Results](https://via.placeholder.com/1000x600/16A34A/FFFFFF?text=Prediction+Results+-+Confidence+%26+Probabilities)

**Features shown:**
- Predicted class badge
- Confidence percentage
- Per-class probability bars
- Model used indicator

### Model Comparison
![Comparison](https://via.placeholder.com/1000x600/3B82F6/FFFFFF?text=Model+Comparison+-+CNN+vs+SVM+Accuracy)

**Features shown:**
- Bar chart with accuracies
- Best model indicator
- Task selector

### Insights & Metrics
![Insights](https://via.placeholder.com/1000x600/8B5CF6/FFFFFF?text=Insights+-+Confusion+Matrix+%26+Metrics)

**Features shown:**
- Confusion matrix heatmap
- Per-class metrics table
- Model selector dropdown

---

## 🚀 Deployment

### Docker Deployment

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  backend:
    build: ./server
    ports:
      - "5000:5000"
    volumes:
      - ./server/model/artifacts:/app/model/artifacts
    environment:
      - FLASK_ENV=production
  
  frontend:
    build: ./client
    ports:
      - "80:80"
    depends_on:
      - backend
```

**Run:** `docker-compose up -d`

### Cloud Platforms

**Heroku:**
```bash
# Backend
cd server
heroku create mango-api
heroku git:remote -a mango-api
git push heroku main

# Frontend
cd client
npm run build
# Deploy to Netlify/Vercel/Heroku
```

**AWS EC2:**
1. Launch t2.medium instance (Ubuntu 22.04)
2. Install Python 3.11, Node.js 18
3. Clone repo and follow installation
4. Configure nginx as reverse proxy
5. Setup SSL with Let's Encrypt

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/mango-detector
gcloud run deploy --image gcr.io/PROJECT_ID/mango-detector --platform managed
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Model not found" error**
```
✓ Solution: Train models first
python server/model/model_trainer.py --task disease --epochs 15
python server/model/compare_models.py --task disease
```

**2. CORS errors in browser**
```
✓ Solution: Ensure Flask-CORS installed
pip install flask-cors

✓ Check app.py has:
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

**3. Out of memory during training**
```
✓ Solution: Reduce batch size
python server/model/model_trainer.py --batch-size 16 --epochs 15
```

**4. React proxy not working**
```
✓ Solution: Verify client/package.json has:
"proxy": "http://localhost:5000"

✓ Restart frontend: npm start
```

**5. TensorFlow warnings**
```
✓ Solution: Suppress info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

**6. Slow predictions**
```
✓ Solutions:
- Use GPU: Install tensorflow-gpu
- Reduce image size
- Enable model caching (already default)
```

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Batch image processing** - Upload multiple images at once
- [ ] **Export reports** - Download predictions as PDF/CSV
- [ ] **User authentication** - Login system with prediction history
- [ ] **Real-time camera** - Live detection via webcam
- [ ] **Mobile app** - React Native iOS/Android app
- [ ] **Grad-CAM visualization** - Highlight important image regions
- [ ] **Model versioning** - A/B testing and rollback
- [ ] **Edge deployment** - TensorFlow Lite for offline use
- [ ] **Multi-language UI** - i18n support (Hindi, Spanish, etc.)
- [ ] **RESTful pagination** - For large result sets

### Research Directions
- **Vision Transformers** - Explore ViT/Swin architectures
- **Self-supervised learning** - Reduce labeled data needs
- **Active learning** - Smart sample selection for labeling
- **Ensemble methods** - Combine multiple model predictions
- **Explainable AI** - LIME/SHAP for interpretability

---

## 👥 Contributing

We welcome contributions! Please follow:

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Guidelines
- Follow PEP 8 (Python) and ESLint (JavaScript)
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **Dataset**: MangoFruitDDS contributors
- **Frameworks**: TensorFlow, React, Flask, Scikit-learn teams
- **Community**: Stack Overflow, GitHub
- **Advisors**: Academic mentors and reviewers
- **Beta Testers**: For valuable feedback

---

## 📞 Contact

**Rakesh Jayanna**
- 💼 GitHub: [@rakeshjayanna](https://github.com/rakeshjayanna) 
- 🔗 LinkedIn: [Rakesh]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/rakesh-jayanna-215a3728b/))
- 🌐 Project: [https://github.com/rakeshjayanna/final-year-project-](https://github.com/rakeshjayanna/final-year-project-)

---

<div align="center">
  <p><strong>Made with ❤️ for Mango Farmers and Food Safety</strong></p>
  <p>⭐ <strong>Star this repo if you find it helpful!</strong></p>
  <p>🔀 Fork • 🐛 Report Bug • ✨ Request Feature</p>
</div>
