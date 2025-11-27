# ML Image Classification Pipeline

An end-to-end Machine Learning pipeline for image classification with deployment capabilities, retraining functionality, and comprehensive monitoring.

## Project Overview

This project demonstrates a complete ML pipeline including:
- **Data Acquisition**: Load and organize image datasets
- **Data Preprocessing**: Image normalization, augmentation, and splitting
- **Model Creation**: CNN-based classifier using transfer learning (MobileNetV2)
- **Model Training**: Training with data augmentation and early stopping
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score, confusion matrix)
- **API Deployment**: Flask-based REST API for predictions
- **Web UI**: Interactive dashboard with visualizations and controls
- **Model Retraining**: Trigger-based retraining with new data
- **Load Testing**: Locust-based performance testing
- **Docker Deployment**: Containerized application for scalability

## Use Case

Image classification using non-tabular data (images). The model can classify images into multiple categories using deep learning techniques.

## Project Structure

```
ml-summative/
│
├── README.md                  # This file
│
├── notebook/
│   └── ml_summative.ipynb     # Jupyter notebook with detailed evaluation
│
├── src/
│   ├── preprocessing.py       # Data preprocessing module
│   ├── model.py              # Model creation and training
│   └── prediction.py         # Prediction module
│
├── data/
│   ├── train/                # Training images organized by class
│   ├── test/                 # Test images
│   ├── upload/               # Temporary upload directory
│   └── retrain/              # Data for retraining
│
├── models/                   # Saved model files
│   ├── image_classifier.tf   # Trained model (SavedModel format)
│   └── image_classifier_metadata.json
│
├── static/                   # Static files (CSS, JS)
│   ├── style.css
│   └── script.js
│
├── templates/                # HTML templates
│   └── index.html
│
├── app.py                    # Flask API application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
├── locustfile.py           # Load testing script
└── .dockerignore           # Docker ignore file
```

##  Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)
- Git (for cloning repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Carine-Ashimwe/Summative-assignment---MLOP/
   cd ml-summative
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Organize your data**
   
   Place your training images in the following structure:
   ```
   data/train/
     ├── class1/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── class2/
     │   ├── image1.jpg
     │   └── ...
     └── ...
   ```
   
   Each subdirectory in `data/train/` represents a class, and images should be placed in their respective class folders.

5. **Train the initial model** (if not already trained)
   
   Open the Jupyter notebook:
   ```bash
   jupyter notebook notebook/ml_summative.ipynb
   ```
   
   Run all cells to train and evaluate the model.

   Alternatively, use Python script:
   ```bash
   python train_model.py  # (create if needed)
   ```

## Running the Application

### Option 1: Run Flask App Directly

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Option 2: Run with Docker

1. **Build and run with Docker Compose** (recommended for production):
   ```bash
   docker-compose up --build
   ```

   This will start multiple API instances on ports 5000, 5001, and 5002.

2. **Build and run single container**:
   ```bash
   docker build -t ml-classification-api .
   docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-classification-api
   ```

### Option 3: Run with Gunicorn (Production)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

## Web UI Features

Access the web interface at `http://localhost:5000`

The UI includes:

1. **Model Uptime Monitoring**: Real-time display of model status and uptime
2. **Image Prediction**: Upload single or multiple images for classification
3. **Data Visualizations**: 
   - Class distribution charts
   - Model performance metrics
   - Prediction confidence distribution
4. **Upload Data**: Bulk upload images for retraining
5. **Trigger Retraining**: Retrain model with new data

## API Endpoints

### Health & Status
- `GET /api/health` - Health check endpoint
- `GET /api/model/uptime` - Get model uptime information
- `GET /api/stats` - Get model statistics

### Prediction
- `POST /api/predict` - Predict single image
  - Body: `multipart/form-data` with `image` field
  - Returns: Prediction result with class and confidence

- `POST /api/predict/batch` - Predict multiple images
  - Body: `multipart/form-data` with multiple `images` fields
  - Returns: List of prediction results

### Retraining
- `POST /api/upload/retrain` - Upload data for retraining
  - Body: `multipart/form-data` with `data` fields (images)
  
- `POST /api/retrain` - Trigger model retraining
  - Returns: Status of retraining process

- `GET /api/retrain/status` - Get retraining status
  - Returns: Current retraining progress and status

### Example API Usage

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@path/to/image.jpg"
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

##  Model Evaluation

Open the Jupyter notebook `notebook/ml_summative.ipynb` to see:
- Detailed preprocessing steps
- Model training process
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - Per-class performance
  - Training history plots

##  Load Testing with Locust

Test the API performance under load:

1. **Install Locust** (if not already installed):
   ```bash
   pip install locust
   ```

2. **Run Locust**:
   ```bash
   locust -f locustfile.py --host=http://localhost:5000
   ```

3. **Access Locust Web UI**:
   - Open browser to `http://localhost:8089`
   - Set number of users and spawn rate
   - Start the test and monitor:
     - Request rates (requests/second)
     - Response times (latency)
     - Error rates
     - Throughput

4. **Run with multiple containers**:
   - Start multiple API instances with docker-compose
   - Use a load balancer (nginx) or test each port separately
   - Compare performance metrics

### Load Testing Results

![Load testing charts](https://github.com/Carine-Ashimwe/Summative-assignment---MLOP/raw/main/data/locust_chart/total_requests_per_second.png)

Expected metrics (varies based on hardware):
- **Single Container**: ~50-100 req/s
- **Multiple Containers**: Linear scaling
- **Average Latency**: 100-300ms per prediction
- **P95 Latency**: 200-500ms

##  Model Retraining

### Trigger Retraining via UI

1. Upload training images using the "Upload Data for Retraining" section
2. Click "Trigger Retraining" button
3. Monitor progress in real-time
4. Model will be automatically reloaded after completion

### Trigger Retraining via API

```bash
# 1. Upload data
curl -X POST http://localhost:5000/api/upload/retrain \
  -F "data=@image1.jpg" \
  -F "data=@image2.jpg"

# 2. Trigger retraining
curl -X POST http://localhost:5000/api/retrain

# 3. Check status
curl http://localhost:5000/api/retrain/status
```

##  Data Visualizations

The UI provides three key visualizations:

1. **Class Distribution**: Shows the distribution of classes in the training dataset
   - **Interpretation**: Balanced distribution ensures better model performance across all classes

2. **Model Performance Metrics**: Training vs validation metrics over epochs
   - **Interpretation**: Convergence and small gap indicate good model generalization

3. **Prediction Confidence Distribution**: Distribution of prediction confidences
   - **Interpretation**: Higher values indicate more certain predictions, suggesting model confidence

##  Docker Deployment

### Docker Compose Setup

The `docker-compose.yml` file sets up multiple API instances for load balancing and scalability testing.

## Requirements

See `requirements.txt` for all Python dependencies. Key dependencies:
- TensorFlow 2.15.0
- Flask 3.0.0
- NumPy, Pandas
- OpenCV
- scikit-learn
- Matplotlib, Seaborn

## Video Demo

[[YouTube Link]](https://www.youtube.com/watch?v=zgohUDb78AE)

##  URLs

- **Deployed url on render**: https://summative-assignment-mlop-1-n8hv.onrender.com/
- **Local Development**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/health
- **Locust Dashboard**: http://localhost:8089

### Running Tests

```bash
# Add tests to test/ directory and run:
pytest
```

### Code Structure

- `src/preprocessing.py`: Handles all data preprocessing operations
- `src/model.py`: Model architecture and training logic
- `src/prediction.py`: Prediction and inference logic
- `app.py`: Flask API application with all endpoints


**Note**: Make sure to have sufficient training data (at least 100-200 images per class recommended) for good model performance.

