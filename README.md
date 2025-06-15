# ResNet-18 ONNX Model Deployment on Cerebrium

## 🚨 **IMPORTANT FOR EVALUATORS**

### **Required Information for test_server.py**

**Deployment URL:**
```
https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier
```

**API Key:**
```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWM3ODMyODZmIiwiaWF0IjoxNzQ5OTc3OTI2LCJleHAiOjIwNjU1NTM5MjZ9.MiBLtpw3gijysnhFsFQ7JJmzXMQO10iOYcYPAGuZ3wqJQqdsE10LMvn7Y_WJ4cu15SZOXc_JSTZqR7gt-erB2PQwFypm5-d3h7Pgbgbt4QQHqFxxeKBezGYSYwvMirSNovDHrV72ZLBNyfTxmn6PtmHXC6RuRQJQJLc2q0lxykkqtrVUf9joI38BmyLnTnTO54-RuIFWJzSht6pj1ATPPLi2gSsjHE55luxWv1R-2ySRVsq9EjQcP7rRMYV1Fhw-Yc_AtmYrRbDo8dK4JM4KuZlHV6DpDQfzs_2TITUHhpuDvDM0MItc2nhCY5SWWn_T583qYCZDHE0Whc8PpESoDA
```

**Quick Test Command:**
```bash
python test_server.py --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier --api-key "" --preset-tests
```

**Dashboard URL:**
```
https://dashboard.cerebrium.ai/projects/p-c783286f/apps/p-c783286f-resnet18-classifier
```

---

## Project Overview

This project implements a complete MLOps pipeline for deploying a ResNet-18 image classification model on the Cerebrium serverless platform. The model is converted from PyTorch to ONNX format and deployed using Docker containers with comprehensive testing and monitoring capabilities.


### **All Requirements Completed**

| Deliverable | Status | Description |
|-------------|--------|-------------|
| `convert_to_onnx.py` |  Complete | PyTorch to ONNX model conversion |
| `model.py` |  Complete | ONNX model loading and image preprocessing |
| `test.py` |  Complete | Local testing framework |
| `app.py` | Complete | FastAPI application for Cerebrium |
| `Dockerfile` |  Complete | Docker container configuration |
| `cerebrium.toml` | Complete | Cerebrium deployment configuration |
| `test_server.py` |  Complete | Deployed model testing script |
| `deploy.py` |  Complete | Automated deployment script |
| `requirements.txt` |  Complete | Python dependencies |
| `DEPLOYMENT.md` |  Complete | Deployment guide |

## 🚀 Quick Start

### 1. **Model Conversion**
```bash
# Convert PyTorch model to ONNX
python convert_to_onnx.py
```

### 2. **Local Testing**
```bash
# Test all components locally
python test.py --verbose --performance
```

### 3. **Deployment**
```bash
# Build and deploy to Cerebrium
python deploy.py --build --test --deploy
```

### 4. **Test Deployed Model**
```bash
# Test the deployed model (replace with your deployment URL and API key)
python test_server.py --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier --api-key "your-api-key" --preset-tests
```

## 📁 Project Structure

```
mtailor_mlops_assessment/
├── Model Conversion
│   ├── convert_to_onnx.py          # PyTorch → ONNX conversion
│   ├── pytorch_model.py            # Original PyTorch model
│   └── pytorch_model_weights.pth   # Pre-trained weights
│
├── Model Components  
│   ├── model.py                    # ONNX inference & preprocessing
│   └── model.onnx                  # Converted ONNX model (44.58 MB)
│
├──  Testing
│   ├── test.py                     # Local testing framework
│   └── test_server.py              # Deployed model testing
│
├──  Deployment
│   ├── app.py                      # FastAPI application
│   ├── Dockerfile                  # Container configuration
│   ├── cerebrium.toml              # Cerebrium config
│   ├── deploy.py                   # Deployment automation
│   ├── requirements.txt            # Dependencies
│   └── DEPLOYMENT.md               # Deployment guide
│
├── Test Data
│   ├── n01440764_tench.jpeg        # Test image (class 0)
│   └── n01667114_mud_turtle.JPEG   # Test image (class 35)
│
└── Documentation
    ├── README.md                   # Original requirements
    └── PROJECT_README.md           # This file
```

## 🔧 Component Details

### **1. Model Conversion (`convert_to_onnx.py`)**
- **Purpose**: Convert PyTorch ResNet-18 to ONNX format
- **Features**:
  - Automatic model loading and conversion
  - Validation against PyTorch outputs
  - Dynamic batch size support
  - Comprehensive error handling

**Usage:**
```bash
python convert_to_onnx.py [--input-model PATH] [--output-model PATH]
```

### **2. Model Components (`model.py`)**
- **ImagePreprocessor**: Handles image preprocessing pipeline
  - RGB conversion, resizing (224x224), normalization
  - Multiple input formats: file paths, PIL Images, NumPy arrays
- **ONNXModelInference**: ONNX model loading and inference
  - CPU/GPU execution providers
  - Top-K predictions with confidence scores
- **ModelPipeline**: End-to-end classification pipeline

**Usage:**
```python
from model import ModelPipeline
pipeline = ModelPipeline('model.onnx')
class_id = pipeline.classify_image('image.jpg')
```

### **3. Local Testing (`test.py`)**
- **Comprehensive Test Suite**: 28 test cases covering:
  - Image preprocessing (9 tests)
  - ONNX model inference (7 tests)  
  - End-to-end pipeline (4 tests)
  - Edge cases and error handling (5 tests)
  - Performance benchmarks (3 tests)

**Usage:**
```bash
python test.py --verbose --performance
```

### **4. FastAPI Application (`app.py`)**
- **REST API Endpoints**:
  - `POST /predict` - Upload image files
  - `POST /predict_url` - Classify from URLs
  - `POST /predict_base64` - Base64 encoded images
  - `GET /health` - Health monitoring
  - `GET /info` - Model information
- **Features**: Structured logging, error handling, CORS support

### **5. Deployment Infrastructure**
- **`Dockerfile`**: Production-ready container with Python 3.11
- **`cerebrium.toml`**: Cerebrium platform configuration
- **`deploy.py`**: Automated deployment with testing
- **`requirements.txt`**: Pinned dependency versions

### **6. Server Testing (`test_server.py`)**
- **Comprehensive Testing**: Tests deployed model functionality
- **Features**:
  - Single image classification
  - Preset tests with known results
  - Performance benchmarking
  - Platform monitoring
  - API endpoint validation

## Model Information

- **Architecture**: ResNet-18 (18-layer residual network)
- **Dataset**: ImageNet (1000 classes)
- **Input**: 224×224 RGB images
- **Output**: Class probabilities for 1000 ImageNet classes
- **Format**: ONNX (44.58 MB)
- **Performance**: < 3 seconds inference time

### **Test Images & Expected Results**
- `n01440764_tench.jpeg` → Class 0 (Tench fish)
- `n01667114_mud_turtle.JPEG` → Class 35 (Mud turtle)

## Deployment Guide

### **Prerequisites**
1. **Docker**: Install Docker Desktop or Docker Engine
2. **Cerebrium Account**: Sign up at https://www.cerebrium.ai/
3. **Cerebrium CLI**: `pip install cerebrium`

### **Step-by-Step Deployment**

#### **1. Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure Cerebrium CLI
cerebrium login
```

#### **2. Build and Test Locally**
```bash
# Build Docker image
python deploy.py --build

# Test Docker image locally
python deploy.py --test
```

#### **3. Deploy to Cerebrium**
```bash
# Deploy using CLI
cerebrium deploy

# Or use automated script
python deploy.py --deploy
```

#### **4. Get Deployment Information**
After successful deployment, you'll receive:
- **Dashboard URL**: https://dashboard.cerebrium.ai/projects/p-c783286f/apps/p-c783286f-resnet18-classifier
- **API Endpoint**: https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier
- **API Key**: Available in your Cerebrium dashboard

## 🧪 Testing the Deployed Model

### **Important: API Key and URL Required**

To test the deployed model, you need:
1. **Deployment URL**: Get from Cerebrium dashboard after deployment
2. **API Key**: Get from your Cerebrium account dashboard

### **test_server.py Usage Guide**

#### **1. Basic Single Image Classification**
```bash
# Test with a single image
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --api-key "your-api-key-here" \
  --image n01440764_tench.jpeg
```

**Expected Output:**
```
 Classification Result:
   Class ID: 0
   Confidence: 10.9109
   Inference Time: 0.093s
   Top 5 Predictions:
     1. Class 0: 10.9109
     2. Class 1: 7.4568
     3. Class 29: 7.2242
     4. Class 397: 7.0951
     5. Class 389: 6.3205
```

#### **2. Preset Tests (Recommended)**
```bash
# Run preset tests with known expected results
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --api-key "your-api-key-here" \
  --preset-tests
```

**Expected Output:**
```
✅ Test tench PASSED: Predicted 0 (expected 0)
✅ Test turtle PASSED: Predicted 35 (expected 35)
Preset tests completed: 2/2 passed
```

#### **3. Performance Testing**
```bash
# Run performance benchmarks
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --api-key "your-api-key-here" \
  --performance \
  --image n01440764_tench.jpeg
```

#### **4. Platform Monitoring**
```bash
# Monitor platform health and availability
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --api-key "your-api-key-here" \
  --monitor
```

#### **5. Comprehensive Test Suite**
```bash
# Run all tests and save results
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --api-key "your-api-key-here" \
  --comprehensive \
  --output test_results.json \
  --verbose
```

### **Alternative: Using Environment Variables**
```bash
# Set environment variable for API key
export CEREBRIUM_API_KEY="your-api-key-here"

# Then run tests without --api-key flag
python test_server.py \
  --url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier \
  --preset-tests
```

### **Command Line Options**

| Option | Description | Example |
|--------|-------------|---------|
| `--url` | **Required** - Deployment URL | `--url https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier` |
| `--api-key` | API key for authentication | `--api-key "your-key"` |
| `--image` | Path to image file | `--image n01440764_tench.jpeg` |
| `--image-url` | URL of image to classify | `--image-url https://example.com/image.jpg` |
| `--top-k` | Number of top predictions | `--top-k 3` |
| `--preset-tests` | Run preset tests | `--preset-tests` |
| `--performance` | Run performance tests | `--performance` |
| `--monitor` | Run platform monitoring | `--monitor` |
| `--comprehensive` | Run all tests | `--comprehensive` |
| `--output` | Save results to JSON file | `--output results.json` |
| `--verbose` | Enable verbose logging | `--verbose` |

### **API Endpoints Available**

The deployed model provides these REST API endpoints:

1. **POST /predict** - Upload image files
   ```bash
   curl -X POST \
     -H "Authorization: Bearer your-api-key" \
     -F "file=@image.jpg" \
     https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier/predict
   ```

2. **POST /predict_url** - Classify from URL
   ```bash
   curl -X POST \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/image.jpg", "top_k": 5}' \
     https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier/predict_url
   ```

3. **POST /predict_base64** - Base64 encoded images
   ```bash
   curl -X POST \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"image_data": "base64-encoded-image", "top_k": 5}' \
     https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier/predict_base64
   ```

4. **GET /health** - Health check
   ```bash
   curl -H "Authorization: Bearer your-api-key" \
     https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier/health
   ```

5. **GET /info** - Model information
   ```bash
   curl -H "Authorization: Bearer your-api-key" \
     https://api.cortex.cerebrium.ai/v4/p-c783286f/resnet18-classifier/info
   ```

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Authentication Errors (401 Unauthorized)**
- **Problem**: Missing or invalid API key
- **Solution**: 
  ```bash
  # Get API key from Cerebrium dashboard
  # Use it in command or set environment variable
  export CEREBRIUM_API_KEY="your-api-key"
  ```

#### **2. File Upload Errors (400 Bad Request)**
- **Problem**: Invalid file type or missing filename
- **Solution**: Ensure image files are valid JPEG/PNG with proper extensions

#### **3. Deployment URL Not Found (404)**
- **Problem**: Incorrect deployment URL
- **Solution**: Check Cerebrium dashboard for correct URL format

#### **4. Timeout Errors**
- **Problem**: Model cold start or network issues
- **Solution**: Retry request, model will warm up after first use

### **Validation Checklist**

Before running tests, ensure:
-  Model is deployed and showing as "healthy" in Cerebrium dashboard
-  API key is valid and has proper permissions
-  Test images (`n01440764_tench.jpeg`, `n01667114_mud_turtle.JPEG`) exist in current directory
-  All dependencies are installed (`pip install -r requirements.txt`)
-  Network connectivity to Cerebrium API endpoints


## 🚀 Next Steps & Improvements

### **Completed Features**
-  Complete MLOps pipeline
-  Production deployment
-  Comprehensive testing
-  Performance monitoring
-  Error handling and logging
-  API documentation

### **Potential Enhancements**
-  CI/CD pipeline integration
-  Model versioning and A/B testing
-  Advanced monitoring and alerting
-  Batch processing capabilities
-  Model performance analytics
-  Auto-scaling optimization

