# Cerebrium Deployment Guide

## Prerequisites

1. **Docker**: Install Docker Desktop or Docker Engine
2. **Cerebrium CLI**: Install with `pip install cerebrium`
3. **Cerebrium Account**: Sign up at https://www.cerebrium.ai/

## Setup

1. **Configure Cerebrium CLI**:
   ```bash
   cerebrium login
   ```

2. **Verify all files are present**:
   - model.onnx (44.58 MB)
   - model.py
   - app.py
   - requirements.txt
   - Dockerfile
   - cerebrium.toml

## Deployment Steps

### Option 1: Using the deployment script
```bash
python deploy.py --build --test --deploy
```

### Option 2: Manual deployment
```bash
# Build Docker image
docker build -t resnet18-classifier:latest .

# Test locally
docker run -p 8000:8000 resnet18-classifier:latest

# Deploy to Cerebrium
cerebrium deploy
```

## Testing the Deployment

### Health Check
```bash
curl https://your-deployment-url/health
```

### Image Classification
```bash
# Upload file
curl -X POST "https://your-deployment-url/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     -F "top_k=5"

# URL-based prediction
curl -X POST "https://your-deployment-url/predict_url" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/image.jpg", "top_k": 5}'
```

## API Endpoints

- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Upload image file
- `POST /predict_url` - Classify image from URL
- `POST /predict_base64` - Classify base64 encoded image

## Model Information

- **Architecture**: ResNet-18
- **Format**: ONNX
- **Input**: 224x224 RGB images
- **Output**: 1000 ImageNet classes
- **Expected Response Time**: < 3 seconds

## Troubleshooting

1. **Docker build fails**: Check that all required files are present
2. **Model loading fails**: Verify model.onnx file integrity
3. **Deployment fails**: Check Cerebrium CLI authentication
4. **Slow inference**: Consider using GPU-enabled deployment

## Monitoring

The deployment includes structured logging and health checks for monitoring:
- Health endpoint for uptime monitoring
- Structured JSON logs for debugging
- Performance metrics in response
