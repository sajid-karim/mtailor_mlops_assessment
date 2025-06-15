"""
Cerebrium Deployment Application

FastAPI application for serving the ONNX ResNet-18 model on Cerebrium platform.
Provides REST API endpoints for image classification.

Endpoints:
    POST /predict - Classify an uploaded image
    POST /predict_url - Classify an image from URL
    GET /health - Health check endpoint
    GET /info - Model information endpoint

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import structlog
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

# Import our model components
from model import ModelPipeline

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="ResNet-18 Image Classification API",
    description="ONNX-based ResNet-18 model for ImageNet classification on Cerebrium",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model pipeline instance
model_pipeline: Optional[ModelPipeline] = None

# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for URL-based prediction."""
    image_url: str = Field(..., description="URL of the image to classify")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool = Field(..., description="Whether the prediction was successful")
    class_id: int = Field(..., description="Predicted class ID (0-999)")
    confidence: float = Field(..., description="Confidence score for top prediction")
    top_predictions: List[Dict[str, Union[int, float]]] = Field(
        ..., description="Top-K predictions with class IDs and confidence scores"
    )
    inference_time: float = Field(..., description="Inference time in seconds")
    model_info: Dict = Field(..., description="Model information")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")
    uptime: float = Field(..., description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")

class Base64PredictionRequest(BaseModel):
    """Request model for base64 image prediction."""
    image_data: str = Field(..., description="Base64 encoded image data")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")

# Global variables for tracking
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the model pipeline on startup."""
    global model_pipeline
    
    try:
        logger.info("Starting up Cerebrium application...")
        
        # Check if model file exists
        model_path = "model.onnx"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize model pipeline
        logger.info("Loading ONNX model pipeline...")
        model_pipeline = ModelPipeline(model_path)
        
        logger.info("Model pipeline loaded successfully", 
                   model_info=model_pipeline.model.get_model_info())
        
    except Exception as e:
        logger.error("Failed to initialize model pipeline", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Cerebrium application...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    current_time = time.time()
    uptime = current_time - start_time
    
    return HealthResponse(
        status="healthy" if model_pipeline is not None else "unhealthy",
        model_loaded=model_pipeline is not None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(current_time)),
        uptime=uptime
    )

@app.get("/info")
async def model_info():
    """Get model information."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = model_pipeline.model.get_model_info()
        preprocessing_info = model_pipeline.preprocessor.get_preprocessing_info()
        
        return {
            "model_info": info,
            "preprocessing_info": preprocessing_info,
            "api_version": "1.0.0",
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"],
            "max_image_size": "10MB",
            "expected_inference_time": "< 3 seconds"
        }
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of top predictions")
):
    """
    Classify an uploaded image file.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        top_k: Number of top predictions to return (1-10)
    
    Returns:
        PredictionResponse with classification results
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Expected image file."
        )
    
    try:
        start_time = time.time()
        
        # Read and process image
        logger.info("Processing uploaded image", filename=file.filename, content_type=file.content_type)
        
        # Read file content
        image_data = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions
        predictions = model_pipeline.classify_pil_image_with_confidence(image, top_k=top_k)
        
        inference_time = time.time() - start_time
        
        # Format response
        top_predictions = [
            {"class_id": int(class_id), "confidence": float(confidence)}
            for class_id, confidence in predictions
        ]
        
        response = PredictionResponse(
            success=True,
            class_id=predictions[0][0],
            confidence=predictions[0][1],
            top_predictions=top_predictions,
            inference_time=inference_time,
            model_info=model_pipeline.model.get_model_info()
        )
        
        logger.info("Image classification completed", 
                   filename=file.filename,
                   predicted_class=predictions[0][0],
                   confidence=predictions[0][1],
                   inference_time=inference_time)
        
        return response
        
    except Exception as e:
        logger.error("Image classification failed", 
                    filename=file.filename, 
                    error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Classification failed: {str(e)}"
        )

@app.post("/predict_url", response_model=PredictionResponse)
async def predict_image_url(request: PredictionRequest):
    """
    Classify an image from URL.
    
    Args:
        request: PredictionRequest with image URL and top_k
    
    Returns:
        PredictionResponse with classification results
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import aiohttp
        
        start_time = time.time()
        
        logger.info("Processing image from URL", url=request.image_url)
        
        # Download image from URL
        async with aiohttp.ClientSession() as session:
            async with session.get(request.image_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download image: HTTP {response.status}"
                    )
                
                image_data = await response.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions
        predictions = model_pipeline.classify_pil_image_with_confidence(image, top_k=request.top_k)
        
        inference_time = time.time() - start_time
        
        # Format response
        top_predictions = [
            {"class_id": int(class_id), "confidence": float(confidence)}
            for class_id, confidence in predictions
        ]
        
        response = PredictionResponse(
            success=True,
            class_id=predictions[0][0],
            confidence=predictions[0][1],
            top_predictions=top_predictions,
            inference_time=inference_time,
            model_info=model_pipeline.model.get_model_info()
        )
        
        logger.info("URL image classification completed",
                   url=request.image_url,
                   predicted_class=predictions[0][0],
                   confidence=predictions[0][1],
                   inference_time=inference_time)
        
        return response
        
    except Exception as e:
        logger.error("URL image classification failed",
                    url=request.image_url,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_base64_image(request: Base64PredictionRequest):
    """
    Classify a base64 encoded image.
    
    Args:
        request: Base64PredictionRequest with image data and top_k
    
    Returns:
        PredictionResponse with classification results
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        logger.info("Processing base64 image")
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            image_data = request.image_data
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        predictions = model_pipeline.classify_pil_image_with_confidence(image, top_k=request.top_k)
        
        inference_time = time.time() - start_time
        
        # Format response
        top_predictions = [
            {"class_id": int(class_id), "confidence": float(confidence)}
            for class_id, confidence in predictions
        ]
        
        response = PredictionResponse(
            success=True,
            class_id=predictions[0][0],
            confidence=predictions[0][1],
            top_predictions=top_predictions,
            inference_time=inference_time,
            model_info=model_pipeline.model.get_model_info()
        )
        
        logger.info("Base64 image classification completed",
                   predicted_class=predictions[0][0],
                   confidence=predictions[0][1],
                   inference_time=inference_time)
        
        return response
        
    except Exception as e:
        logger.error("Base64 image classification failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type="InternalError"
        ).dict()
    )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 