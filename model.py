"""
ONNX Model Inference and Image Preprocessing

This module contains classes for loading and running inference with the ONNX model,
and preprocessing images for the ResNet-18 classification model.

Classes:
    - ImagePreprocessor: Handles image preprocessing pipeline
    - ONNXModelInference: Handles ONNX model loading and inference

Usage:
    preprocessor = ImagePreprocessor()
    model = ONNXModelInference('model.onnx')
    
    processed_image = preprocessor.preprocess('image.jpg')
    prediction = model.predict(processed_image)
"""

import logging
from pathlib import Path
from typing import Union, Tuple, List
import numpy as np
from PIL import Image
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing for ResNet-18 model inference.
    
    Implements the same preprocessing pipeline as the original PyTorch model:
    1. Convert to RGB format
    2. Resize to 224x224 (bilinear interpolation)
    3. Normalize pixel values to [0,1] by dividing by 255
    4. Normalize using ImageNet statistics
    """
    
    def __init__(self):
        """Initialize the image preprocessor with ImageNet normalization parameters."""
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Target image dimensions
        self.target_size = (224, 224)
        
        logger.info("ImagePreprocessor initialized with ImageNet normalization")
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """
        Resize image using bilinear interpolation.
        
        Args:
            image: PIL Image object
            size: Target size as (width, height)
            
        Returns:
            Resized PIL Image
        """
        return image.resize(size, Image.BILINEAR)
    
    def to_rgb(self, image: Image.Image) -> Image.Image:
        """
        Convert image to RGB format if needed.
        
        Args:
            image: PIL Image object
            
        Returns:
            RGB PIL Image
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image_array: Image array with shape (H, W, C) and values in [0,1]
            
        Returns:
            Normalized image array
        """
        # Normalize each channel: (pixel - mean) / std
        normalized = (image_array - self.mean) / self.std
        return normalized.astype(np.float32)
    
    def preprocess_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array with shape (1, 3, 224, 224)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image processing fails
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path)
            return self.preprocess_from_pil(image)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image from {image_path}: {str(e)}")
            raise
    
    def preprocess_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image object.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image array with shape (1, 3, 224, 224)
        """
        try:
            # Step 1: Convert to RGB
            image = self.to_rgb(image)
            
            # Step 2: Resize to 224x224
            image = self.resize_image(image, self.target_size)
            
            # Step 3: Convert to numpy array and normalize to [0,1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Step 4: Apply ImageNet normalization
            image_array = self.normalize_image(image_array)
            
            # Step 5: Convert from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Step 6: Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.debug(f"Preprocessed image shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess PIL image: {str(e)}")
            raise
    
    def preprocess_from_numpy(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess numpy array image.
        
        Args:
            image_array: Image array with shape (H, W, C) and values in [0, 255]
            
        Returns:
            Preprocessed image array with shape (1, 3, 224, 224)
        """
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            
            image = Image.fromarray(image_array)
            return self.preprocess_from_pil(image)
            
        except Exception as e:
            logger.error(f"Failed to preprocess numpy image: {str(e)}")
            raise
    
    def get_preprocessing_info(self) -> dict:
        """
        Get information about preprocessing parameters.
        
        Returns:
            Dictionary with preprocessing configuration
        """
        return {
            'target_size': self.target_size,
            'normalization_mean': self.mean.tolist(),
            'normalization_std': self.std.tolist(),
            'output_format': 'NCHW (batch, channels, height, width)',
            'pixel_range': '[0, 1] -> normalized with ImageNet stats'
        }


class ONNXModelInference:
    """
    Handles ONNX model loading and inference for ResNet-18 classification.
    
    Provides methods to load the ONNX model and run predictions on preprocessed images.
    """
    
    def __init__(self, model_path: Union[str, Path], providers: List[str] = None):
        """
        Initialize ONNX model inference.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['CPUExecutionProvider'])
        """
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        # Set default providers if none specified
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.providers = providers
        
        # Load the model
        self.load_model()
        
        logger.info(f"ONNXModelInference initialized with model: {self.model_path}")
    
    def load_model(self) -> None:
        """
        Load ONNX model and initialize inference session.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        
        try:
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=self.providers
            )
            
            # Get input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            logger.info("Model loaded successfully")
            logger.info(f"Input: {self.input_name} {self.input_shape}")
            logger.info(f"Output: {self.output_name} {self.output_shape}")
            logger.info(f"Execution providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image data.
        
        Args:
            input_data: Preprocessed image array with shape (1, 3, 224, 224)
            
        Returns:
            Model predictions with shape (1, 1000) - class probabilities
            
        Raises:
            ValueError: If input shape is incorrect
            Exception: If inference fails
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input shape
        expected_shape = (1, 3, 224, 224)
        if input_data.shape != expected_shape:
            raise ValueError(f"Input shape mismatch. Expected: {expected_shape}, Got: {input_data.shape}")
        
        # Ensure correct data type
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        try:
            # Run inference
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: input_data}
            )
            
            predictions = outputs[0]
            logger.debug(f"Inference completed. Output shape: {predictions.shape}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    def predict_class(self, input_data: np.ndarray) -> int:
        """
        Get predicted class ID from preprocessed image data.
        
        Args:
            input_data: Preprocessed image array with shape (1, 3, 224, 224)
            
        Returns:
            Predicted class ID (0-999 for ImageNet)
        """
        predictions = self.predict(input_data)
        class_id = np.argmax(predictions[0])
        return int(class_id)
    
    def predict_top_k(self, input_data: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k predictions with class IDs and confidence scores.
        
        Args:
            input_data: Preprocessed image array with shape (1, 3, 224, 224)
            k: Number of top predictions to return
            
        Returns:
            List of (class_id, confidence) tuples sorted by confidence
        """
        predictions = self.predict(input_data)
        probabilities = predictions[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        
        # Create list of (class_id, confidence) tuples
        top_k_predictions = [
            (int(idx), float(probabilities[idx])) 
            for idx in top_k_indices
        ]
        
        return top_k_predictions
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded ONNX model.
        
        Returns:
            Dictionary with model information
        """
        if self.session is None:
            return {"error": "Model not loaded"}
        
        return {
            'model_path': str(self.model_path),
            'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
            'input_name': self.input_name,
            'output_name': self.output_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'execution_providers': self.session.get_providers(),
            'num_classes': 1000
        }


class ModelPipeline:
    """
    Complete inference pipeline combining preprocessing and model inference.
    
    Provides a high-level interface for end-to-end image classification.
    """
    
    def __init__(self, model_path: Union[str, Path], providers: List[str] = None):
        """
        Initialize the complete inference pipeline.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers
        """
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModelInference(model_path, providers)
        
        logger.info("ModelPipeline initialized successfully")
    
    def classify_image(self, image_path: Union[str, Path]) -> int:
        """
        Classify image from file path and return class ID.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted class ID (0-999)
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_from_path(image_path)
        
        # Get prediction
        class_id = self.model.predict_class(processed_image)
        
        logger.info(f"Image {image_path} classified as class {class_id}")
        return class_id
    
    def classify_image_with_confidence(self, image_path: Union[str, Path], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Classify image and return top-k predictions with confidence scores.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_id, confidence) tuples
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_from_path(image_path)
        
        # Get top-k predictions
        predictions = self.model.predict_top_k(processed_image, top_k)
        
        logger.info(f"Image {image_path} top-{top_k} predictions: {predictions}")
        return predictions
    
    def classify_pil_image(self, image: Image.Image) -> int:
        """
        Classify PIL Image object and return class ID.
        
        Args:
            image: PIL Image object
            
        Returns:
            Predicted class ID (0-999)
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_from_pil(image)
        
        # Get prediction
        class_id = self.model.predict_class(processed_image)
        
        return class_id
    
    def classify_pil_image_with_confidence(self, image: Image.Image, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Classify PIL Image object and return top-k predictions with confidence scores.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_id, confidence) tuples
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_from_pil(image)
        
        # Get top-k predictions
        predictions = self.model.predict_top_k(processed_image, top_k)
        
        logger.info(f"PIL image top-{top_k} predictions: {predictions}")
        return predictions


# Example usage and testing functions
def test_preprocessing():
    """Test image preprocessing functionality."""
    print("Testing ImagePreprocessor...")
    
    preprocessor = ImagePreprocessor()
    
    # Test with sample image
    try:
        processed = preprocessor.preprocess_from_path('n01667114_mud_turtle.JPEG')
        print(f"✅ Preprocessing successful. Output shape: {processed.shape}")
        print(f"✅ Output dtype: {processed.dtype}")
        print(f"✅ Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Print preprocessing info
        info = preprocessor.get_preprocessing_info()
        print("Preprocessing configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")


def test_model_inference():
    """Test ONNX model inference functionality."""
    print("\nTesting ONNXModelInference...")
    
    try:
        model = ONNXModelInference('model.onnx')
        
        # Test with dummy data
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        predictions = model.predict(dummy_input)
        
        print(f"✅ Inference successful. Output shape: {predictions.shape}")
        print(f"✅ Output dtype: {predictions.dtype}")
        
        # Test class prediction
        class_id = model.predict_class(dummy_input)
        print(f"✅ Predicted class: {class_id}")
        
        # Test top-k predictions
        top_k = model.predict_top_k(dummy_input, k=3)
        print(f"✅ Top-3 predictions: {top_k}")
        
        # Print model info
        info = model.get_model_info()
        print("Model information:")
        for key, value in info.items():
            if key == 'model_size_mb':
                print(f"  {key}: {value:.2f} MB")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ Model inference failed: {e}")


def test_complete_pipeline():
    """Test complete inference pipeline."""
    print("\nTesting ModelPipeline...")
    
    try:
        pipeline = ModelPipeline('model.onnx')
        
        # Test with sample image
        class_id = pipeline.classify_image('n01667114_mud_turtle.JPEG')
        print(f"✅ Pipeline classification successful. Class ID: {class_id}")
        
        # Test with confidence scores
        predictions = pipeline.classify_image_with_confidence('n01667114_mud_turtle.JPEG', top_k=3)
        print(f"✅ Pipeline with confidence successful. Top-3: {predictions}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    print("="*60)
    print("TESTING MODEL.PY COMPONENTS")
    print("="*60)
    
    test_preprocessing()
    test_model_inference()
    test_complete_pipeline()
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60) 