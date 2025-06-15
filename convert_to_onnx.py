"""
PyTorch to ONNX Model Conversion Script

This script converts the PyTorch ResNet-18 classification model to ONNX format
for deployment on Cerebrium platform.

Usage:
    python convert_to_onnx.py [--input-model PATH] [--output-model PATH] [--opset-version VERSION]

Requirements:
    - torch
    - onnx
    - onnxruntime
    - numpy
    - PIL
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

# Import the model architecture
from pytorch_model import Classifier, BasicBlock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """Handles PyTorch to ONNX model conversion and validation."""
    
    def __init__(self, pytorch_model_path: str, onnx_model_path: str, opset_version: int = 11):
        """
        Initialize the ONNX converter.
        
        Args:
            pytorch_model_path: Path to PyTorch model weights (.pth file)
            onnx_model_path: Path where ONNX model will be saved (.onnx file)
            opset_version: ONNX opset version (default: 11 for broad compatibility)
        """
        self.pytorch_model_path = Path(pytorch_model_path)
        self.onnx_model_path = Path(onnx_model_path)
        self.opset_version = opset_version
        self.model = None
        
        # Model specifications
        self.input_shape = (1, 3, 224, 224)  # (batch_size, channels, height, width)
        self.num_classes = 1000
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """Load and prepare PyTorch model for conversion."""
        logger.info(f"Loading PyTorch model from {self.pytorch_model_path}")
        
        if not self.pytorch_model_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {self.pytorch_model_path}")
        
        try:
            # Initialize model architecture (ResNet-18)
            model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes)
            
            # Load pre-trained weights
            state_dict = torch.load(self.pytorch_model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
            logger.info("PyTorch model loaded successfully")
            logger.info(f"Model architecture: ResNet-18 with {self.num_classes} classes")
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise
    
    def convert_to_onnx(self) -> None:
        """Convert PyTorch model to ONNX format."""
        if self.model is None:
            raise ValueError("PyTorch model not loaded. Call load_pytorch_model() first.")
        
        logger.info("Starting ONNX conversion...")
        
        try:
            # Create dummy input tensor
            dummy_input = torch.randn(self.input_shape)
            
            # Define input and output names
            input_names = ['input']
            output_names = ['output']
            
            # Define dynamic axes for flexible batch size
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            # Export to ONNX
            torch.onnx.export(
                self.model,                     # PyTorch model
                dummy_input,                    # Model input (dummy)
                str(self.onnx_model_path),     # Output file path
                export_params=True,             # Store trained parameter weights
                opset_version=self.opset_version,  # ONNX version
                do_constant_folding=True,       # Optimize constant folding
                input_names=input_names,        # Input tensor names
                output_names=output_names,      # Output tensor names
                dynamic_axes=dynamic_axes,      # Dynamic batch size
                verbose=False                   # Reduce verbosity
            )
            
            logger.info(f"ONNX model saved to: {self.onnx_model_path}")
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {str(e)}")
            raise
    
    def validate_onnx_model(self) -> bool:
        """Validate the converted ONNX model."""
        logger.info("Validating ONNX model...")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(str(self.onnx_model_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation: PASSED")
            
            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(str(self.onnx_model_path))
            
            # Get model input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            logger.info(f"Input shape: {input_info.shape}")
            logger.info(f"Input type: {input_info.type}")
            logger.info(f"Output shape: {output_info.shape}")
            logger.info(f"Output type: {output_info.type}")
            
            # Test with dummy data
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            ort_outputs = ort_session.run(None, {input_info.name: dummy_input})
            
            output_shape = ort_outputs[0].shape
            expected_shape = (self.input_shape[0], self.num_classes)
            
            if output_shape == expected_shape:
                logger.info(f"ONNX inference test: PASSED (output shape: {output_shape})")
                return True
            else:
                logger.error(f"Output shape mismatch. Expected: {expected_shape}, Got: {output_shape}")
                return False
                
        except Exception as e:
            logger.error(f"ONNX model validation failed: {str(e)}")
            return False
    
    def compare_outputs(self, test_image_path: str = None) -> bool:
        """Compare PyTorch and ONNX model outputs for consistency."""
        logger.info("Comparing PyTorch and ONNX model outputs...")
        
        try:
            # Use test image if provided, otherwise use dummy data
            if test_image_path and Path(test_image_path).exists():
                logger.info(f"Using test image: {test_image_path}")
                img = Image.open(test_image_path).convert('RGB')
                test_input = self.model.preprocess_numpy(img).unsqueeze(0)
            else:
                logger.info("Using dummy input for comparison")
                test_input = torch.randn(self.input_shape)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(test_input).numpy()
            
            # ONNX inference
            ort_session = ort.InferenceSession(str(self.onnx_model_path))
            input_name = ort_session.get_inputs()[0].name
            onnx_output = ort_session.run(None, {input_name: test_input.numpy()})[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            logger.info(f"Max absolute difference: {max_diff:.8f}")
            logger.info(f"Mean absolute difference: {mean_diff:.8f}")
            
            # Check if outputs are close (tolerance for floating point differences)
            tolerance = 1e-5
            if max_diff < tolerance:
                logger.info("Output comparison: PASSED")
                return True
            else:
                logger.warning(f"Output difference exceeds tolerance ({tolerance})")
                return False
                
        except Exception as e:
            logger.error(f"Output comparison failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the converted ONNX model."""
        try:
            onnx_model = onnx.load(str(self.onnx_model_path))
            
            info = {
                'model_path': str(self.onnx_model_path),
                'model_size_mb': self.onnx_model_path.stat().st_size / (1024 * 1024),
                'opset_version': self.opset_version,
                'input_shape': self.input_shape,
                'output_classes': self.num_classes,
                'graph_inputs': len(onnx_model.graph.input),
                'graph_outputs': len(onnx_model.graph.output),
                'graph_nodes': len(onnx_model.graph.node)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {}


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument(
        '--input-model', 
        type=str, 
        default='pytorch_model_weights.pth',
        help='Path to PyTorch model weights file'
    )
    parser.add_argument(
        '--output-model', 
        type=str, 
        default='model.onnx',
        help='Path for output ONNX model file'
    )
    parser.add_argument(
        '--opset-version', 
        type=int, 
        default=11,
        help='ONNX opset version (default: 11)'
    )
    parser.add_argument(
        '--test-image', 
        type=str, 
        default='n01667114_mud_turtle.JPEG',
        help='Test image for output comparison'
    )
    parser.add_argument(
        '--skip-validation', 
        action='store_true',
        help='Skip model validation steps'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = ONNXConverter(
            pytorch_model_path=args.input_model,
            onnx_model_path=args.output_model,
            opset_version=args.opset_version
        )
        
        # Load PyTorch model
        converter.load_pytorch_model()
        
        # Convert to ONNX
        converter.convert_to_onnx()
        
        if not args.skip_validation:
            # Validate ONNX model
            validation_passed = converter.validate_onnx_model()
            
            # Compare outputs
            comparison_passed = converter.compare_outputs(args.test_image)
            
            if validation_passed and comparison_passed:
                logger.info("✅ Conversion completed successfully!")
            else:
                logger.warning("⚠️ Conversion completed with warnings")
        
        # Display model information
        model_info = converter.get_model_info()
        if model_info:
            logger.info("\n" + "="*50)
            logger.info("ONNX MODEL INFORMATION")
            logger.info("="*50)
            for key, value in model_info.items():
                if key == 'model_size_mb':
                    logger.info(f"{key}: {value:.2f} MB")
                else:
                    logger.info(f"{key}: {value}")
            logger.info("="*50)
        
        logger.info(f"ONNX model ready for deployment: {args.output_model}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
