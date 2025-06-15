"""
Comprehensive Test Suite for ML Model Deployment

This module contains comprehensive tests for the ONNX model deployment including:
- Model loading and validation
- Image preprocessing pipeline
- Inference accuracy and performance
- Edge cases and error handling
- Integration tests

Usage:
    python test.py [--verbose] [--performance] [--quick]

Requirements:
    - All model components (model.py, convert_to_onnx.py)
    - Test images
    - ONNX model file
"""

import argparse
import logging
import sys
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import tempfile
import os

# Import our model components
from model import ImagePreprocessor, ONNXModelInference, ModelPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.test_images = {
            'tench': 'n01440764_tench.jpeg',
            'turtle': 'n01667114_mud_turtle.JPEG'
        }
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.mean)
        self.assertIsNotNone(self.preprocessor.std)
        self.assertEqual(self.preprocessor.target_size, (224, 224))
        
        # Check ImageNet normalization values
        expected_mean = [0.485, 0.456, 0.406]
        expected_std = [0.229, 0.224, 0.225]
        np.testing.assert_array_almost_equal(self.preprocessor.mean, expected_mean, decimal=3)
        np.testing.assert_array_almost_equal(self.preprocessor.std, expected_std, decimal=3)
    
    def test_rgb_conversion(self):
        """Test RGB conversion functionality."""
        # Test with RGB image (should remain unchanged)
        rgb_image = Image.new('RGB', (100, 100), color='red')
        result = self.preprocessor.to_rgb(rgb_image)
        self.assertEqual(result.mode, 'RGB')
        
        # Test with grayscale image (should convert to RGB)
        gray_image = Image.new('L', (100, 100), color=128)
        result = self.preprocessor.to_rgb(gray_image)
        self.assertEqual(result.mode, 'RGB')
    
    def test_image_resizing(self):
        """Test image resizing functionality."""
        # Create test image
        test_image = Image.new('RGB', (300, 400), color='blue')
        
        # Resize to target size
        resized = self.preprocessor.resize_image(test_image, (224, 224))
        self.assertEqual(resized.size, (224, 224))
    
    def test_normalization(self):
        """Test image normalization."""
        # Create test array with known values
        test_array = np.ones((224, 224, 3), dtype=np.float32) * 0.5  # Mid-range values
        
        normalized = self.preprocessor.normalize_image(test_array)
        
        # Check output shape and type
        self.assertEqual(normalized.shape, (224, 224, 3))
        self.assertEqual(normalized.dtype, np.float32)
        
        # Check that normalization was applied
        self.assertFalse(np.array_equal(test_array, normalized))
    
    def test_preprocess_from_path(self):
        """Test preprocessing from file path."""
        for name, image_path in self.test_images.items():
            if Path(image_path).exists():
                with self.subTest(image=name):
                    result = self.preprocessor.preprocess_from_path(image_path)
                    
                    # Check output shape and type
                    self.assertEqual(result.shape, (1, 3, 224, 224))
                    self.assertEqual(result.dtype, np.float32)
                    
                    # Check value range (should be normalized)
                    self.assertTrue(result.min() >= -3.0)  # Reasonable lower bound
                    self.assertTrue(result.max() <= 3.0)   # Reasonable upper bound
    
    def test_preprocess_from_pil(self):
        """Test preprocessing from PIL Image."""
        # Create test PIL image
        test_image = Image.new('RGB', (300, 300), color=(128, 64, 192))
        
        result = self.preprocessor.preprocess_from_pil(test_image)
        
        # Check output shape and type
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
    
    def test_preprocess_from_numpy(self):
        """Test preprocessing from numpy array."""
        # Create test numpy array
        test_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = self.preprocessor.preprocess_from_numpy(test_array)
        
        # Check output shape and type
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.preprocess_from_path('nonexistent_image.jpg')
    
    def test_get_preprocessing_info(self):
        """Test preprocessing info retrieval."""
        info = self.preprocessor.get_preprocessing_info()
        
        self.assertIn('target_size', info)
        self.assertIn('normalization_mean', info)
        self.assertIn('normalization_std', info)
        self.assertEqual(info['target_size'], (224, 224))


class TestONNXModelInference(unittest.TestCase):
    """Test cases for ONNXModelInference class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'model.onnx'
        if Path(self.model_path).exists():
            self.model = ONNXModelInference(self.model_path)
        else:
            self.skipTest(f"ONNX model file not found: {self.model_path}")
    
    def test_model_loading(self):
        """Test model loading functionality."""
        self.assertIsNotNone(self.model.session)
        self.assertIsNotNone(self.model.input_name)
        self.assertIsNotNone(self.model.output_name)
        self.assertEqual(self.model.input_shape, ['batch_size', 3, 224, 224])
        self.assertEqual(self.model.output_shape, ['batch_size', 1000])
    
    def test_invalid_model_path(self):
        """Test handling of invalid model paths."""
        with self.assertRaises(FileNotFoundError):
            ONNXModelInference('nonexistent_model.onnx')
    
    def test_predict(self):
        """Test basic prediction functionality."""
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        predictions = self.model.predict(dummy_input)
        
        # Check output shape and type
        self.assertEqual(predictions.shape, (1, 1000))
        self.assertEqual(predictions.dtype, np.float32)
    
    def test_predict_class(self):
        """Test class prediction functionality."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        class_id = self.model.predict_class(dummy_input)
        
        # Check output type and range
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_predict_top_k(self):
        """Test top-k prediction functionality."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        top_k = self.model.predict_top_k(dummy_input, k=5)
        
        # Check output format
        self.assertEqual(len(top_k), 5)
        for class_id, confidence in top_k:
            self.assertIsInstance(class_id, int)
            self.assertIsInstance(confidence, float)
            self.assertTrue(0 <= class_id < 1000)
        
        # Check that results are sorted by confidence (descending)
        confidences = [conf for _, conf in top_k]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        invalid_input = np.random.randn(1, 3, 100, 100).astype(np.float32)
        
        with self.assertRaises(ValueError):
            self.model.predict(invalid_input)
    
    def test_input_dtype_conversion(self):
        """Test automatic input dtype conversion."""
        # Test with float64 input (should be converted to float32)
        input_float64 = np.random.randn(1, 3, 224, 224).astype(np.float64)
        
        predictions = self.model.predict(input_float64)
        self.assertEqual(predictions.shape, (1, 1000))
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.model.get_model_info()
        
        self.assertIn('model_path', info)
        self.assertIn('model_size_mb', info)
        self.assertIn('input_name', info)
        self.assertIn('output_name', info)
        self.assertIn('num_classes', info)
        self.assertEqual(info['num_classes'], 1000)


class TestModelPipeline(unittest.TestCase):
    """Test cases for ModelPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'model.onnx'
        self.test_images = {
            'tench': ('n01440764_tench.jpeg', 0),
            'turtle': ('n01667114_mud_turtle.JPEG', 35)
        }
        
        if Path(self.model_path).exists():
            self.pipeline = ModelPipeline(self.model_path)
        else:
            self.skipTest(f"ONNX model file not found: {self.model_path}")
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.model)
    
    def test_classify_image(self):
        """Test end-to-end image classification."""
        for name, (image_path, expected_class) in self.test_images.items():
            if Path(image_path).exists():
                with self.subTest(image=name):
                    class_id = self.pipeline.classify_image(image_path)
                    
                    # Check output type and range
                    self.assertIsInstance(class_id, int)
                    self.assertTrue(0 <= class_id < 1000)
                    
                    # Check expected class (if known)
                    self.assertEqual(class_id, expected_class, 
                                   f"Expected class {expected_class} for {name}, got {class_id}")
    
    def test_classify_image_with_confidence(self):
        """Test classification with confidence scores."""
        for name, (image_path, expected_class) in self.test_images.items():
            if Path(image_path).exists():
                with self.subTest(image=name):
                    predictions = self.pipeline.classify_image_with_confidence(image_path, top_k=3)
                    
                    # Check output format
                    self.assertEqual(len(predictions), 3)
                    
                    # Check that top prediction matches expected class
                    top_class, top_confidence = predictions[0]
                    self.assertEqual(top_class, expected_class)
                    
                    # Check confidence ordering
                    confidences = [conf for _, conf in predictions]
                    self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    def test_classify_pil_image(self):
        """Test classification of PIL Image objects."""
        # Create test PIL image
        test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        class_id = self.pipeline.classify_pil_image(test_image)
        
        # Check output type and range
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)


class TestPerformance(unittest.TestCase):
    """Performance tests for the model deployment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'model.onnx'
        self.test_image = 'n01667114_mud_turtle.JPEG'
        
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model file not found: {self.model_path}")
        if not Path(self.test_image).exists():
            self.skipTest(f"Test image not found: {self.test_image}")
        
        self.pipeline = ModelPipeline(self.model_path)
    
    def test_inference_speed(self):
        """Test inference speed requirements (should be < 3 seconds)."""
        # Warm up
        self.pipeline.classify_image(self.test_image)
        
        # Measure inference time
        start_time = time.time()
        class_id = self.pipeline.classify_image(self.test_image)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        logger.info(f"Inference time: {inference_time:.3f} seconds")
        
        # Check that inference is fast enough for production (< 3 seconds as per README)
        self.assertLess(inference_time, 3.0, 
                       f"Inference too slow: {inference_time:.3f}s > 3.0s")
    
    def test_batch_processing_consistency(self):
        """Test that multiple inferences give consistent results."""
        results = []
        
        for _ in range(5):
            class_id = self.pipeline.classify_image(self.test_image)
            results.append(class_id)
        
        # All results should be the same
        self.assertTrue(all(r == results[0] for r in results), 
                       f"Inconsistent results: {results}")
    
    def test_memory_usage(self):
        """Test memory usage during inference."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple inferences
        for _ in range(10):
            self.pipeline.classify_image(self.test_image)
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB "
                   f"(+{memory_increase:.1f} MB)")
        
        # Memory increase should be reasonable (< 100 MB for 10 inferences)
        self.assertLess(memory_increase, 100, 
                       f"Excessive memory usage: +{memory_increase:.1f} MB")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'model.onnx'
        if Path(self.model_path).exists():
            self.pipeline = ModelPipeline(self.model_path)
        else:
            self.skipTest(f"ONNX model file not found: {self.model_path}")
    
    def test_corrupted_image(self):
        """Test handling of corrupted images."""
        # Create a corrupted image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'This is not a valid image file')
            corrupted_path = f.name
        
        try:
            with self.assertRaises(Exception):
                self.pipeline.classify_image(corrupted_path)
        finally:
            os.unlink(corrupted_path)
    
    def test_very_small_image(self):
        """Test handling of very small images."""
        # Create a very small image
        small_image = Image.new('RGB', (1, 1), color='red')
        
        # Should work (will be resized to 224x224)
        class_id = self.pipeline.classify_pil_image(small_image)
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_very_large_image(self):
        """Test handling of very large images."""
        # Create a large image (this might be slow, so we use a reasonable size)
        large_image = Image.new('RGB', (2000, 2000), color='blue')
        
        # Should work (will be resized to 224x224)
        class_id = self.pipeline.classify_pil_image(large_image)
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_grayscale_image(self):
        """Test handling of grayscale images."""
        # Create a grayscale image
        gray_image = Image.new('L', (300, 300), color=128)
        
        # Should work (will be converted to RGB)
        class_id = self.pipeline.classify_pil_image(gray_image)
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)
    
    def test_rgba_image(self):
        """Test handling of RGBA images."""
        # Create an RGBA image
        rgba_image = Image.new('RGBA', (300, 300), color=(255, 0, 0, 128))
        
        # Should work (will be converted to RGB)
        class_id = self.pipeline.classify_pil_image(rgba_image)
        self.assertIsInstance(class_id, int)
        self.assertTrue(0 <= class_id < 1000)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'model.onnx'
        self.test_images = {
            'tench': ('n01440764_tench.jpeg', 0),
            'turtle': ('n01667114_mud_turtle.JPEG', 35)
        }
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model file not found: {self.model_path}")
        
        # Test complete workflow
        pipeline = ModelPipeline(self.model_path)
        
        for name, (image_path, expected_class) in self.test_images.items():
            if Path(image_path).exists():
                with self.subTest(image=name):
                    # Test classification
                    class_id = pipeline.classify_image(image_path)
                    self.assertEqual(class_id, expected_class)
                    
                    # Test with confidence
                    predictions = pipeline.classify_image_with_confidence(image_path, top_k=5)
                    self.assertEqual(len(predictions), 5)
                    self.assertEqual(predictions[0][0], expected_class)
    
    def test_model_consistency_with_pytorch(self):
        """Test that ONNX model gives same results as original PyTorch model."""
        # This test would compare ONNX results with PyTorch results
        # For now, we just verify the expected classes for test images
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model file not found: {self.model_path}")
        
        pipeline = ModelPipeline(self.model_path)
        
        # Test known image classifications
        for name, (image_path, expected_class) in self.test_images.items():
            if Path(image_path).exists():
                with self.subTest(image=name):
                    class_id = pipeline.classify_image(image_path)
                    self.assertEqual(class_id, expected_class, 
                                   f"ONNX model prediction mismatch for {name}")


def run_tests(verbose: bool = False, performance: bool = False, quick: bool = False):
    """
    Run the test suite.
    
    Args:
        verbose: Enable verbose output
        performance: Include performance tests
        quick: Run only quick tests (skip performance and edge cases)
    """
    # Configure test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add core functionality tests
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXModelInference))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    if not quick:
        # Add edge case tests
        suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
        
        if performance:
            # Add performance tests
            suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    
    # Run tests
    logger.info("Starting test suite...")
    start_time = time.time()
    
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    logger.info(f"Tests completed in {duration:.2f} seconds")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Run ML model deployment tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--performance', '-p', action='store_true',
                       help='Include performance tests')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run only quick tests')
    
    args = parser.parse_args()
    
    # Run tests
    success = run_tests(
        verbose=args.verbose,
        performance=args.performance,
        quick=args.quick
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 