"""
Cerebrium Server Testing Script

This script tests the deployed ResNet-18 model on Cerebrium platform.
It provides comprehensive testing capabilities including:
- Single image classification
- Batch testing with preset images
- Platform monitoring and performance testing
- API endpoint validation

Usage:
    python test_server.py --url <deployment-url> --image <image-path>
    python test_server.py --url <deployment-url> --preset-tests
    python test_server.py --url <deployment-url> --monitor

Requirements:
    - requests
    - PIL (Pillow)
    - numpy

Environment Variables:
    CEREBRIUM_API_KEY: Your Cerebrium API key (if required)
    DEPLOYMENT_URL: Default deployment URL
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any
import requests
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CerebriumModelTester:
    """
    Comprehensive tester for deployed ResNet-18 model on Cerebrium.
    
    Provides methods to test various aspects of the deployed model including
    functionality, performance, and platform monitoring.
    """
    
    def __init__(self, deployment_url: str, api_key: Optional[str] = None):
        """
        Initialize the Cerebrium model tester.
        
        Args:
            deployment_url: URL of the deployed model on Cerebrium
            api_key: Optional API key for authentication
        """
        self.deployment_url = deployment_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication headers if API key provided
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        
        # Test images with expected classifications
        self.test_images = {
            'tench': {
                'path': 'n01440764_tench.jpeg',
                'expected_class': 0,
                'description': 'Tench fish - should classify as class 0'
            },
            'turtle': {
                'path': 'n01667114_mud_turtle.JPEG', 
                'expected_class': 35,
                'description': 'Mud turtle - should classify as class 35'
            }
        }
        
        logger.info(f"CerebriumModelTester initialized for: {self.deployment_url}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Test the health endpoint of the deployed model.
        
        Returns:
            Health check response data
            
        Raises:
            requests.RequestException: If health check fails
        """
        logger.info("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.deployment_url}/health", timeout=30)
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"✅ Health check passed: {health_data['status']}")
            logger.info(f"Model loaded: {health_data['model_loaded']}")
            logger.info(f"Uptime: {health_data['uptime']:.2f} seconds")
            
            return health_data
            
        except requests.RequestException as e:
            logger.error(f"❌ Health check failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information from the deployed service.
        
        Returns:
            Model information data
            
        Raises:
            requests.RequestException: If request fails
        """
        logger.info("Getting model information...")
        
        try:
            response = self.session.get(f"{self.deployment_url}/info", timeout=30)
            response.raise_for_status()
            
            info_data = response.json()
            logger.info("✅ Model info retrieved successfully")
            logger.info(f"Model type: {info_data.get('model_info', {}).get('num_classes', 'Unknown')} classes")
            logger.info(f"API version: {info_data.get('api_version', 'Unknown')}")
            
            return info_data
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to get model info: {e}")
            raise
    
    def classify_image_file(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image by uploading the file.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Classification response data
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            requests.RequestException: If request fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Classifying image file: {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                params = {'top_k': top_k} if top_k > 1 else {}
                
                # Create headers without Content-Type for file upload
                headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
                
                response = requests.post(
                    f"{self.deployment_url}/predict",
                    files=files,
                    params=params,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
            
            result = response.json()
            
            if result.get('success', False):
                logger.info("✅ Classification successful")
                logger.info(f"Predicted class: {result['class_id']}")
                logger.info(f"Confidence: {result['confidence']:.4f}")
                logger.info(f"Inference time: {result['inference_time']:.3f}s")
            else:
                logger.error(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to classify image: {e}")
            raise
    
    def classify_image_url(self, image_url: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image from URL.
        
        Args:
            image_url: URL of the image to classify
            top_k: Number of top predictions to return
            
        Returns:
            Classification response data
            
        Raises:
            requests.RequestException: If request fails
        """
        logger.info(f"Classifying image from URL: {image_url}")
        
        try:
            payload = {
                'image_url': image_url,
                'top_k': top_k
            }
            
            response = self.session.post(
                f"{self.deployment_url}/predict_url",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('success', False):
                logger.info("✅ URL classification successful")
                logger.info(f"Predicted class: {result['class_id']}")
                logger.info(f"Confidence: {result['confidence']:.4f}")
                logger.info(f"Inference time: {result['inference_time']:.3f}s")
            else:
                logger.error(f"❌ URL classification failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to classify image from URL: {e}")
            raise
    
    def classify_image_base64(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image using base64 encoding.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Classification response data
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            requests.RequestException: If request fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Classifying image via base64: {image_path}")
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                'image_data': image_data,
                'top_k': top_k
            }
            
            response = self.session.post(
                f"{self.deployment_url}/predict_base64",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('success', False):
                logger.info("✅ Base64 classification successful")
                logger.info(f"Predicted class: {result['class_id']}")
                logger.info(f"Confidence: {result['confidence']:.4f}")
                logger.info(f"Inference time: {result['inference_time']:.3f}s")
            else:
                logger.error(f"❌ Base64 classification failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to classify base64 image: {e}")
            raise
    
    def run_preset_tests(self) -> Dict[str, Any]:
        """
        Run preset tests with known images and expected results.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Running preset tests...")
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'results': []
        }
        
        for test_name, test_config in self.test_images.items():
            image_path = test_config['path']
            expected_class = test_config['expected_class']
            description = test_config['description']
            
            if not Path(image_path).exists():
                logger.warning(f"Skipping test {test_name}: Image file not found: {image_path}")
                continue
            
            logger.info(f"Running test: {test_name} - {description}")
            
            test_result = {
                'test_name': test_name,
                'description': description,
                'image_path': image_path,
                'expected_class': expected_class,
                'passed': False,
                'error': None
            }
            
            try:
                # Test file upload method
                result = self.classify_image_file(image_path, top_k=3)
                
                if result.get('success', False):
                    predicted_class = result['class_id']
                    confidence = result['confidence']
                    inference_time = result['inference_time']
                    
                    test_result.update({
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'top_predictions': result.get('top_predictions', [])
                    })
                    
                    # Check if prediction matches expected class
                    if predicted_class == expected_class:
                        test_result['passed'] = True
                        test_results['passed_tests'] += 1
                        logger.info(f"✅ Test {test_name} PASSED: Predicted {predicted_class} (expected {expected_class})")
                    else:
                        test_results['failed_tests'] += 1
                        logger.error(f"❌ Test {test_name} FAILED: Predicted {predicted_class} (expected {expected_class})")
                        test_result['error'] = f"Wrong prediction: got {predicted_class}, expected {expected_class}"
                else:
                    test_results['failed_tests'] += 1
                    test_result['error'] = result.get('error', 'Classification failed')
                    logger.error(f"❌ Test {test_name} FAILED: {test_result['error']}")
                
            except Exception as e:
                test_results['failed_tests'] += 1
                test_result['error'] = str(e)
                logger.error(f"❌ Test {test_name} FAILED with exception: {e}")
            
            test_results['results'].append(test_result)
            test_results['total_tests'] += 1
        
        # Summary
        logger.info(f"Preset tests completed: {test_results['passed_tests']}/{test_results['total_tests']} passed")
        
        return test_results
    
    def performance_test(self, image_path: str, num_requests: int = 10) -> Dict[str, Any]:
        """
        Run performance tests to measure response times and throughput.
        
        Args:
            image_path: Path to test image
            num_requests: Number of requests to make
            
        Returns:
            Performance test results
        """
        logger.info(f"Running performance test with {num_requests} requests...")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            logger.info(f"Request {i+1}/{num_requests}")
            
            try:
                request_start = time.time()
                result = self.classify_image_file(image_path, top_k=1)
                request_end = time.time()
                
                if result.get('success', False):
                    response_time = request_end - request_start
                    response_times.append(response_time)
                    successful_requests += 1
                    logger.info(f"Request {i+1} completed in {response_time:.3f}s")
                else:
                    failed_requests += 1
                    logger.error(f"Request {i+1} failed")
                
            except Exception as e:
                failed_requests += 1
                logger.error(f"Request {i+1} failed with exception: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = np.mean(response_times)
            min_response_time = np.min(response_times)
            max_response_time = np.max(response_times)
            p95_response_time = np.percentile(response_times, 95)
            throughput = successful_requests / total_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = throughput = 0
        
        performance_results = {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'p95_response_time': p95_response_time,
            'throughput_rps': throughput,
            'success_rate': successful_requests / num_requests * 100
        }
        
        # Log results
        logger.info("Performance Test Results:")
        logger.info(f"  Total requests: {num_requests}")
        logger.info(f"  Successful: {successful_requests}")
        logger.info(f"  Failed: {failed_requests}")
        logger.info(f"  Success rate: {performance_results['success_rate']:.1f}%")
        logger.info(f"  Average response time: {avg_response_time:.3f}s")
        logger.info(f"  Min response time: {min_response_time:.3f}s")
        logger.info(f"  Max response time: {max_response_time:.3f}s")
        logger.info(f"  95th percentile: {p95_response_time:.3f}s")
        logger.info(f"  Throughput: {throughput:.2f} requests/second")
        
        return performance_results
    
    def monitor_platform(self) -> Dict[str, Any]:
        """
        Monitor the Cerebrium platform and deployment status.
        
        Returns:
            Platform monitoring results
        """
        logger.info("Monitoring Cerebrium platform...")
        
        monitoring_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'health_status': None,
            'model_info': None,
            'response_times': {},
            'availability': True,
            'errors': []
        }
        
        # Test health endpoint
        try:
            start_time = time.time()
            health_data = self.health_check()
            health_response_time = time.time() - start_time
            
            monitoring_results['health_status'] = health_data
            monitoring_results['response_times']['health'] = health_response_time
            
        except Exception as e:
            monitoring_results['availability'] = False
            monitoring_results['errors'].append(f"Health check failed: {str(e)}")
        
        # Test model info endpoint
        try:
            start_time = time.time()
            model_info = self.get_model_info()
            info_response_time = time.time() - start_time
            
            monitoring_results['model_info'] = model_info
            monitoring_results['response_times']['info'] = info_response_time
            
        except Exception as e:
            monitoring_results['errors'].append(f"Model info failed: {str(e)}")
        
        # Test prediction endpoint with a simple request
        test_image = None
        for test_config in self.test_images.values():
            if Path(test_config['path']).exists():
                test_image = test_config['path']
                break
        
        if test_image:
            try:
                start_time = time.time()
                result = self.classify_image_file(test_image, top_k=1)
                predict_response_time = time.time() - start_time
                
                monitoring_results['response_times']['predict'] = predict_response_time
                
                if not result.get('success', False):
                    monitoring_results['errors'].append("Prediction test failed")
                    
            except Exception as e:
                monitoring_results['errors'].append(f"Prediction test failed: {str(e)}")
        
        # Log monitoring results
        logger.info("Platform Monitoring Results:")
        logger.info(f"  Availability: {'✅ UP' if monitoring_results['availability'] else '❌ DOWN'}")
        logger.info(f"  Health response time: {monitoring_results['response_times'].get('health', 'N/A'):.3f}s")
        logger.info(f"  Info response time: {monitoring_results['response_times'].get('info', 'N/A'):.3f}s")
        logger.info(f"  Predict response time: {monitoring_results['response_times'].get('predict', 'N/A'):.3f}s")
        
        if monitoring_results['errors']:
            logger.warning(f"  Errors detected: {len(monitoring_results['errors'])}")
            for error in monitoring_results['errors']:
                logger.warning(f"    - {error}")
        
        return monitoring_results
    
    def comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run a comprehensive test suite covering all functionality.
        
        Returns:
            Complete test results
        """
        logger.info("Running comprehensive test suite...")
        
        suite_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'deployment_url': self.deployment_url,
            'tests': {}
        }
        
        # 1. Platform monitoring
        try:
            logger.info("1. Platform Monitoring Test")
            suite_results['tests']['platform_monitoring'] = self.monitor_platform()
        except Exception as e:
            logger.error(f"Platform monitoring failed: {e}")
            suite_results['tests']['platform_monitoring'] = {'error': str(e)}
        
        # 2. Preset tests
        try:
            logger.info("2. Preset Classification Tests")
            suite_results['tests']['preset_tests'] = self.run_preset_tests()
        except Exception as e:
            logger.error(f"Preset tests failed: {e}")
            suite_results['tests']['preset_tests'] = {'error': str(e)}
        
        # 3. Performance tests
        test_image = None
        for test_config in self.test_images.values():
            if Path(test_config['path']).exists():
                test_image = test_config['path']
                break
        
        if test_image:
            try:
                logger.info("3. Performance Tests")
                suite_results['tests']['performance'] = self.performance_test(test_image, num_requests=5)
            except Exception as e:
                logger.error(f"Performance tests failed: {e}")
                suite_results['tests']['performance'] = {'error': str(e)}
        
        # 4. API endpoint tests
        try:
            logger.info("4. API Endpoint Tests")
            endpoint_results = {}
            
            if test_image:
                # Test different input methods
                endpoint_results['file_upload'] = self.classify_image_file(test_image, top_k=3)
                endpoint_results['base64'] = self.classify_image_base64(test_image, top_k=3)
            
            suite_results['tests']['api_endpoints'] = endpoint_results
        except Exception as e:
            logger.error(f"API endpoint tests failed: {e}")
            suite_results['tests']['api_endpoints'] = {'error': str(e)}
        
        logger.info("Comprehensive test suite completed!")
        return suite_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Test deployed ResNet-18 model on Cerebrium')
    
    # Required arguments
    parser.add_argument('--url', required=True,
                       help='Deployment URL of the model on Cerebrium')
    
    # Optional arguments
    parser.add_argument('--api-key', 
                       help='Cerebrium API key (or set CEREBRIUM_API_KEY env var)')
    parser.add_argument('--image', 
                       help='Path to image file to classify')
    parser.add_argument('--image-url', 
                       help='URL of image to classify')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to return (default: 5)')
    
    # Test modes
    parser.add_argument('--preset-tests', action='store_true',
                       help='Run preset tests with known images')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--monitor', action='store_true',
                       help='Run platform monitoring tests')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive test suite')
    
    # Output options
    parser.add_argument('--output', 
                       help='Save results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv('CEREBRIUM_API_KEY')
    
    # Initialize tester
    try:
        tester = CerebriumModelTester(args.url, api_key)
    except Exception as e:
        logger.error(f"Failed to initialize tester: {e}")
        sys.exit(1)
    
    results = {}
    
    try:
        # Single image classification
        if args.image:
            logger.info(f"Classifying single image: {args.image}")
            result = tester.classify_image_file(args.image, args.top_k)
            results['single_image'] = result
            
            if result.get('success', False):
                print("\n🎯 Classification Result:")
                print(f"   Class ID: {result['class_id']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Inference Time: {result['inference_time']:.3f}s")
                print(f"   Top {args.top_k} Predictions:")
                for i, pred in enumerate(result.get('top_predictions', []), 1):
                    print(f"     {i}. Class {pred['class_id']}: {pred['confidence']:.4f}")
            else:
                print(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
        
        # URL image classification
        if args.image_url:
            logger.info(f"Classifying image from URL: {args.image_url}")
            result = tester.classify_image_url(args.image_url, args.top_k)
            results['url_image'] = result
        
        # Preset tests
        if args.preset_tests:
            results['preset_tests'] = tester.run_preset_tests()
        
        # Performance tests
        if args.performance:
            test_image = args.image
            if not test_image:
                # Use first available test image
                for test_config in tester.test_images.values():
                    if Path(test_config['path']).exists():
                        test_image = test_config['path']
                        break
            
            if test_image:
                results['performance'] = tester.performance_test(test_image)
            else:
                logger.error("No test image available for performance testing")
        
        # Platform monitoring
        if args.monitor:
            results['monitoring'] = tester.monitor_platform()
        
        # Comprehensive test suite
        if args.comprehensive:
            results['comprehensive'] = tester.comprehensive_test_suite()
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {args.output}")
        
        logger.info("Testing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 