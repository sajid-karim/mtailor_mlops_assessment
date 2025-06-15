"""
Cerebrium Deployment Script

This script helps deploy the ResNet-18 ONNX model to Cerebrium platform.
It handles Docker image building and deployment configuration.

Usage:
    python deploy.py [--build] [--deploy] [--test]

Requirements:
    - Docker installed
    - Cerebrium CLI installed (pip install cerebrium)
    - Cerebrium account and API key configured
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CerebriumDeployer:
    """Handles deployment to Cerebrium platform."""
    
    def __init__(self):
        """Initialize the deployer."""
        self.project_root = Path.cwd()
        self.required_files = [
            'model.onnx',
            'model.py',
            'app.py',
            'requirements.txt',
            'Dockerfile',
            'cerebrium.toml'
        ]
        
    def check_prerequisites(self) -> bool:
        """Check if all required files and tools are available."""
        logger.info("Checking prerequisites...")
        
        # Check required files
        missing_files = []
        for file_path in self.required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Docker found: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker not found. Please install Docker.")
            return False
        
        # Check Cerebrium CLI
        try:
            result = subprocess.run(['cerebrium', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Cerebrium CLI found: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Cerebrium CLI not found. Install with: pip install cerebrium")
            logger.info("You can still build the Docker image locally.")
        
        logger.info("✅ Prerequisites check completed")
        return True
    
    def build_docker_image(self, tag: str = "resnet18-classifier:latest") -> bool:
        """Build Docker image locally."""
        logger.info(f"Building Docker image: {tag}")
        
        try:
            # Build Docker image
            cmd = ['docker', 'build', '-t', tag, '.']
            logger.info(f"Running: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True, text=True)
            
            logger.info("✅ Docker image built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def test_docker_image(self, tag: str = "resnet18-classifier:latest") -> bool:
        """Test the Docker image locally."""
        logger.info(f"Testing Docker image: {tag}")
        
        try:
            # Run container in detached mode
            cmd = ['docker', 'run', '-d', '-p', '8000:8000', '--name', 'test-resnet18', tag]
            logger.info(f"Running: {' '.join(cmd)}")
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Wait for container to start
            logger.info("Waiting for container to start...")
            time.sleep(10)
            
            # Test health endpoint
            try:
                import requests
                response = requests.get('http://localhost:8000/health', timeout=30)
                if response.status_code == 200:
                    logger.info("✅ Health check passed")
                    logger.info(f"Response: {response.json()}")
                else:
                    logger.error(f"Health check failed: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"Failed to test health endpoint: {e}")
                return False
            finally:
                # Stop and remove container
                subprocess.run(['docker', 'stop', 'test-resnet18'], 
                             capture_output=True, check=False)
                subprocess.run(['docker', 'rm', 'test-resnet18'], 
                             capture_output=True, check=False)
            
            logger.info("✅ Docker image test completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker test failed: {e}")
            # Cleanup on failure
            subprocess.run(['docker', 'stop', 'test-resnet18'], 
                         capture_output=True, check=False)
            subprocess.run(['docker', 'rm', 'test-resnet18'], 
                         capture_output=True, check=False)
            return False
    
    def deploy_to_cerebrium(self) -> bool:
        """Deploy to Cerebrium platform."""
        logger.info("Deploying to Cerebrium...")
        
        try:
            # Check if cerebrium.toml exists
            config_file = self.project_root / 'cerebrium.toml'
            if not config_file.exists():
                logger.error("cerebrium.toml not found")
                return False
            
            # Deploy using Cerebrium CLI
            cmd = ['cerebrium', 'deploy']
            logger.info(f"Running: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True, text=True)
            
            logger.info("✅ Deployment to Cerebrium completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cerebrium deployment failed: {e}")
            return False
    
    def get_deployment_info(self) -> dict:
        """Get information about the deployment."""
        info = {
            'project_root': str(self.project_root),
            'required_files': self.required_files,
            'docker_image': 'resnet18-classifier:latest',
            'endpoints': {
                'health': '/health',
                'info': '/info',
                'predict': '/predict',
                'predict_url': '/predict_url',
                'predict_base64': '/predict_base64'
            },
            'model_info': {
                'type': 'ONNX ResNet-18',
                'input_size': '224x224 RGB',
                'output_classes': 1000,
                'model_file': 'model.onnx'
            }
        }
        
        return info
    
    def create_deployment_readme(self) -> None:
        """Create a deployment README with instructions."""
        readme_content = """# Cerebrium Deployment Guide

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
curl -X POST "https://your-deployment-url/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@your_image.jpg" \\
     -F "top_k=5"

# URL-based prediction
curl -X POST "https://your-deployment-url/predict_url" \\
     -H "Content-Type: application/json" \\
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
"""
        
        readme_path = self.project_root / 'DEPLOYMENT.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created deployment guide: {readme_path}")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy ResNet-18 model to Cerebrium')
    parser.add_argument('--build', action='store_true', 
                       help='Build Docker image')
    parser.add_argument('--test', action='store_true', 
                       help='Test Docker image locally')
    parser.add_argument('--deploy', action='store_true', 
                       help='Deploy to Cerebrium')
    parser.add_argument('--info', action='store_true', 
                       help='Show deployment information')
    parser.add_argument('--create-readme', action='store_true', 
                       help='Create deployment README')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    deployer = CerebriumDeployer()
    
    # Check prerequisites
    if not deployer.check_prerequisites():
        logger.error("Prerequisites check failed")
        sys.exit(1)
    
    success = True
    
    # Show deployment info
    if args.info:
        info = deployer.get_deployment_info()
        logger.info("Deployment Information:")
        print(json.dumps(info, indent=2))
    
    # Create deployment README
    if args.create_readme:
        deployer.create_deployment_readme()
    
    # Build Docker image
    if args.build:
        if not deployer.build_docker_image():
            success = False
    
    # Test Docker image
    if args.test and success:
        if not deployer.test_docker_image():
            success = False
    
    # Deploy to Cerebrium
    if args.deploy and success:
        if not deployer.deploy_to_cerebrium():
            success = False
    
    if success:
        logger.info("🎉 All operations completed successfully!")
        if args.deploy:
            logger.info("Your model is now deployed on Cerebrium!")
            logger.info("Check the Cerebrium dashboard for your deployment URL.")
    else:
        logger.error("❌ Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 