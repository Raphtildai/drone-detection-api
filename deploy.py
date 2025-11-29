#!/usr/bin/env python3
"""
🚁 Drone Detection System Deployment Script
Supports multiple deployment platforms: Heroku, Docker, and local testing
"""

import os
import subprocess
import sys
import json
import shutil
from pathlib import Path

def print_header(message):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🚁 {message}")
    print(f"{'='*60}")

def check_requirements():
    """Check if all required tools are installed"""
    print_header("CHECKING REQUIREMENTS")
    
    required_tools = {
        'python': 'Python 3.8+',
        'pip': 'Python Package Manager',
        'git': 'Version Control'
    }
    
    optional_tools = {
        'docker': 'Container Deployment',
        'heroku': 'Heroku Platform',
        'aws': 'AWS CLI (for AWS deployment)'
    }
    
    all_available = True
    
    # Check required tools
    for tool, description in required_tools.items():
        try:
            if tool == 'python':
                result = subprocess.run([sys.executable, '--version'], capture_output=True, text=True, check=True)
                version = result.stdout.strip()
            else:
                result = subprocess.run([tool, '--version'], capture_output=True, text=True, check=True)
                version = result.stdout.strip().split('\n')[0]
            
            print(f"✅ {description}: {version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {description}: NOT FOUND")
            all_available = False
    
    # Check optional tools
    print("\n📦 Optional Tools:")
    for tool, description in optional_tools.items():
        try:
            if tool == 'aws':
                result = subprocess.run([tool, '--version'], capture_output=True, text=True, check=True)
                version = result.stdout.strip().split('\n')[0]
            else:
                result = subprocess.run([tool, '--version'], capture_output=True, text=True, check=True)
                version = result.stdout.strip().split('\n')[0]
            print(f"✅ {description}: {version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {description}: Not installed (optional)")
    
    return all_available

def check_python_dependencies():
    """Check and install Python dependencies"""
    print_header("CHECKING PYTHON DEPENDENCIES")
    
    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Try to import key dependencies
        import flask
        import torch
        import numpy
        import librosa
        
        print("✅ All core dependencies are available")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing dependencies from requirements.txt...")
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def setup_environment():
    """Setup environment variables and configuration"""
    print_header("SETTING UP ENVIRONMENT")
    
    env_template = {
        "FLASK_ENV": "production",
        "DEBUG": "False",
        "PORT": "5000",
        "MODEL_PATH": "models/best_model.pth",
        "UPLOAD_FOLDER": "/tmp/uploads",
        "MAX_CONTENT_LENGTH": "16777216"  # 16MB max file size
    }
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            for key, value in env_template.items():
                f.write(f"{key}={value}\n")
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    # Create required directories
    directories = ['models', 'static', 'templates', 'uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_application():
    """Test if the application starts correctly"""
    print_header("TESTING APPLICATION")
    
    try:
        # Test import
        import app
        print("✅ Application imports successfully")
        
        # Test model loading
        from model_loader import ModelLoader
        model_loader = ModelLoader()
        if model_loader.is_loaded():
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model not loaded (using untrained model)")
        
        # Test audio processor
        from audio_processor import AudioProcessor
        audio_processor = AudioProcessor()
        print("✅ Audio processor initialized")
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Application test failed: {e}")
        return False

def deploy_heroku():
    """Deploy to Heroku"""
    print_header("DEPLOYING TO HEROKU")
    
    try:
        # Check if already logged in
        result = subprocess.run(['heroku', 'auth:whoami'], capture_output=True, text=True)
        if result.returncode != 0:
            print("🔐 Logging into Heroku...")
            subprocess.run(['heroku', 'login'], check=True)
        else:
            print(f"✅ Already logged in as: {result.stdout.strip()}")
        
        # Create or use existing app
        app_name = input("Enter Heroku app name (or press Enter for auto-generated): ").strip()
        if app_name:
            print(f"🚀 Creating Heroku app: {app_name}")
            subprocess.run(['heroku', 'create', app_name], check=True)
        else:
            print("🚀 Creating Heroku app with auto-generated name...")
            subprocess.run(['heroku', 'create'], check=True)
        
        # Add buildpacks if needed
        print("📦 Setting up buildpacks...")
        subprocess.run(['heroku', 'buildpacks:add', 'heroku/python'], check=True)
        
        # Set environment variables
        print("⚙️  Configuring environment...")
        subprocess.run(['heroku', 'config:set', 'FLASK_ENV=production'], check=True)
        subprocess.run(['heroku', 'config:set', 'DEBUG=False'], check=True)
        
        # Initialize git if not already
        if not os.path.exists('.git'):
            print("📦 Initializing git repository...")
            subprocess.run(['git', 'init'], check=True)
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial deployment'], check=True)
        
        # Deploy to Heroku
        print("🔼 Deploying to Heroku...")
        subprocess.run(['git', 'push', 'heroku', 'main'], check=True)
        
        # Open the application
        print("🌐 Opening application...")
        subprocess.run(['heroku', 'open'], check=True)
        
        print("✅ Heroku deployment successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Heroku deployment failed: {e}")
        return False

def build_docker():
    """Build Docker image for deployment"""
    print_header("BUILDING DOCKER IMAGE")
    
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models static templates uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV DEBUG=False
ENV PORT=5000

# Run the application
CMD ["python", "app.py"]
"""
    
    try:
        # Create Dockerfile
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        print("✅ Dockerfile created")
        
        # Build Docker image
        image_name = "drone-detection-system"
        print(f"🐳 Building Docker image: {image_name}")
        subprocess.run(['docker', 'build', '-t', image_name, '.'], check=True)
        
        print("✅ Docker image built successfully")
        print(f"\n📋 Run the container with:")
        print(f"  docker run -p 5000:5000 {image_name}")
        print(f"\n📋 Or for production with environment variables:")
        print(f"  docker run -p 5000:5000 -e FLASK_ENV=production {image_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def deploy_docker():
    """Deploy using Docker"""
    if build_docker():
        print_header("DOCKER DEPLOYMENT READY")
        print("""
🎉 Your Docker image is ready! You can now:

1. Run locally:
   docker run -p 5000:5000 drone-detection-system

2. Push to Docker Hub:
   docker tag drone-detection-system yourusername/drone-detection-system
   docker push yourusername/drone-detection-system

3. Deploy to cloud platforms:
   - AWS ECS
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform
        """)
        return True
    return False

def setup_local_development():
    """Setup for local development"""
    print_header("LOCAL DEVELOPMENT SETUP")
    
    try:
        # Install development dependencies
        print("📦 Installing development dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest', 'pytest-flask'], check=True)
        
        # Create test data directory
        os.makedirs('test_data', exist_ok=True)
        
        print("✅ Local development setup complete")
        print("\n📋 To run the application locally:")
        print("  python app.py")
        print("\n📋 Or with custom settings:")
        print("  FLASK_ENV=development python app.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Local setup failed: {e}")
        return False

def generate_deployment_guide():
    """Generate a deployment guide"""
    print_header("GENERATING DEPLOYMENT GUIDE")
    
    guide = """
# 🚁 Drone Detection System - Deployment Guide

## Quick Start
1. Run the deployment script: `python deploy.py`
2. Choose your deployment method
3. Follow the on-screen instructions

## Deployment Options

### 1. Heroku (Recommended for beginners)
- Fully managed platform
- Free tier available
- Easy git-based deployments

### 2. Docker (Flexible deployment)
- Run anywhere Docker is supported
- Consistent environments
- Cloud platform compatible

### 3. Local Development
- Test and develop locally
- Debugging and development

## Environment Variables
- `FLASK_ENV`: Set to 'production' for deployment
- `DEBUG`: Set to 'False' in production
- `PORT`: Application port (default: 5000)
- `MODEL_PATH`: Path to trained model file

## Required Dependencies
- Python 3.8+
- PyTorch
- Flask
- Librosa for audio processing
- Plotly for visualizations

## Troubleshooting

### Common Issues:
1. **Model not loading**: Ensure model file exists in models/ directory
2. **Audio processing errors**: Check audio file formats and codecs
3. **Memory issues**: Reduce MAX_CONTENT_LENGTH for file uploads

### Support:
- Check logs: `heroku logs --tail` (for Heroku)
- Test locally first
- Verify all dependencies are installed
    """
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("✅ Deployment guide created: DEPLOYMENT_GUIDE.md")
    return True

def main():
    """Main deployment function"""
    print_header("DRONE DETECTION SYSTEM DEPLOYMENT")
    
    if not check_requirements():
        print("\n❌ Please install the missing requirements and try again.")
        sys.exit(1)
    
    if not check_python_dependencies():
        print("\n❌ Please install Python dependencies and try again.")
        sys.exit(1)
    
    setup_environment()
    
    if not test_application():
        print("\n❌ Application tests failed. Please fix issues before deployment.")
        sys.exit(1)
    
    # Deployment options
    print_header("CHOOSE DEPLOYMENT METHOD")
    print("1. 🚀 Heroku Deployment (Recommended)")
    print("2. 🐳 Docker Deployment")
    print("3. 💻 Local Development Setup")
    print("4. 📚 Generate Deployment Guide")
    print("5. ✅ Run All Checks Only")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        deploy_heroku()
    elif choice == '2':
        deploy_docker()
    elif choice == '3':
        setup_local_development()
    elif choice == '4':
        generate_deployment_guide()
    elif choice == '5':
        print("✅ All checks completed successfully!")
    else:
        print("❌ Invalid choice")
    
    print_header("DEPLOYMENT COMPLETE")
    print("""
🎉 Next Steps:
1. Test your deployed application
2. Monitor application logs
3. Set up monitoring and alerts
4. Consider setting up a CDN for static assets

📚 For more information, see DEPLOYMENT_GUIDE.md
    """)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Deployment cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Deployment failed with error: {e}")
        sys.exit(1)