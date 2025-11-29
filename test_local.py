#!/usr/bin/env python3
"""
Local Development Server for Drone Detection System
Run this script to test the application locally before deployment
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def check_environment():
    """Check if all required environment variables are set"""
    required_dirs = ['models', 'static', 'templates']
    missing_dirs = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("Creating missing directories...")
        for directory in missing_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
    
    # Check for model file
    model_paths = [
        'models/best_model.pth',
        'best_model.pth',
        '../models/best_model.pth'
    ]
    
    model_found = any(os.path.exists(path) for path in model_paths)
    if not model_found:
        print("⚠️  No trained model file found. The system will use an untrained model.")
        print("   For best results, place a trained model in the models/ directory.")
    else:
        print("✅ Model file found")

def main():
    """Main function to run the local server"""
    print("🚁 Drone Detection System - Local Development Server")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check environment
    check_environment()
    
    try:
        # Import and run the application
        from app import app
        
        print("\n✅ Application loaded successfully")
        print("📊 Starting local development server...")
        print("🌐 Access the application at: http://localhost:5000")
        print("📝 Logs are being written to: app.log")
        print("⏹️  Press Ctrl+C to stop the server\n")
        
        # Run the application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        print("\n📋 Troubleshooting steps:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if app.py exists in the current directory")
        print("3. Verify Python path and module structure")
        return 1
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)