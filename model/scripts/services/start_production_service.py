#!/usr/bin/env python3
"""
Production startup script for CrescendAI Real Model Service
Handles proper environment setup, logging, and graceful shutdown
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    
    # Check if model file exists
    model_file = Path("results/final_finetuned_model.pkl")
    if not model_file.exists():
        print(f"❌ Error: Model file not found at {model_file}")
        print("Please ensure the trained model is available")
        return False
    
    print(f"✅ Model file found: {model_file}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Error: Python 3.8+ required, found {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    return True

def setup_environment():
    """Setup virtual environment and dependencies"""
    
    # Check if virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("🔄 Activating virtual environment...")
        venv_activate = Path(".venv/bin/activate")
        if venv_activate.exists():
            # Note: We can't actually activate venv from Python, user needs to do this
            print("⚠️  Please activate virtual environment first:")
            print("   source .venv/bin/activate")
            return False
        else:
            print("⚠️  Virtual environment not found. Creating...")
            subprocess.run(["uv", "venv"], check=True)
            print("⚠️  Please activate virtual environment:")
            print("   source .venv/bin/activate")
            return False
    
    print(f"✅ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
    
    # Install/update dependencies
    print("📦 Installing dependencies...")
    result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    
    print("✅ Dependencies installed")
    return True

def start_service(host="0.0.0.0", port=8002, reload=False):
    """Start the FastAPI service"""
    
    print(f"🚀 Starting CrescendAI Real Model Service...")
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📚 API docs: http://{host}:{port}/docs")
    print(f"🏥 Health check: http://{host}:{port}/health")
    print("=" * 60)
    
    # Start uvicorn server
    cmd = [
        "uvicorn", 
        "real_model_service:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info"
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down service...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the service
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Service failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start CrescendAI Real Model Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    args = parser.parse_args()
    
    print("🎹 CrescendAI Real Model Service")
    print("=" * 60)
    
    # Run checks
    if not args.skip_checks:
        if not check_requirements():
            sys.exit(1)
        
        if not setup_environment():
            sys.exit(1)
    
    # Start service
    success = start_service(args.host, args.port, args.reload)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()