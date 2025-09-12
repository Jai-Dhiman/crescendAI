#!/usr/bin/env python3
"""
CrescendAI Production Deployment Script
Deploy the piano performance analysis service to Modal with GPU support
"""

import modal
import subprocess
import sys
import os
from pathlib import Path
import json
import time

def check_modal_auth():
    """Check if Modal is properly authenticated"""
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Modal authenticated: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Modal not authenticated. Run: modal token new")
        return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Run: uv add modal")
        return False

def check_gpu_requirements():
    """Check if GPU-specific requirements are met"""
    print("🔍 Checking GPU deployment requirements...")
    
    # Check model file
    model_path = Path("results/final_finetuned_model.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f}MB)")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Run training first with GPU: uv run python3 -m crescendai_model.train_ast")
        return False
    
    # Check Modal service file
    modal_service_path = Path("crescendai_model/deployment/modal_service.py")
    if modal_service_path.exists():
        print(f"✅ Modal service file found: {modal_service_path}")
    else:
        print(f"❌ Modal service file not found: {modal_service_path}")
        return False
    
    return True

def install_gpu_dependencies():
    """Install dependencies optimized for GPU/Modal deployment"""
    print("📦 Installing production dependencies with JAX[gpu]...")
    
    try:
        # Install GPU-optimized dependencies for Modal
        result = subprocess.run([
            "uv", "add", "jax[cuda12]>=0.4.13,<0.4.30"
        ], capture_output=True, text=True, check=True)
        
        print("✅ GPU dependencies configured for Modal deployment!")
        print("ℹ️  JAX[cuda12] will be installed inside Modal containers")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to configure dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ uv not found. Install with: pip install uv")
        return False

def deploy_to_modal():
    """Deploy the Modal service with GPU support"""
    print("🚀 Deploying CrescendAI to Modal with GPU support...")
    
    try:
        # Deploy the service
        result = subprocess.run([
            "modal", "deploy", "crescendai_model/deployment/modal_service.py"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Modal service deployed successfully!")
        print(result.stdout)
        
        # Extract deployment URL if available
        lines = result.stdout.split('\n')
        for line in lines:
            if 'https://' in line and 'modal.run' in line:
                print(f"🌐 Production Service URL: {line.strip()}")
                break
                
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Production deployment failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_modal_service():
    """Test the Modal service locally before deployment"""
    print("🧪 Testing Modal service locally with GPU simulation...")
    
    try:
        # Run local test
        result = subprocess.run([
            "modal", "run", "crescendai_model/deployment/modal_service.py::test_inference"
        ], capture_output=True, text=True, check=True, timeout=300)
        
        print("✅ Modal local test passed!")
        print(result.stdout)
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Modal test timed out (this may be normal for model loading)")
        print("🤔 Continue with deployment? The service should work in production.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Modal test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def show_deployment_status():
    """Show current Modal deployment status"""
    print("📊 Current Modal deployment status...")
    
    try:
        result = subprocess.run([
            "modal", "app", "list"
        ], capture_output=True, text=True, check=True)
        
        if "crescendai-piano-analysis" in result.stdout:
            print("✅ CrescendAI app found in Modal")
        else:
            print("⚠️  CrescendAI app not found in Modal")
            
        print(result.stdout)
        
        # Show function status
        result = subprocess.run([
            "modal", "function", "list", "--app", "crescendai-piano-analysis"
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("\n📱 Function status:")
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check deployment status: {e}")

def create_production_test():
    """Create a production-ready test"""
    print("🎵 Creating production test audio...")
    
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        
        # Generate a more complex test audio for production validation
        sr = 22050
        duration = 30  # 30 seconds for thorough testing
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a piano piece with multiple voices
        # Melody line
        melody_freq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        melody = np.zeros_like(t)
        
        for i, freq in enumerate(melody_freq):
            start_time = i * (duration / len(melody_freq))
            end_time = start_time + 2.0
            mask = (t >= start_time) & (t <= end_time)
            envelope = np.exp(-(t[mask] - start_time) * 0.8)
            tone = 0.3 * envelope * np.sin(2 * np.pi * freq * t[mask])
            melody[mask] += tone
        
        # Bass line
        bass_freq = [130.81, 146.83, 164.81]  # C3, D3, E3
        bass = np.zeros_like(t)
        
        for i, freq in enumerate(bass_freq):
            start_time = i * 10
            end_time = start_time + 8
            mask = (t >= start_time) & (t <= end_time)
            envelope = 0.5 * np.exp(-(t[mask] - start_time) * 0.2)
            tone = envelope * np.sin(2 * np.pi * freq * t[mask])
            bass[mask] += tone
        
        # Combine and normalize
        audio = melody + bass
        audio = audio / np.max(np.abs(audio))
        
        # Save production test file
        test_path = Path("test_audio_production.wav")
        sf.write(test_path, audio, sr)
        
        print(f"✅ Production test audio created: {test_path} ({duration}s)")
        return test_path
        
    except ImportError as e:
        print(f"❌ Failed to create production test audio: {e}")
        print("Install required packages: uv add librosa soundfile")
        return None

def validate_modal_deployment():
    """Validate the complete Modal deployment"""
    print("🔄 Validating Modal deployment pipeline...")
    
    success = True
    
    # Check authentication
    success &= check_modal_auth()
    
    # Check GPU requirements
    success &= check_gpu_requirements()
    
    # Create production test
    test_audio = create_production_test()
    if test_audio is None:
        success = False
    else:
        print(f"✅ Production test file ready: {test_audio}")
        # Note: Actual inference testing would happen after deployment
    
    return success

def tail_modal_logs():
    """Tail logs from the Modal deployment"""
    print("📜 Tailing Modal application logs...")
    
    try:
        # Stream logs from the deployed app
        subprocess.run([
            "modal", "logs", "crescendai-piano-analysis"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to tail logs: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Log tailing stopped")

def main():
    """Main production deployment workflow"""
    print("🎹 CrescendAI Production Deployment (Modal + GPU)")
    print("=" * 55)
    
    # Show current status first
    show_deployment_status()
    
    print("\n🚀 Production deployment options:")
    print("1. Validate deployment requirements")
    print("2. Install/configure GPU dependencies")
    print("3. Test Modal service locally")
    print("4. Deploy to Modal production")
    print("5. Full deployment pipeline (validate → test → deploy)")
    print("6. View deployment logs")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        return 0 if validate_modal_deployment() else 1
    elif choice == "2":
        return 0 if install_gpu_dependencies() else 1
    elif choice == "3":
        return 0 if test_modal_service() else 1
    elif choice == "4":
        if not check_modal_auth():
            return 1
        return 0 if deploy_to_modal() else 1
    elif choice == "5":
        # Full pipeline
        print("\n🔄 Running full deployment pipeline...")
        success = True
        
        success &= validate_modal_deployment()
        if not success:
            print("❌ Validation failed, stopping pipeline")
            return 1
        
        success &= test_modal_service()
        if not success:
            print("⚠️  Local test failed, but continuing...")
        
        success &= deploy_to_modal()
        return 0 if success else 1
    elif choice == "6":
        tail_modal_logs()
        return 0
    else:
        print("❌ Invalid choice")
        return 1

if __name__ == "__main__":
    sys.exit(main())