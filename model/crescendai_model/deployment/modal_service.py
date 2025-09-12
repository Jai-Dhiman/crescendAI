#!/usr/bin/env python3
"""
CrescendAI Modal Service
JAX/Flax Audio Spectrogram Transformer deployment for piano performance analysis
"""

import modal
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import json
import time

# Configure JAX for inference (CPU mode for cold starts, can optimize later)
jax.config.update('jax_platform_name', 'cpu')

# Modal app setup
app = modal.App("crescendai-piano-analysis")

# Create Modal image with all dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        # Core ML frameworks
        "jax[cpu]>=0.4.13,<0.4.20",  # Pin to avoid define_bool_state issue
        "flax>=0.7.2,<0.8.0", 
        "optax>=0.1.7,<0.2.0",
        # Scientific computing
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<1.12.0",
        "scikit-learn>=1.3.0,<1.7.0",  # Pin to match pickle version
        # Audio processing
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        # Data handling  
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "datasets>=4.0.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # Web framework
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
    ])
    .add_local_file(
        "/Users/jdhiman/Documents/crescendai/model/results/final_finetuned_model.pkl",
        "/model/final_finetuned_model.pkl"
    )
    .add_local_dir(
        "/Users/jdhiman/Documents/crescendai/model/crescendai_model",
        "/app/crescendai_model"
    )
)

# Model loading and caching (cold loading strategy)
class ModelManager:
    """Manages JAX/Flax model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.params = None
        self.model_config = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained JAX/Flax model"""
        if self.is_loaded:
            return
            
        start_time = time.time()
        logging.info("Loading JAX/Flax model...")
        
        try:
            # Load the pickled model and parameters
            with open("/model/final_finetuned_model.pkl", "rb") as f:
                model_data = pickle.load(f)
            
            # Debug: Print structure to understand parameter layout
            logging.info(f"Model data type: {type(model_data)}")
            if isinstance(model_data, dict):
                logging.info(f"Model data keys: {list(model_data.keys())}")
                if 'params' in model_data:
                    logging.info(f"Param structure: {list(model_data['params'].keys())}")
            
            # Extract model components
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.params = model_data.get('params') or model_data.get('state', {}).get('params')
                self.model_config = model_data.get('config', {})
            else:
                # Handle different pickle formats
                self.params = model_data
                logging.info(f"Direct params structure: {list(self.params.keys()) if hasattr(self.params, 'keys') else 'Not a dict'}")
                
            # Import model architecture
            import sys
            sys.path.append('/app')
            from crescendai_model.models.production_ast import ProductionAST, remap_attention_params
            
            # Initialize model if needed  
            if self.model is None:
                model_config = self.model_config or {
                    'patch_size': 16,
                    'embed_dim': 768,
                    'num_layers': 12,
                    'num_heads': 12,
                    'mlp_dim': 3072,
                    'dropout_rate': 0.1,
                    'num_classes': 19
                }
                self.model = ProductionAST(**model_config)
                
            # Remap parameters to match model structure
            if self.params and 'params' in self.params:
                self.params = remap_attention_params(self.params['params'])
            elif hasattr(self.params, 'keys'):
                self.params = remap_attention_params(self.params)
            
            self.is_loaded = True
            load_time = time.time() - start_time
            logging.info(f"Model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def predict(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Run inference on mel-spectrogram input"""
        if not self.is_loaded:
            self.load_model()
            
        try:
            # Ensure proper input shape: [batch, time, freq]
            if mel_spectrogram.ndim == 2:
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
            
            # Convert to JAX array
            inputs = jnp.array(mel_spectrogram)
            
            # Run inference
            outputs = self.model.apply(self.params, inputs, training=False)
            
            # Convert back to numpy and ensure proper shape
            predictions = np.array(outputs)
            if predictions.ndim > 1:
                predictions = predictions.squeeze()
            
            # Ensure 19-dimensional output
            if predictions.shape[-1] != 19:
                raise ValueError(f"Expected 19-dimensional output, got {predictions.shape}")
                
            return predictions
            
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")

# Global model manager instance
model_manager = ModelManager()

@app.function(
    image=modal_image,
    gpu="A10G",
    timeout=300,
    memory=8192,
    scaledown_window=60,  # Cold loading strategy
)
def analyze_piano_performance(
    mel_spectrogram: List[List[float]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main Modal function for piano performance analysis
    
    Args:
        mel_spectrogram: 2D list representing mel-spectrogram [time, freq]
        metadata: Optional metadata about the audio
        
    Returns:
        Dict containing 19-dimensional analysis results
    """
    start_time = time.time()
    
    try:
        # Convert input to numpy array
        mel_spec = np.array(mel_spectrogram, dtype=np.float32)
        
        # Validate input shape
        if mel_spec.ndim != 2:
            raise ValueError(f"Expected 2D mel-spectrogram, got shape {mel_spec.shape}")
        
        if mel_spec.shape[1] != 128:
            raise ValueError(f"Expected 128 mel bands, got {mel_spec.shape[1]}")
        
        logging.info(f"Processing mel-spectrogram: {mel_spec.shape}")
        
        # Run model inference
        predictions = model_manager.predict(mel_spec)
        
        # Prepare response
        dimension_names = [
            "Timing_Stable_Unstable",
            "Articulation_Short_Long",
            "Articulation_Soft_cushioned_Hard_solid", 
            "Dynamic_Sophisticated/mellow_Raw/crude",
            "Dynamic_Little_dynamic_range_Large_dynamic_range",
            "Music_Making_Fast_paced_Slow_paced",
            "Music_Making_Flat_Spacious",
            "Music_Making_Not_sensitive_Very_sensitive",
            "Music_Making_Unimaginative_Imaginative",
            "Music_Making_Inarticulated_Well_articulated",
            "Music_Making_Unexpressive_Very_expressive",
            "Music_Making_Unvaried_Extremely_varied",
            "Music_Making_Technical_Musical",
            "Pedal_Appropriate_Inappropriate",
            "Timbre_Incoherent_Coherent",
            "Overall_Unpleasant_Pleasant",
            "Overall_Boring_Interesting",
            "Overall_Emotionless_Emotional",
            "Overall_Not_impressive_Very_impressive"
        ]
        
        # Create dimension results
        results = {}
        for i, name in enumerate(dimension_names):
            results[name] = float(predictions[i])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        response = {
            "status": "success",
            "results": results,
            "metadata": {
                "model_version": "AST-19D-v1.0",
                "processing_time_seconds": processing_time,
                "input_shape": mel_spec.shape,
                "timestamp": time.time(),
                **(metadata or {})
            }
        }
        
        logging.info(f"Analysis completed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        error_time = time.time() - start_time
        logging.error(f"Analysis failed after {error_time:.2f}s: {e}")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "metadata": {
                "processing_time_seconds": error_time,
                "timestamp": time.time(),
                **(metadata or {})
            }
        }

@app.function(
    image=modal_image,
    timeout=60,
)
def health_check() -> Dict[str, str]:
    """Health check endpoint for service monitoring"""
    return {
        "status": "healthy",
        "service": "crescendai-piano-analysis",
        "version": "1.0.0",
        "timestamp": time.time()
    }

# Local testing function
@app.local_entrypoint()
def main():
    """Local testing entrypoint"""
    import numpy as np
    
    # Create sample mel-spectrogram for testing
    sample_mel = np.random.rand(100, 128).tolist()  # 100 time frames, 128 mel bands
    
    print("🎹 Testing CrescendAI Modal service...")
    
    # Test health check
    health = health_check.remote()
    print(f"Health check: {health}")
    
    # Test analysis
    result = analyze_piano_performance.remote(
        mel_spectrogram=sample_mel,
        metadata={"test": True}
    )
    
    print(f"Analysis result: {result}")

if __name__ == "__main__":
    main()