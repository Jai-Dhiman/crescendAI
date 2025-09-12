# CrescendAI Model Deployment Guide

## Overview

This guide covers deploying the CrescendAI piano performance analysis model using Modal for serverless GPU inference.

## Architecture

```
[iOS/Web Apps] → [Cloudflare Workers] → [Modal Service] → [JAX/Flax AST Model]
     ↓                    ↓                   ↓                    ↓
[Audio Upload] → [Basic Preprocessing] → [GPU Inference] → [19D Analysis]
```

## Quick Start

### 1. Setup Modal Account

```bash
# Install Modal CLI (already done with uv add modal)
modal token new  # Follow authentication flow
```

### 2. Deploy Service

```bash
# Option A: Use deployment script
python3 deploy.py

# Option B: Direct deployment
modal deploy src/modal_service.py
```

### 3. Test Deployment

```bash
# Test locally first
modal run src/modal_service.py

# Check service status
modal app list
```

## Files Created

### Core Service Files

- **`src/modal_service.py`** - Main Modal deployment with JAX/Flax model loading
- **`src/preprocessing_helpers.py`** - Audio preprocessing for Cloudflare Workers integration
- **`src/api_contracts.py`** - Pydantic models for API contracts
- **`deploy.py`** - Deployment management script

### Key Features

- ✅ **JAX Native**: Keeps model in original JAX/Flax format
- ✅ **Cold Loading**: Cost-optimized, loads model per request
- ✅ **A10G GPU**: Optimal price/performance for 86M param model
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **API Contracts**: Type-safe integration with existing backend

## Integration with Existing Backend

### Cloudflare Workers Integration

Your Rust backend needs to:

1. **Route audio uploads** to preprocessing function
2. **Call Modal API** with preprocessed mel-spectrograms  
3. **Return results** to iOS/web clients

### Example Rust Integration

```rust
// In your Cloudflare Worker
async fn analyze_piano_performance(audio_bytes: Vec<u8>) -> Result<AnalysisResponse> {
    // 1. Basic preprocessing (convert to mel-spectrogram)
    let mel_spec = preprocess_audio(audio_bytes).await?;
    
    // 2. Call Modal service
    let modal_response = call_modal_service(mel_spec).await?;
    
    // 3. Return to client
    Ok(modal_response)
}
```

## Cost Estimates

### Cold Loading Strategy

- **Per Request**: ~$0.15-0.30 (3-8s including cold start)
- **Idle Cost**: $0 (no baseline charges)
- **Monthly (1K requests)**: ~$220
- **Monthly (10K requests)**: ~$1,950

### Performance

- **Model Loading**: 2-5s (cold start)
- **Inference**: 1-3s (86M parameters)
- **Total**: <8s end-to-end

## Monitoring & Debugging

### Check Service Health

```bash
# View logs
modal logs crescendai-piano-analysis

# Check function status  
modal function list

# Monitor usage
modal usage
```

### Test Individual Components

```bash
# Test preprocessing
python3 -c "
from src.preprocessing_helpers import create_preprocessing_pipeline
processor = create_preprocessing_pipeline()
print('Preprocessing ready')
"

# Test API contracts
python3 src/api_contracts.py
```

## Optimization Opportunities

### Near-term (for cost reduction)

1. **ONNX Conversion**: Convert JAX model to ONNX for faster loading
2. **Warm Containers**: Keep containers warm during peak hours
3. **Batch Processing**: Process multiple requests together

### Medium-term

1. **Edge Preprocessing**: Move more preprocessing to Cloudflare Workers
2. **Model Compression**: Quantization and pruning
3. **Smart Routing**: Route to different instance types based on load

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check if `results/final_finetuned_model.pkl` exists (327MB)
   - Verify JAX/Flax dependencies in Modal image

2. **Authentication Errors**
   - Run `modal token new` to re-authenticate
   - Check Modal account quota/billing

3. **Timeout Errors**  
   - Model loading can take 5-15s on first cold start
   - Increase timeout in client calls

4. **Memory Errors**
   - 86M parameter model needs 8GB+ memory
   - Increase container memory if needed

### Debug Commands

```bash
# Check model file
ls -la results/final_finetuned_model.pkl

# Test local model loading
python3 -c "
import pickle
with open('results/final_finetuned_model.pkl', 'rb') as f:
    data = pickle.load(f)
print('Model loaded successfully')
"

# Validate Modal setup
modal profile current
modal app list
```

## Next Steps

### Integration Tasks

1. **Update Rust Backend**: Add preprocessing and Modal API calls
2. **Update iOS App**: Point to new analysis endpoints  
3. **Update Web App**: Handle new response format
4. **Add Monitoring**: Set up logging and metrics

### Production Checklist

- [ ] Modal account with adequate quotas
- [ ] Error monitoring and alerting
- [ ] Cost monitoring and budgets
- [ ] Load testing with realistic traffic
- [ ] Backup/fallback strategies

## Support

- Modal documentation: <https://modal.com/docs>
- CrescendAI architecture: See `/docs` directory
- Model training: See `src/train_ast.py`

---

**Ready for deployment!** The Modal service is complete and ready to integrate with your existing Cloudflare Workers backend.
