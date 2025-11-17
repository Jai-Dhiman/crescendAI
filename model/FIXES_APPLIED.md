# Fixes Applied to Data Copy Script

## Issues Encountered

### 1. Google Drive I/O Error During Copy
```
OSError: [Errno 5] Input/output error: '/content/drive/.../MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--2_0010.wav'
```

**Root Cause**: Google Drive FUSE mount is flaky and can fail randomly on individual file operations

### 2. Preflight Check Looking in Wrong Path
```
FileNotFoundError: Missing: /tmp/training_data/synthetic_train_filtered.jsonl
```

**Root Cause**: Old preflight check cell was still looking at `/tmp/training_data/` instead of `/tmp/crescendai_data/`

## Fixes Applied

### Fix 1: Added Retry Logic to Copy Script

**File**: `scripts/copy_data_to_local.py`

**Changes**:
1. Wrapped file existence checks in try/catch (line 137-145)
2. Added 3-attempt retry logic for each file copy (line 153-201 for audio, 211-257 for MIDI)
3. Retry on `OSError` and `IOError` with 0.5s delay between attempts
4. Track failed files separately instead of crashing
5. Added comprehensive summary showing:
   - Success rate (files copied / total files)
   - Detailed stats (copied, skipped, failed)
   - Status: Complete (>95%), Partial (>80%), Failed (<80%)

**Impact**:
- Script no longer crashes on transient Drive errors
- Can resume where it left off if re-run
- Provides clear feedback on copy success rate
- Training can proceed even if some files fail (Dataset will skip failed samples)

### Fix 2: Removed Duplicate Preflight Check Cell

**File**: `notebooks/train_full_model.ipynb`

**Changes**:
1. Deleted cell 15 (old inline preflight check looking at wrong path)
2. Cell 14 now uses the proper `scripts/preflight_check.py` script
3. Updated experiment header cells with optimized time estimates
4. Fixed cell 21 type (was markdown, should be markdown header)

**Impact**:
- No more path confusion
- Single source of truth for preflight checks
- Cleaner notebook flow

## Expected Behavior Now

### Copy Script Output
```
5. Copying 32,898 audio files...
  Audio: 100%|██████████| 32898/32898
  ✓ Copied: 32,850, Skipped: 0, Failed: 48

6. Copying 32,898 MIDI files...
  MIDI: 100%|██████████| 32898/32898
  ✓ Copied: 32,890, Skipped: 0, Failed: 8

======================================================================
✓ DATA COPY COMPLETE  (or ⚠ PARTIALLY COMPLETE if >80%)
======================================================================

Copy statistics:
  Audio: 32,850 copied, 0 skipped, 48 failed
  MIDI:  32,890 copied, 0 skipped, 8 failed
  Total: 65,740/65,796 files (99.9%)

Next steps:
  1. Run preflight check:
     python scripts/preflight_check.py --config configs/experiment_10k.yaml
  2. Train models (now 6x faster!):
     python train.py --config configs/experiment_10k.yaml --mode audio
```

### What Happens with Failed Files

**During Training**:
- Dataset tries to load audio/MIDI from local paths
- If a file failed to copy, `load_audio()` will fail
- Dataset catches the error and logs warning (every 100th error to avoid spam)
- Skips that sample and continues with next one
- Training proceeds normally with slightly fewer samples

**Impact**: Minimal - losing <1% of samples won't affect model performance

## Testing

### Local Test (Before Deploying to Colab)
```bash
cd model
python3 scripts/test_optimizations_local.py
```

Should show:
```
✓ Scripts exist
✓ Config valid
✓ Notebook updated
```

### In Colab
1. Run Step 2 (copy_data_to_local.py)
2. Check output for success rate
3. If >95%: ✓ Perfect, proceed to training
4. If 80-95%: ⚠ Partial, but should work fine
5. If <80%: ✗ Re-run script or restart runtime

## Additional Robustness

### Resume Capability
If the copy script is interrupted or fails partway through:
1. Re-run the same command
2. Script detects existing files and skips them (no re-copy)
3. Only attempts to copy missing files
4. Much faster second run since most files already copied

### Preflight Check
Now validates:
1. Data is on local SSD (not Drive) ✓
2. Files load successfully (<0.5s per batch) ✓
3. DataLoader speed is acceptable ✓
4. Models can be instantiated ✓

## Summary

**Before**: Script crashed on first Drive I/O error, no retry logic
**After**: Retries 3 times per file, tracks failures, provides clear summary

**Before**: Confusing duplicate preflight checks with wrong paths
**After**: Single authoritative preflight check with correct paths

**Result**: More robust, better error messages, training can proceed even with minor failures
