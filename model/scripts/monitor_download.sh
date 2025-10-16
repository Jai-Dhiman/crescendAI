#!/bin/bash
# Monitor MAESTRO download progress

ZIP_FILE="data/maestro-v3.0.0.zip"
TARGET_SIZE_GB=108
LOG_FILE="data/download.log"

echo "=================================================="
echo "MAESTRO Download Monitor"
echo "=================================================="
echo ""

if [ ! -f "$ZIP_FILE" ]; then
    echo "Download not started yet"
    echo ""
    echo "To start download:"
    echo "  curl -C - -o $ZIP_FILE https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    exit 0
fi

# Get current size
CURRENT_SIZE=$(ls -l "$ZIP_FILE" | awk '{print $5}')
CURRENT_SIZE_GB=$(echo "scale=2; $CURRENT_SIZE / 1024 / 1024 / 1024" | bc)
PERCENT=$(echo "scale=1; ($CURRENT_SIZE_GB / $TARGET_SIZE_GB) * 100" | bc)

echo "Progress: $CURRENT_SIZE_GB GB / $TARGET_SIZE_GB GB ($PERCENT%)"
echo ""

# Show download stats from log
if [ -f "$LOG_FILE" ]; then
    echo "Latest from download log:"
    tail -3 "$LOG_FILE" | grep -v "^$"
    echo ""
fi

# Check if download is still running
if ps aux | grep -v grep | grep -q "curl.*maestro.*zip"; then
    echo "Status: Download in progress"
    
    # Estimate time remaining
    if [ -f "$LOG_FILE" ]; then
        SPEED=$(tail -10 "$LOG_FILE" | grep -oE '[0-9]+\.?[0-9]*M/s' | tail -1 | sed 's/M\/s//')
        if [ ! -z "$SPEED" ]; then
            REMAINING_GB=$(echo "$TARGET_SIZE_GB - $CURRENT_SIZE_GB" | bc)
            TIME_REMAINING_MIN=$(echo "scale=0; ($REMAINING_GB * 1024) / $SPEED / 60" | bc)
            echo "Estimated time remaining: ~$TIME_REMAINING_MIN minutes"
        fi
    fi
else
    if [ $(echo "$CURRENT_SIZE_GB >= 107" | bc) -eq 1 ]; then
        echo "Status: Download complete!"
        echo ""
        echo "Next step: Process test batch"
        echo "  python3 scripts/batch_manager.py --process --limit 5"
    else
        echo "Status: Download stopped/failed"
        echo ""
        echo "To resume download:"
        echo "  curl -C - -o $ZIP_FILE https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    fi
fi

echo ""
echo "=================================================="
echo ""
echo "To check progress continuously:"
echo "  watch -n 5 bash scripts/monitor_download.sh"
