#!/usr/bin/env sh
# =============================================================================
# entrypoint.sh — backtester container entry point
#
# Reads backtest_config.yaml, extracts field values with awk, and invokes
# ml_backtest with the resolved arguments.
#
# Usage (called by Docker CMD):
#   /app/entrypoint.sh /app/backtest_config.yaml
# =============================================================================

set -eu

CONFIG="${1:-/app/backtest_config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG" >&2
    echo "        Mount it with: -v ./backtest_config.yaml:/app/backtest_config.yaml:ro" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse a flat YAML value by key.
# Handles:  key: value  and  key: "value"  (strips surrounding quotes).
# ---------------------------------------------------------------------------
yaml_get() {
    awk -F': ' -v key="$1" '
        $1 == key {
            val = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
            gsub(/^["'"'"']|["'"'"']$/, "", val)
            print val
            exit
        }
    ' "$CONFIG"
}

SYMBOL=$(yaml_get "symbol")
FEATURE_CSV=$(yaml_get "feature_csv")
MODEL_PT=$(yaml_get "model_pt")
FEATURE_SCALER=$(yaml_get "feature_scaler_csv")
TARGET_SCALER=$(yaml_get "target_scaler_csv")
OUTPUT_DIR=$(yaml_get "output_dir")

# Validate required fields
for field_name in SYMBOL FEATURE_CSV MODEL_PT FEATURE_SCALER TARGET_SCALER OUTPUT_DIR; do
    eval "field_val=\$$field_name"
    if [ -z "$field_val" ]; then
        echo "[ERROR] Missing required config field: $field_name" >&2
        exit 1
    fi
done

# Validate that input files exist before invoking the binary
for path in "$FEATURE_CSV" "$MODEL_PT" "$FEATURE_SCALER" "$TARGET_SCALER"; do
    if [ ! -f "$path" ]; then
        echo "[ERROR] Required file not found: $path" >&2
        echo "        Has the pipeline container finished running?" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== ML Backtest ==="
echo "  Config:         $CONFIG"
echo "  Symbol:         $SYMBOL"
echo "  Feature CSV:    $FEATURE_CSV"
echo "  Model:          $MODEL_PT"
echo "  Feature scaler: $FEATURE_SCALER"
echo "  Target scaler:  $TARGET_SCALER"
echo "  Output dir:     $OUTPUT_DIR"
echo ""

exec /app/ml_backtest \
    "$FEATURE_CSV" \
    "$SYMBOL" \
    "$MODEL_PT" \
    "$FEATURE_SCALER" \
    "$TARGET_SCALER"
