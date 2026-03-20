#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  GreenSync — Project Launcher
# ─────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colours
GREEN='\033[0;32m'
AMBER='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo -e "${GREEN}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║  ⚡  GreenSync Microgrid Forecaster  ⚡  ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${RESET}"

# ── Python env check ──────────────────────────
if ! command -v python3 &> /dev/null; then
    echo -e "${AMBER}⚠ python3 not found. Please install Python 3.10+${RESET}"
    exit 1
fi

# ── Install deps ──────────────────────────────
echo -e "${CYAN}[1/4] Installing Python dependencies...${RESET}"
pip install -q -r requirements.txt

# ── Run preprocessing pipeline ───────────────
echo -e "${CYAN}[2/4] Running data preprocessing pipeline...${RESET}"
python3 -c "
import sys; sys.path.insert(0, '.')
from utils.preprocessing import run_pipeline
from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH
df = run_pipeline(str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH))
print(f'  ✓ Processed {len(df)} rows → {PROCESSED_DATA_PATH}')
"

# ── Train model ──────────────────────────────
echo -e "${CYAN}[3/4] Training forecasting model...${RESET}"
python3 -c "
import sys; sys.path.insert(0, '.')
from models.train_model import train
metrics = train()
print(f'  ✓ Training complete | MAPE: {metrics[\"overall_mape\"]}%')
"

# ── Launch services ──────────────────────────
echo -e "${CYAN}[4/4] Launching GreenSync services...${RESET}"
echo ""
echo -e "  📊 Dashboard  → ${GREEN}http://localhost:8501${RESET}"
echo -e "  🔌 REST API   → ${GREEN}http://localhost:8000/docs${RESET}"
echo ""

# Start FastAPI in background
PYTHONPATH="$PROJECT_DIR" uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo -e "  API PID: $API_PID"

sleep 2

# Start Streamlit dashboard (foreground)
PYTHONPATH="$PROJECT_DIR" streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --theme.backgroundColor "#0A0F0A" \
    --theme.primaryColor "#00E676" \
    --theme.textColor "#E8F5E9"

# Cleanup on exit
kill $API_PID 2>/dev/null || true
