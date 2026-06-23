#!/usr/bin/env bash
# Download AutoWeather4D runtime assets (Wan, DiffusionRenderer, sample waymo.h5).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "Wan checkpoints"
echo "============================================================"
(cd 3rd/VideoX-Fun && python scripts/download_wan_checkpoints.py)

echo ""
echo "============================================================"
echo "DiffusionRenderer checkpoints"
echo "============================================================"
(cd 3rd/cosmos-transfer1-diffusion-renderer && python scripts/download_diffusion_renderer_checkpoints.py)

echo ""
echo "============================================================"
echo "Sample scene waymo.h5"
echo "============================================================"
python data/download_waymo_h5.py

echo ""
echo "============================================================"
echo "All assets downloaded."
echo "============================================================"
