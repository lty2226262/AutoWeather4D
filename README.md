<h1 align="center">🌤️ <i>AutoWeather4D</i>: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing</h1>

<p align="center">
  <a href="https://lty2226262.github.io/autoweather4d/"><img src="https://img.shields.io/badge/Project%20Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.26546"><img src="https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper"></a>
</p>

<div align="center">
  <video src="https://github-production-user-asset-6210df.s3.amazonaws.com/11869313/570491567-1124afd1-06e8-4f98-878c-72d808cccf3e.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260327%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260327T153206Z&X-Amz-Expires=300&X-Amz-Signature=a40337b5dc53c07be0b3e3a51c29a12f705ab1a287c69c1d964404c32263038d&X-Amz-SignedHeaders=host" width="100%" controls></video>
  <p><i>AutoWeather4D enables fine-grained control over weather from real-world driving videos.</i></p>
</div>

**Scene `.h5` in, weather `.mp4` out.**

## 📣 Updates

* **[TODO]** H5 preprocessing instructions and tooling

## 🛠️ Install

```bash
conda env create --file auto-weather-4d.yaml
conda activate autoweather4d
pip install -r requirements.txt

pip install nvidia-cudnn-cu12 nvidia-cuda-nvcc-cu12
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

CUDA_HOME=$CONDA_PREFIX pip install transformer-engine[pytorch]==1.12.0

# Download checkpoints and sample scene
bash scripts/download_assets.sh
# or only the sample scene:
# python data/download_waymo_h5.py
```

## 🚀 Run

```bash
python run.py \
  --input data/waymo.h5 \
  --output output/rain/rain.mp4 \
  --weather rain
```

`--weather`: `rain` | `snow` | `fog` | `night`

Run all four for one scene:

```bash
for w in rain snow fog night; do
  python run.py \
    --input data/waymo.h5 \
    --output output/$w/$w.mp4 \
    --weather $w
done
```

### 🔄 Pipeline

| Weather | Stages |
|---------|--------|
| rain, snow | G-buffer → DiffusionRenderer forward → VidRefiner (Wan SDEdit) |
| fog, night | BRDF relit blend → VidRefiner (Canny edge + Wan SDEdit) |

### 📁 Output & intermediates

By default (`keep_intermediates: false` in `configs/default.yaml`) only the file passed to `--output` is kept:

```text
output/
├── rain/rain.mp4
├── snow/snow.mp4
├── fog/fog.mp4
└── night/night.mp4
```

G-buffer, edge videos, and blended intermediates are removed after a successful run.  
To debug, set `keep_intermediates: true` in config or pass `--keep-intermediates`.

## 📂 Layout

```text
run.py              CLI entry
rendering/          render pipeline, vidrefiner, cleanup
configs/            default.yaml (+ optional overrides)
3rd/                DiffusionRenderer, VideoX-Fun
data/               inputs
output/             rendered mp4s
```

## 📜 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{liu2026autoweather4d,
  title={AutoWeather4D: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing},
  author={Liu, Tianyu and Xiong, Weitao and Zhang, Manyuan and Li, Peng and Luo, Kunming and Liu, Yuan and Tan, Ping},
  journal={arXiv preprint},
  year={2026}
}
```
