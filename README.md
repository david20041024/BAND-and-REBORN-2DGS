# BAND-2DGS and REBORN-2DGS
We use Colab T4 GPU to conduct our methods.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/david20041024/BAND-and-REBORN-2DGS/blob/main/band.ipynb)

## Tools

- **SuperSplat** – High-performance WebGPU viewer by PlayCanvas https://github.com/playcanvas/supersplat
- **MeshLab** -  The open source system for processing and editing 3D triangular meshes https://www.meshlab.net/

## Installation

```bash
# download
git clone https://github.com/david20041024/BAND-and-REBORN-2DGS.git --recursive

# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate surfel_splatting
```

## DTU training, rendering, and evaluating
### To train the original model, we use the following command:
```bash
python scripts/dtu_eval.py --dtu <path to the preprocessed DTU dataset>  \
     --DTU_Official <path to the official DTU dataset>
```
### Preprocess

### To apply our proposed BAND-2DGS framework, we resume training from a pretrained checkpoint and provide additional steps:
```bash
python boundary.py -s <path to dtu>/<scanN> \
      --start_checkpoint output/<scanN>/chkpnt30000.pth \
      --xyz_mesh <path to P> \
      --xyz_nonboundary <path to P'>
```
### To evaluate our reconstructed meshes, we replace the original \texttt{point\_cloud.ply} in the existing directory with our generated results:
```bash
python scripts/dtu_eval.py --dtu <path to the preprocessed DTU dataset>  \
     --DTU_Official <path to the official DTU dataset> --skip_training
```
