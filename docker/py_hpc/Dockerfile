FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/miniconda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}:/opt/miniconda/envs/venv/lib:/opt/miniconda/envs/venv/lib/python3.12/site-packages/torch/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    wget \
    bzip2 \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh && \
    /opt/miniconda/bin/conda create -y --name venv python=3.11 && \
    /opt/miniconda/bin/conda clean -afy && \
    echo "source activate venv" >> ~/.bashrc && \
    /opt/miniconda/bin/conda init bash

SHELL ["/bin/bash", "-c"]

RUN source /opt/miniconda/bin/activate venv && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy scipy cython numba matplotlib seaborn opencv-python pyyaml imageio imageio-ffmpeg scikit-image beartype colorama moviepy dacite tabulate svg.py h5py einops plyfile e3nn lpips colorspacious tqdm && \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 && \
    pip install --no-cache-dir pandas wandb -U scikit-learn timm lightning['extra'] tensorboard \
    pytorch-metric-learning pytorch-ignite pytorch-gradcam pytorch-msssim pytorchvideo torchmetrics torch_optimizer torchsampler \
    torchgeometry torchio piq torchdiffeq hydra-core --upgrade fvcore flax[all] jaxtyping tensorflow nibabel dicom2nifti hdf5storage pydicom \
    "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html gsplat segmentation-models-pytorch ruamel.yaml nerfview viser icecream dahuffman vector-quantize-pytorch \
    datashader holoviews hvplot jupyterlab mahotas && \
    pip install --no-cache-dir --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cudf-cu12>=24.8.0a0,<=24.8" "dask-cudf-cu12>=24.8.0a0,<=24.8" "cuml-cu12>=24.8.0a0,<=24.8" "cugraph-cu12>=24.8.0a0,<=24.8" \
    "cuspatial-cu12>=24.8.0a0,<=24.8" "cuproj-cu12>=24.8.0a0,<=24.8" "cuxfilter-cu12>=24.8.0a0,<=24.8" "cucim-cu12>=24.8.0a0,<=24.8" \
    "pylibraft-cu12>=24.8.0a0,<=24.8" "raft-dask-cu12>=24.8.0a0,<=24.8" "cuvs-cu12>=24.8.0a0,<=24.8" "pylibraft-cu12>=24.8.0a0,<=24.8" \
    "dask-cuda>=24.8.0a0,<=24.8"

ENTRYPOINT ["python"]
