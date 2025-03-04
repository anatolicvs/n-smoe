FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so.1


ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /usr/src/

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
    libxrender1 \
    tk-dev \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p /opt/miniconda && rm ~/miniconda.sh

ENV PATH=/opt/miniconda/bin:$PATH
RUN conda create -y --name venv python=3.11
ENV PATH=/opt/miniconda/envs/venv/bin:$PATH
RUN echo "source activate venv" >> ~/.bashrc && conda init bash

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir numpy scipy cython numba matplotlib seaborn opencv-python pyyaml imageio && \
    pip install --no-cache-dir imageio-ffmpeg scikit-image beartype colorama moviepy dacite tabulate svg.py h5py && \
    pip install --no-cache-dir einops plyfile e3nn lpips colorspacious tqdm thop pandas wandb scikit-learn timm lightning['extra'] tensorflow tensorboard && \
    pip install --no-cache-dir monailabel pytorch-metric-learning pytorch-ignite pytorchvideo torchmetrics pytorch-msssim torch_optimizer && \
    pip install --no-cache-dir torchsampler torchgeometry torchio piq torchdiffeq hydra-core fvcore flax[all] jaxtyping  nibabel && \
    pip install --no-cache-dir dicom2nifti hdf5storage pydicom "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir segmentation-models-pytorch ruamel.yaml nerfview viser icecream dahuffman vector-quantize-pytorch && \
    pip install --no-cache-dir datashader holoviews hvplot jupyterlab mahotas iopath SimpleITK monai connected-components-3d diffusers[torch] commentjson pytorch_msssim albumentations &&\
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
     
    # gsplat removed temporarily

RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && pip install -e .

# RUN pip install --no-cache-dir --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
#     --extra-index-url=https://pypi.nvidia.com \
#     "cudf-cu12==24.8.2" \
#     "dask-cudf-cu12==24.8.2" \
#     "cuml-cu12==24.8.0" \
#     "cugraph-cu12==24.8.0" \
#     "cuspatial-cu12==24.8.0" \
#     "cuproj-cu12==24.8.0" \
#     "cuxfilter-cu12==24.8.0" \
#     "cucim-cu12==24.8.0" \
#     "pylibraft-cu12==24.8.1" \
#     "raft-dask-cu12==24.8.1" \
#     "cuvs-cu12==24.8.0"

ENTRYPOINT ["python"]


