FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}:/opt/miniconda/envs/venv/lib:/opt/miniconda/envs/venv/lib/python3.11/site-packages/torch/lib
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


WORKDIR /usr/src/

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

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
    ttf-mscorefonts-installer \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p /opt/miniconda && rm ~/miniconda.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda create -y --name venv python=3.11
ENV PATH=/opt/miniconda/envs/venv/bin:$PATH
RUN echo "source activate venv" >> ~/.bashrc && conda init bash

RUN pip install requests==2.28.1 && \
    conda install -y -n venv -c conda-forge nifty=1.2.1 python-elf=0.3.0 && \
    pip install numpy scipy cython numba matplotlib seaborn opencv-python pyyaml rawpy imageio imageio-ffmpeg scikit-image beartype colorama moviepy dacite tabulate svg.py h5py einops plyfile e3nn lpips colorspacious tqdm commentjson && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 && \
    pip install pandas wandb -U scikit-learn timm lightning['extra'] tensorboard pytorch-metric-learning pytorch-ignite pytorch-gradcam pytorch-msssim pytorchvideo torchmetrics torch_optimizer torchsampler torchgeometry torchio piq torchdiffeq hydra-core --upgrade fvcore flax[all] jaxtyping tensorflow nibabel dicom2nifti hdf5storage pydicom "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html gsplat segmentation-models-pytorch ruamel.yaml nerfview viser icecream dahuffman vector-quantize-pytorch datashader holoviews hvplot jupyterlab mahotas && \
    pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "cudf-cu12>=24.8.0a0,<=24.8" "dask-cudf-cu12>=24.8.0a0,<=24.8" "cuml-cu12>=24.8.0a0,<=24.8" "cugraph-cu12>=24.8.0a0,<=24.8" "cuspatial-cu12>=24.8.0a0,<=24.8" "cuproj-cu12>=24.8.0a0,<=24.8" "cuxfilter-cu12>=24.8.0a0,<=24.8" "cucim-cu12>=24.8.0a0,<=24.8" "pylibraft-cu12>=24.8.0a0,<=24.8" "raft-dask-cu12>=24.8.0a0,<=24.8" "cuvs-cu12>=24.8.0a0,<=24.8" "pylibraft-cu12>=24.8.0a0,<=24.8" "dask-cuda>=24.8.0a0,<=24.8"

ENTRYPOINT ["python"]