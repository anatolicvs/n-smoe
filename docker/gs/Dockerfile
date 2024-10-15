FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}:/opt/miniconda/envs/venv/lib:/opt/miniconda/envs/venv/lib/python3.11/site-packages/torch/lib
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /usr/src/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-10 \
    wget \
    curl \
    git \
    cmake \
    libglfw3-dev \
    libglew-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    imagemagick \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda create -y --name venv python=3.11
ENV PATH=/opt/miniconda/envs/venv/bin:$PATH
RUN echo "source activate venv" >> ~/.bashrc && conda init bash

RUN conda install -n venv -y -c pytorch -c conda-forge -c defaults \
    cudatoolkit=11.8 \
    plyfile \
    pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN conda install -n venv -y jupyterlab

RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive /usr/src/gaussian-splatting && \
    git clone https://github.com/colmap/colmap.git /usr/src/colmap
    # cd /usr/src/colmap && mkdir build && cd build && \
    # cmake .. -GNinja && ninja && ninja install

RUN pip install --no-cache-dir numpy scipy cython numba matplotlib seaborn opencv-python pyyaml imageio imageio-ffmpeg scikit-image beartype colorama moviepy dacite tabulate svg.py h5py einops lpips colorspacious tqdm pandas wandb scikit-learn timm "lightning[extra]" tensorboard monailabel pytorch-metric-learning pytorch-ignite pytorchvideo torchmetrics torch_optimizer torchsampler torchgeometry torchio piq torchdiffeq hydra-core fvcore "flax[all]" jaxtyping tensorflow nibabel dicom2nifti hdf5storage pydicom "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html segmentation-models-pytorch ruamel.yaml nerfview viser icecream vector-quantize-pytorch datashader holoviews hvplot jupyterlab mahotas iopath SimpleITK monai connected-components-3d "diffusers[torch]"

RUN echo "alias ll='ls -la'" >> ~/.bashrc && \
    echo "export PS1='\u@\h:\w\$ '" >> ~/.bashrc && \
    echo "source activate venv" >> ~/.bashrc

RUN git config --global user.name "Aytac Ozkan" && \
    git config --global user.email "aytac@linux.com"

ENTRYPOINT ["/bin/bash"]