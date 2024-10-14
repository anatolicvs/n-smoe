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
    libboost-all-dev \
    ninja-build \
    vim \
    nano \
    tmux \
    zsh \
    gdb \
    strace && \
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

RUN git clone https://gitlab.inria.fr/bkerbl/simple-knn.git /usr/src/submodules/simple-knn && \
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /usr/src/submodules/diff-gaussian-rasterization && \
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive /usr/src/gaussian-splatting

RUN echo "alias ll='ls -la'" >> ~/.bashrc && \
    echo "export PS1='\u@\h:\w\$ '" >> ~/.bashrc && \
    echo "source activate venv" >> ~/.bashrc

RUN git config --global user.name "Aytac Ozkan" && \
    git config --global user.email "aytac@linux.com"


ENTRYPOINT ["/bin/bash"]
