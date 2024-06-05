FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV CUDA_HOME /usr/local/cuda
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}:/opt/miniconda/envs/venv/lib:/opt/miniconda/envs/venv/lib/python3.11/site-packages/torch/lib
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /usr/src/

RUN apt-get update && apt-get install -y build-essential gcc wget bzip2 curl git ffmpeg libsm6 libxext6 ninja-build && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p /opt/miniconda && rm ~/miniconda.sh

ENV PATH /opt/miniconda/bin:$PATH

RUN conda create -y --name venv python=3.11
ENV PATH /opt/miniconda/envs/venv/bin:$PATH

RUN echo "source activate venv" >> ~/.bashrc && conda init bash

RUN nvcc --version

RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy scipy cython numba matplotlib seaborn opencv-python pyyaml imageio imageio-ffmpeg scikit-image \
                               beartype colorama moviepy dacite tabulate svg.py h5py einops plyfile e3nn lpips colorspacious tqdm

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

RUN pip install pandas wandb -U scikit-learn timm lightning['extra'] tensorboard \
                               pytorch-metric-learning pytorch-ignite pytorch-gradcam pytorch-msssim pytorchvideo torchmetrics torch_optimizer torchsampler \
                               torchensemble torchattacks torchgeometry torchio piq torchdiffeq hydra-core --upgrade fvcore flax[all] jaxtyping tensorflow nibabel dicom2nifti hdf5storage pydicom \
                               "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html gsplat segmentation-models-pytorch cupy ruamel.yaml

COPY ../cu/block_ops.cu /usr/src/
COPY ../cu/build_ops.py /usr/src/
COPY ../cu/build.sh /usr/src/

RUN chmod +x /usr/src/build.sh
RUN /usr/src/build.sh

ENTRYPOINT ["python"]
