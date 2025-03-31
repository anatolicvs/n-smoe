# Neural Steered Mixture of Experts for Medical Image Denoising, and Super-resolution (Submitted to Pattern Recognition, 2025) 
[Aytaç Özkan](https://www.linkedin.com/in/aytacozkan/), [Thomas Sikora](https://scholar.google.com/citations?user=2kr3tg0AAAAJ&hl=en)

[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5193694#paper-references-widget) 

:star: If our work is helpful to your research, please help star this repo. Thanks! :hugs: 

---
In medical imaging (MI) analysis, achieving high-fidelity spatial resolution remains challenging due to extended acquisition durations and limited frame rates, constraining the quality and diagnostic value of clinical data. 
Super-resolution (SR) methodologies reconstruct high-resolution (HR) representations from low-resolution (LR) input, mitigating hardware and temporal constraints while enhancing interpretability and diagnostic reliability. 
Current single-image super-resolution (SISR) paradigms often rely on oversimplified noise assumptions, modeled as Additive White Gaussian Noise (AWGN), which fail to capture the complex and modality-dependent noise distributions inherent to clinical imaging scenarios. 
These limitations require more sophisticated SR frameworks capable of accurately representing non-stationary degradations and ensuring robust performance across imaging modalities. 
We introduce a neural parametric Steered Mixture of Experts (N-SMoE) framework that leverages a generative adversarial-based training paradigm and a Stochastic Degradation Model (SDM), applying diverse perturbations to downsampled inputs to approximate clinical conditions. This framework combines a novel encoder network with an implicit probabilistic SMoE decoder. 
The encoder utilizes a Laplacian resizer with bandpass filtering to capture local spatial information and employs multi-head attention to preserve high-frequency (HF) structural patterns while estimating latent representations of the input image. The imsplicit probabilistic gating mechanisms of the SMoE decoder, using two-dimensional edge-aware kernels, represent the signal of interest with continuous transitions, making this autoregressive approach more robust and effective for SR and denoising. 
The proposed N-SMoE framework not only provides interpretability for the learned representations but also demonstrates state-of-the-art (SOTA) performance in restoration tasks across multiple medical imaging datasets, achieving improved fidelity and perceptual metrics. 
><img src="https://cvws.icloud-content.com/B/ASl90x_qG3llUNno3mPIMjzarS5dAcZfJsw6hfclA2EyTjtElwvEw1XM/N-SMoE.drawio.png?o=Ap0bRpcgn5yq_uIRXvU7p3e-zFdToe1Msc85sd-gAWbb&v=1&x=3&a=CAog4r35xLyD5qXcjMkWc7Rf1YUQbzVKhRTIGFcolOGYPv8SbxDT2bjy3jIY07aU9N4yIgEAUgTarS5dWgTEw1XMaidBAAr8Wip4MgZdFJs_E6mhGuUE7VApdlKIHVkpVI4pz4UEp-T3ImVyJyeyRNu_-cpTD_yByYaXxmVD7T8bFJcDqYi54d0Pwzx6OOJnJcRVRw&e=1743463455&fl=&r=107f7a65-6d70-4724-9bcc-b4fc71847aaa-1&k=Dsqwvj3HIH7cDWgXTAHi8Q&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=127&s=H9kCwkjMqo0h9ZA2Ie6s2xjbym4&cd=i" align="middle" width="800">
---

## Requirements
* Python 3.8, Pytorch 1.13.0
* More detail (See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `virnet` can be created and activated with:

```
conda create -n virnet python=3.8 -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
conda activate virnet
```

## :snowflake: Testing Demo
Befor testing, please first download the checkpoint from this [link](https://github.com/zsyOAOA/VIRNet/releases/tag/v1.0) and put them in the foloder "model_zoo".
1. General testing demo.
```
    python scripts/testing_demo.py --task task_name --in_path: input_path --out_path: output_path --sf sr_scale
```
+ --task: task name, "denoising-syn", "denoising-real", "sisr"
+ --in_path: input path of the low-quality images, image path or folder  
+ --out_path: output folder
+ --sf: scale factor for image super-resolution, 2, 3, or 4

2. Reproduce the results in Table 1 of our paper.
```
    python scripts/denoising_virnet_syn.py --save_dir output_path --noise_type niid
```
3. Reproduce the results in Table 2 of our paper.
```
    python scripts/denoising_virnet_syn.py --save_dir output_path --noise_type iid
```
4. Reproduce the results on SIDD dataset in Table 4 of our paper.
```
    python scripts/denoising_virnet_real_sidd.py --save_dir output_path --sidd_dir sidd_data_path
```
5. Reproduce the results on DND dataset in Table 4 of our paper.
```
    python scripts/denoising_virnet_real_dnd.py --save_dir output_path --dnd_dir dnd_data_path
```
6. Reproduce the results of super-resolution in Table 5 of our paper.
```
    python scripts/sisr_virnet_syn.py --save_dir output_path --sf 4 --nlevel 0.1
```
+ --nlevel: noise level, 0.1, 2.55 or 7.65

## :sunny: Training Pipeline:
### :pear: Image Denoising on Synthetic Data:
1. Download the source images from [Waterloo](https://kedema.org/project/exploration/index.html), [CBSD432](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing), [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) and [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as groundtruth and fill the data path in the [config](configs/denoising_syn.json) file.

2. Begin to train: 

```
    CUDA_VISIBLE_DEVICES=gpu_id python train_denoising_syn.py --save_dir path_for_log
```

### :apple: Image Denoising on Real-world Data:
1. Download the training datasets [SIDD](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip) and validation datasets [noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat).

2. Crop the training datasets into small image patches using this [script](datasets/prepare_data/Denoising/SIDD/im2patch_train.py), and fill the data path in the [config](configs/denoising_real.json) file.

3. Begin to training:
```
    CUDA_VISIBLE_DEVICES=gpu_id python train_denoising_real.py --save_dir path_for_log
```

### :peach: Image Super-resolution
1. Download the high-resolution images of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and crop them into small image patches using this [script](datasets/prepare_data/SISR/im2patch_train.py).

2. Fill data path in the [config](configs/sisr_x4.json) file.

3. Begin to train:

```
    CUDA_VISIBLE_DEVICES=gpu_id python train_SISR.py --save_dir path_for_log --config configs/sisr_x4.json
```

### :dolphin: Citation
```
    @article{yue2024variational,
      title={Deep Variational Network Toward Blind Image Restoration},
      author={Yue, Zongsheng and Yong, Hongwei and Zhao, Qian and Zhang, Lei and Meng, Deyu and Wong, Kwan-Yee K},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2024}
    }
```

### :envelope: Contact
If you have any questions, please feel free to contact me via `zsyzam@gmail.com`.

