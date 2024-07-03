# Surf-D: Generating High-Quality Surfaces of Arbitrary Topologies Using Diffusion Models (ECCV 2024)

<img src="./assets/fig_virtual_try_on.gif">

**Surf-D: Generating High-Quality Surfaces of Arbitrary Topologies Using Diffusion Models**<br>
**[Project Page](https://yzmblog.github.io/projects/SurfD)**
| **[Paper](https://arxiv.org/abs/2311.17050)**

This is an official implementation of Surf-D using [PyTorch](https://pytorch.org/)

>We present **Surf-D**, a novel method for generating high-quality 3D shapes as **Surfaces** with arbitrary topologies using **Diffusion** models. Previous methods explored shape generation with different representations and they suffer from limited topologies and poor geometry details. To generate high-quality surfaces of arbitrary topologies, we use the Unsigned Distance Field (UDF) as our surface representation to accommodate arbitrary topologies. Furthermore, we propose a new pipeline that employs a point-based AutoEncoder to learn a compact and continuous latent space for accurately encoding UDF and support high-resolution mesh extraction. We further show that our new pipeline significantly outperforms the prior approaches to learning the distance fields, such as the grid-based AutoEncoder, which is not scalable and incapable of learning accurate UDF. In addition, we adopt a curriculum learning strategy to efficiently embed various surfaces. With the pretrained shape latent space, we employ a latent diffusion model to acquire the distribution of various shapes. Extensive experiments are presented on using Surf-D for unconditional generation, category conditional generation, image conditional generation, and text-to-shape tasks. The experiments demonstrate the superior performance of Surf-D in shape generation across multiple modalities as conditions.

## Installation

We recommend to use [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda env create -f environment.yaml
    conda activate SurfD

    cd meshudf
    python3 setup.py build_ext --inplace

## Download pretrained models
Download our pretrained models at [google drive](https://drive.google.com/drive/folders/19Wdbg-zOB48IZ3KSxayRK3q1v1HfWRUP?usp=sharing)

## Generate from Diffusion:
Unconditional generation:

    python -m sample.generate_uncond \
        --model_path pretrained_models/diffusion_uncond.pt \
        --output_dir ./outputs/uncond/ \
        --cond_mode no_cond \
        --ae_dir pretrained_models/ae_deepfashion3d.pt  \
        --num_samples 10 \
        --resolution 512

Sketch conditional generation:

    python -m sample.generate_sketch \
        --model_path pretrained_models/diffusion_sketch.pt \
        --output_dir ./outputs/sketch_cond/ \
        --cond_mode sketch \
        --ae_dir pretrained_models/ae_deepfashion3d.pt \
        --sketch_path demo_images/sketch.png \
        --resolution 512

Image conditional generation:

    python -m sample.generate_image \
        --model_path pretrained_models/diffusion_image.pt \
        --output_dir ./outputs/image_cond/ \
        --cond_mode img \
        --ae_dir pretrained_models/ae_pix3d.pt \
        --image_path demo_images/image.jpg \
        --mask_path demo_images/mask.jpg \
        --resolution 512

Text conditional generation:

    python -m sample.generate_text \
        --model_path pretrained_models/diffusion_text.pt \
        --output_dir ./outputs/text_cond/ \
        --cond_mode text \
        --ae_dir pretrained_models/ae_text.pt  \
        --prompt "a dining chair" \
        --watertight --num_samples 10 \
        --resolution 512

# Training

## Prepare dataset
Down load DeepFashion3D dataset at [DeepFashion3D](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D), Pix3D dataset at [Pix3D](http://pix3d.csail.mit.edu/) and ShapeNet dataset at [ShapeNet](https://shapenet.org/).

## Preprocess the dataset
    cd AutoEncoder/encdc
    python preprocess_udfs.py /path/to/data_root /path/to/output dataset_name

## AutoEncoder Training:
```
cd AutoEncoder/encdc
```

    python train_encdec.py ../cfg/xxx/xxx.yaml


## Diffusion Training:

    python train_diffcloth.py --cond_mode no_cond --save_dir xxx --overwrite --data_dir xxx --ae_dir xxx --log_interval 25 --save_interval 10000 --dataset deepfashion3d

<a name="citation"></a>
## Citation
If you find this work useful for your research, please consider citing our paper: 

```bibtex
@article{yu2023surf,
  title={Surf-D: High-Quality Surface Generation for Arbitrary Topologies using Diffusion Models},
  author={Yu, Zhengming and Dou, Zhiyang and Long, Xiaoxiao and Lin, Cheng and Li, Zekun and Liu, Yuan and M{\"u}ller, Norman and Komura, Taku and Habermann, Marc and Theobalt, Christian and others},
  journal={arXiv preprint arXiv:2311.17050},
  year={2023}
}
```