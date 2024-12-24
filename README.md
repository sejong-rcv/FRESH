# FRESH: Fusion-based 3D apple Recognition via Estimating Stem direction Heading

## Official Pytorch Implementation of [FRESH: Fusion-based 3D apple Recognition via Estimating Stem direction Heading](https://www.mdpi.com/2077-0472/14/12/2161)
#### Authors: [Geonhwa Son](https://sites.google.com/view/geonhwa), [Seunghyeon Lee](https://sites.google.com/view/seunghyeon-lee), [Yukyung Choi](https://scholar.google.com/citations?user=vMrPtrAAAAAJ&hl=ko&oi=sra)

![2024-12-24 142641](https://github.com/user-attachments/assets/f21d43b6-be37-4fc8-bdeb-34cda01dff8e)


## Abstract
 In 3D apple detection, the challenge of direction for apple stem harvesting for agricultural robotics has not yet been resolved. Addressing the issue of determining the stem direction of apples is essential for the harvesting processes employed by automated robots. 
 This research proposes a 3D apple detection framework to identify stem direction. First, we constructed a dataset for 3D apple detection that considers the 3-axis rotation of apples based on stem direction. Secondly, we designed a 3D detection algorithm that not only 
 recognizes the dimensions and location of apples, as existing methods do, but also predicts their 3-axis rotation. Furthermore, we effectively fused 3D point clouds with 2D images to leverage the geometric data from point clouds and the semantic information from 
 images, enhancing the apple detection performance. Experimental results indicated that our method achieved AP@0.25 89.56% for 3D detection by considering apple rotation, surpassing the existing methods. Moreover, we experimentally validated that the proposed loss 
 function most effectively estimated the rotation among the various approaches we explored. This study shows the effectiveness of 3D apple detection with consideration of rotation, emphasizing its potential for practical application in autonomous robotic systems.
 
> **PDF**: [FRESH: Fusion-based 3D apple Recognition via Estimating Stem direction Heading](https://www.mdpi.com/2077-0472/14/12/2161/pdf)

---

## Usage

## Prerequisites

### Recommended Environment
 * OS : Ubuntu 18.04
 * CUDA : 11.3
 * Python 3.7
 * Pytorch 1.12.1 Torchvision 0.13.1
 * GPU : NVIDA A100 (40G)

* Required packages are listed in **environment.yaml**. You can install by running:

```
conda env create -f environment.yaml
conda activate PVLR
```

### Data Preparation
* For convenience, we provide the features we used. You can find them here.
   * Thumos : [Google Drive](https://drive.google.com/file/d/1o8Jx0joiL9fO9Um3T-qu9SAi_fX5MpPV/view?usp=sharing)
   * Annet : [Google Drive](https://drive.google.com/file/d/1M9BWg3Jx17Jf7JsxQs_FvLQ23pxtkja7/view?usp=sharing)
* The **feature** directory should be organized as follows:
~~~~
├── PVLR
   ├── data
      ├── thumos
          ├── Thumos14_CLIP
          ├── Thumos14-Annotations
          ├── Thumos14reduced
          └── Thumos14reduced-Annotations
      ├── annet
          ├── Anet_CLIP
          ├── ActivityNet1.2-Annotations
          └── ActivityNet1.3
~~~~
* Considering the difficulty in achieving perfect reproducibility due to different model initializations depending on the experimental device (e.g., different GPU setup), we provide the initialized model parameters we used.

* Please note that the parameters provided are the initial parameters before any training has been conducted.
   * ckpt (thumos) : [Google Drive](https://drive.google.com/file/d/1iepClS4hohz2uH8Mfgajjr-y9-VFuITG/view?usp=drive_link)
   * ckpt (annet) : [Google Drive](https://drive.google.com/file/d/1TzFzJL4k3odpYYwm9sx2iQN--oaf_x5B/view?usp=drive_link)

* The **checkpoint file**  should be organized as follows:
~~~~
├── PVLR
   ├── data
      ├── ...
      ├── ...
      ├── init_thumos.pth
      └── init_annet.pth
~~~~

## Run

### Training
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py --model-name PVLR
```

### Inference
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python eval/inference.py --pretrained-ckpt output/ckpt/PVLR/Best_model.pkl
```
## References
We referenced the repos below for the code.
* [CLIP](https://github.com/openai/CLIP)
* [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)
* [CoLA](https://github.com/zhang-can/CoLA)




