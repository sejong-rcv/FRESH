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

### Installation 
1. Install mmcv-full (v1.6.0)
   
   - We used version 1.6.0, but you should check the [documentation](https://mmcv.readthedocs.io/en/v1.7.0/get_started/installation.html) and install it according to the version of CUDA and torch you use.
     
       ```
       pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
       ```

2. Install mmdet (v0.24.0), mmsegmentation (v0.24.1)
   ```
   pip install mmdet==0.24.0
   pip install mmsegmentation==0.24.1
   ```

3. Clone the FRESHNet repository.
   ```
   git clone https://github.com/sejong-rcv/FRESH.git
   cd mmdetection3d
   pip install -v -e .
   ```

4. Install MinkowskiEngine
   - Users will also need to install Minkowski as the sparse convolution backend. If necessary please Follow the original [installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation):
     
     ```
     apt install build-essential python3-dev libopenblas-dev
     ```
  
     ```
     python3 setup.py install --user --force_cuda --blas=openblas
     ```
   
### Data Preparation
* For convenience, we provide the dataset we reconstructed. You can find them here.
   * [Google Drive](https://drive.google.com/file/d/1ZkcB5bkoV3gpScAgrumJMwrX8zZdPjs4/view?usp=drive_link)
     
~~~~
├── FRESH
   ├── data
      ├── papple_trainval
          ├── calib
          ├── depth
          ├── image
          └── label
~~~~
* Generate dataset info files by running:
  ```
  python tools/create_data.py papple --root-path data/Papple --out-dir data/Papple --extra-tag papple
  ```  

* The directory structure after processing should be as follows.

~~~~
├── FRESH
   ├── data
      ├── papple_trainval
          ├── calib
          ├── depth
          ├── image
          └── label
      ├── points
      ├── papple_infos_train.pkl
      ├── papple_infos_val.pkl
      └── papple_infos_test.pkl
~~~~

* Please note that the parameters provided are the initial parameters before any training has been conducted.
   * ckpt : [Google Drive](https://drive.google.com/file/d/16D612c1CR_NjUE9ZRaYfVkmsXcJ7JBr9/view?usp=drive_link)

* The **checkpoint file**  should be organized as follows:
~~~~
├──  FRESHNet
   ├── ckpt
      ├── log
      ├── log (json)
      ├── config.py
      ├── epoch_10.pth
      ├── epoch_11.pth
      ├── epoch_12.pth
      ├── epoch_13.pth
      ├── epoch_14.pth
      └── epoch_15.pth
~~~~

## Run

### Training
```
bash tools/dist_train.sh configs/freshnet/fresh-ff_stem_direction.py 4 --work-dir work_dirs/
```

### Inference
```
bash tools/dist_test.sh configs/freshnet/fresh-ff_stem_direction.py work_dirs/ 4 --eval mAP
```
## References
We referenced the repos below for the code.
* [TR3D](https://github.com/SamsungLabs/tr3d.git)
* [3D-Metrics](https://github.com/M-G-A/3D-Metrics.git)



