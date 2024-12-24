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

## Recommended Environment
- OS: Ubuntu 18.04
- CUDA-cuDNN: 11.3.0-8
- GPU: NVIDIA-A100 (40G)
- Python-Torch: 3.7-1.12.1
  
See [environment.yaml](https://github.com/sejong-rcv/INSANet/blob/main/environment.yaml) for more details

## Installation
The environment file has all the dependencies that are needed for INSANet.

We offer guides on how to install dependencies via docker and conda.

First, clone the repository:
### Git Clone
```
git clone https://github.com/sejong-rcv/INSANet.git
cd INSANet
```

### 1. Docker
- **Prerequisite**
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  - Note that nvidia-cuda:11.3.0 is deprecated. See [issue](https://github.com/NVIDIA/nvidia-docker/issues/1745).

 - **Build Image & Make Container**
 ```
cd docker
make docker-make
```

- **Run Container**
 ```
cd ..
nvidia-docker run -it --name insanet -v $PWD:/workspace -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all --shm-size=32G insanet:maintainer /bin/bash
```

### 2. Conda
- **Prerequisite**
  - Required dependencies are listed in environment.yaml.
```
conda env create -f environment.yml
conda activate insanet
```

If your environment support CUDA 11.3,
```
conda env create -f environment_cu113.yml
conda activate insanet
```

---

## Dataset
The datasets used to train and evaluate model are as follows:
- [KAIST Multispectral Pedestrian Detection Benchmark](https://github.com/SoonminHwang/rgbt-ped-detection)

The dataloader in [datasets.py](https://github.com/sejong-rcv/INSANet/blob/main/src/datasets.py) assumes that the dataset is located in the data folder and structured as follows:

### KAIST
- First, you should download the dataset. we provide the script to download the dataset (please see data/download_kaist).
- Train: We use paired annotations provided in [AR-CNN](https://github.com/luzhang16/AR-CNN).
- Evaluation:  We use sanitized (improved) annotations provided in [MSDS-RCNN](https://github.com/Li-Chengyang/MSDS-RCNN).
```
├── data
   └── kaist-rgbt
      ├── annotations_paired
         ├── set00
            ├── V000
               ├── lwir
                  ├── I00000.txt
                  ├── ...
               ├── visible
                  ├── I00000.txt
                  ├── ...
            ├── V001
               ├── lwir
                  ├── I00000.txt
                  ├── ...
               ├── visible
                  ├── I00000.txt
                  ├── ...
            └── ...
         ├── ... (set02-set10)
         └── set11
            ├── V000
               ├── lwir
                  ├── I00019.txt
                  ├── ...
               ├── visible
                  ├── I00019.txt
                  ├── ...
      ├── images
         ├─ The structure is identical to the "annotations_paired" folder:
         └─ A pair of images has its own train annotations with the same file name.

├── src
   ├── kaist_annotations_test20.json
   ├── imageSets
      ├── train-all-02.txt # List of file names for train.
      └── test-all-20.txt 
```

---

## Demo
Our pre-trained model on the KAIST dataset can be downloaded from [pretrained/download_pretrained.py](https://github.com/sejong-rcv/INSANet/blob/main/pretrained/download_pretrained.py) or [google drive](https://drive.google.com/file/d/1C56Jq1K2TuXFAp9f5UDkSF7Y-FucAG0L/view).

You can infer and evaluate a pre-trained model on the KAIST dataset as follows the below.
```
python pretrained/download_pretrained.py
sh src/script/inference.sh
```

---

## Train & Inference
All train and inference scripts can be found in [src/script/train_eval.sh](https://github.com/sejong-rcv/INSANet/blob/main/src/script/train_eval.sh) and [src/script/inference.sh](https://github.com/sejong-rcv/INSANet/blob/main/src/script/inference.sh).

We provide a per-epoch evaluation in the training phase for convenience.
However, you might see OOM in the early epoch so the per-epoch evaluation is proceed after 10 epochs.

```
cd src/script
sh train_eval.sh
```

If you want to identify the number of (multiple) GPUs and THREADs, add 'CUDA_VISIBLE_DEVICES' and 'OMP_NUM_THREADS'(optional).
```
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python src/train_eval.py
```

---

## Evaluation
If you only want to evaluate, please see the scripts in [src/utils/evaluation_script.sh](https://github.com/sejong-rcv/INSANet/blob/main/src/utils/evaluation_script.sh) and [src/utils/evaluation_scene.sh](https://github.com/sejong-rcv/INSANet/blob/main/src/utils/evaluation_scene.sh).

As mentioned in the paper, we evaluate the performance for standard evaluation protocol (_All, Day, and Night_) on the KAIST dataset, as well as the performance by region (_Campus, Road, Downtown_), and provide them as evaluation_script.sh and evaluation_scene.sh, respectively.

If you want to evaluate a experiment result files (.txt format), follow:
```
cd src/utils
python evaluation_script.py \
    --annFile '../kaist_annotations_test20.json' \
    --rstFiles '../exps/INSANet/Epoch010_test_det.txt'
```
Note that this step is primarily used to evaluate performance per training epochs (result files saved in src/exps).

If you don't want to bother writing down the names of all those files, follow:
```
cd src/utils
python evaluation_script.py \
    --annFile '../kaist_annotations_test20.json' \
    --jobsDir '../exps/INSANet'
```
Arguments, jobsDir, evaluates all result files in the folder in a sequential.

---

## Benchmark
You can evaluate the result of the model with the scripts and draw all the state-of-the-art methods in a figure.

The figure represents the log-average miss rate (LAMR), the most popular metric for pedestrian detection tasks.

Annotation files only support a JSON format, whereas result files support a JSON and text format (multiple result files are supported). See the below.

```
cd evaluation
python evaluation_script.py \
    --annFile ./KAIST_annotation.json \
    --rstFile state_of_arts/INSANet_result.txt \
              state_of_arts/MLPD_result.txt \
              state_of_arts/MBNet_result.txt \
              state_of_arts/MSDS-RCNN_result.txt \
              state_of_arts/CIAN_result.txt \
    --evalFig KAIST_BENCHMARK.jpg

(optional) $ sh evaluation_script.sh
```

<p align="center"><img src="evaluation/KAIST_BENCHMARK.jpg"></p>

---

## Experiments

Note that &dagger; is the re-implemented performance with the proposed fusion method (other settings, such as the backbone and the training parameters, follow our approach). 

### KAIST
<table>
 <tr>
  <th rowspan="2"> Method </th>
  <th colspan="6"> Miss-Rate (%) </th>
 </tr>
 <tr>
  <th> ALL </th>
  <th> DAY </th>
  <th> NIGHT </th>
  <th> <i>Campus</i> </th>
  <th> <i>Road</i> </th>
  <th> <i>Downtown</i> </th>
 </tr>
 <tr>
  <td> ACF </td>
  <td align="center"> 47.32 </td>
  <td align="center"> 42.57 </td>
  <td align="center"> 56.17 </td>
  <td align="center"> 16.50 </td>
  <td align="center"> 6.68 </td>
  <td align="center"> 18.45 </td>
 </tr>
 <tr>
  <td> Halfway Fusion </td>
  <td align="center"> 25.75 </td>
  <td align="center"> 24.88 </td>
  <td align="center"> 26.59 </td>
  <td align="center"> - </td>
  <td align="center"> - </td>
  <td align="center"> - </td>
 </tr>
 <tr>
  <td> MSDS-RCNN </td>
  <td align="center"> 11.34 </td>
  <td align="center"> 10.53 </td>
  <td align="center"> 12.94 </td>
  <td align="center"> 11.26 </td>
  <td align="center"> 3.60 </td>
  <td align="center"> 14.80 </td>
 </tr>
 <tr>
  <td> AR-CNN </td>
  <td align="center"> 9.34 </td>
  <td align="center"> 9.94 </td>
  <td align="center"> 8.38 </td>
  <td align="center"> 11.73 </td>
  <td align="center"> 3.38 </td>
  <td align="center"> 11.73 </td>
 </tr>
  <td> Halfway Fusion&dagger; </td>
  <td align="center"> 8.31 </td>
  <td align="center"> 8.36 </td>
  <td align="center"> 8.27 </td>
  <td align="center"> 10.80 </td>
  <td align="center"> 3.74 </td>
  <td align="center"> 11.00 </td>
 <tr>
  <td> MBNet </td>
  <td align="center"> 8.31 </td>
  <td align="center"> 8.36 </td>
  <td align="center"> 8.27 </td>
  <td align="center"> 10.80 </td>
  <td align="center"> 3.74 </td>
  <td align="center"> 11.00 </td>
 </tr>
 <tr>
  <td> MLPD </td>
  <td align="center"> 7.58 </td>
  <td align="center"> 7.95 </td>
  <td align="center"> 6.95 </td>
  <td align="center"> 9.21 </td>
  <td align="center"> 5.04 </td>
  <td align="center"> 9.32 </td>
 </tr>
 <tr>
  <td> ICAFusion </td>
  <td align="center"> 7.17 </td>
  <td align="center"> 6.82 </td>
  <td align="center"> 7.85 </td>
  <td align="center"> - </td>
  <td align="center"> - </td>
  <td align="center"> - </td>
 </tr>
 <tr>
  <td> CFT&dagger; </td>
  <td align="center"> 6.75 </td>
  <td align="center"> 7.76 </td>
  <td align="center"> 4.59 </td>
  <td align="center"> 9.45 </td>
  <td align="center"> 3.47 </td>
  <td align="center"> 8.72 </td>
 </tr>
 <tr>
  <td> GAFF </td>
  <td align="center"> 6.48 </td>
  <td align="center"> 8.35 </td>
  <td align="center"> 3.46 </td>
  <td align="center"> 7.95 </td>
  <td align="center"> 3.70 </td>
  <td align="center"> 8.35 </td>
 </tr>
 <tr>
  <td> CFR </td>
  <td align="center"> 5.96 </td>
  <td align="center"> 7.77 </td>
  <th> 2.40 </th>
  <th> 7.45 </th>
  <td align="center"> 4.10 </td>
  <td align="center"> 7.25 </td>
 </tr>
 <tr>
  <td> <b>Ours<sub>(w/o shift)</sub></b> </td>
  <td align="center"> 6.12 </td>
  <td align="center"> 7.19 </td>
  <td align="center"> 4.37 </td>
  <td align="center"> 9.05 </td>
  <td align="center"> 3.24 </td>
  <td align="center"> 7.25 </td>
 </tr>
 <tr>
  <td> <b>Ours<sub>(w/ shift)</sub></b> </td>
  <th> 5.50 </th>
  <th> 6.29 </th>
  <td align="center"> 4.20 </td>
  <td align="center"> 7.64 </td>
  <th> 3.06 </th>
  <th> 6.72 </th>
 </tr>
</table>

---

## Acknowledgements
This paper would not have been possible without some awesome researches: [MLPD](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [KAIST](https://github.com/SoonminHwang/rgbt-ped-detection). 

We would also like to thank all the authors of our references for their excellent research.

---

## Citation
If our work is useful in your research, please consider citing our paper:
```
@article{lee2024insanet,
  title={INSANet: INtra-INter spectral attention network for effective feature fusion of multispectral pedestrian detection},
  author={Lee, Sangin and Kim, Taejoo and Shin, Jeongmin and Kim, Namil and Choi, Yukyung},
  journal={Sensors},
  volume={24},
  number={4},
  pages={1168},
  year={2024},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```




