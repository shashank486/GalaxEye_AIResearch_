# EO-SAR Change Detection using U-Net

## Project Title & Description

This project implements a deep learning-based EO-SAR satellite image change detection pipeline using a baseline U-Net segmentation architecture. The objective is to detect structural and environmental changes between pre-event and post-event satellite imagery.

The workflow includes:
- dataset preprocessing,
- multimodal image fusion,
- segmentation model training,
- imbalance-aware optimization,
- evaluation using IoU/F1 metrics,
- and qualitative prediction visualization.

The model receives concatenated pre-event and post-event images as a 6-channel input and predicts a binary change mask highlighting changed regions.

---

# Requirements

## Python Version

```bash
Python 3.10+
```
## Dependencies

# Install dependencies using:

```bash:
pip install -r requirements.txt
```

# Environment Setup
- Create Virtual Environment
- Using venv

```bash 
python -m venv venv
```
# Activate Environment
-  macOS/Linux
```bash
source venv/bin/activate
```
- Windows

```bash:
venv\Scripts\activate
```

# Dataset Structure

# Expected dataset directory structure:
 Create the dataset folder in directory include the dataset(train,test,val) in this folder

# 1. Training

## training from the hugging face dataset
run in Google colab (GPU) (upload the file given python files for training: dataset,model,loss,train.py) or run in any code editor 

- run the below commands before training: 
1. 

```bash
!pip install -q huggingface_hub
```

2. 
 ```bash
 from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="doron333/change-detection-dataset",
    repo_type="dataset",
    local_dir="/content/change_dataset"
)
```

3. 
```bash
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset

train_dataset = ChangeDetectionDataset(
    "/content/change_dataset/train"
)
```
4. 
```bash
!unzip /content/change_dataset/train.zip -d /content/change_dataset/
```

5.  
```bash 
!python train.py
```


* ## then download the latest trained model for evalution and testing

# 2. Evaluation
- Run Evaluation
-  val dataset
```bash
python evaluate.py \
--data_path dataset/val \
--weights model_latest.pth
```

- test dataset
```bash
python evaluate.py \
--data_path dataset/test \
--weights model_latest.pth
```

# 3. Prediction Visualization

- Validation dataset:
```bash
python visualize.py \
--data_path dataset/val \
--weights model_latest.pth
```

- test dataset:
```bash
python visualize.py \
--data_path dataset/test \
--weights model_latest.pth
```


# Model Weights
- drive link: 
 ```bash
https://drive.google.com/drive/folders/1bO-nmbV87PZ_4sJNYfFJYzzKaOeps_ks?usp=sharing
```
- hugging face :
```bash

```
# Results
 - Validation Results

| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.0127 |
| Precision | 0.0231 |
| Recall    | 0.0583 |
| F1 Score  | 0.0229 |

- Confusion Matrix (val)
  
|                  | Predicted No Change | Predicted Change |
| ---------------- | ------------------- | ---------------- |
| Actual No Change | 16,912,570          | 4,480,376        |
| Actual Change    | 397,509             | 98,569           |


# Test Results
| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.0072 |
| Precision | 0.0081 |
| Recall    | 0.1184 |
| F1 Score  | 0.0137 |

- Confusion Matrix (Test Set)

|                  | Predicted No Change | Predicted Change |
| ---------------- | ------------------- | ---------------- |
| Actual No Change | 3,884,505           | 1,120,069        |
| Actual Change    | 30,610              | 11,088           |



# References

- Research Papers
1. U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, Thomas Brox.
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.
2. ChangeFormer: A Transformer-Based Siamese Network for Change Detection
Wele Gedara Chaminda Bandara, Jeya Maria Jose Valanarasu, Vishal M. Patel.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022.
3. Fully Convolutional Networks for Semantic Segmentation
Jonathan Long, Evan Shelhamer, Trevor Darrell.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

# Dataset References
- Hugging Face Change Detection Dataset : https://huggingface.co/datasets/doron333/change-detection-dataset?utm_source=chatgpt.com

# Frameworks and Libraries
1. PyTorch Documentation
2. Torchvision Documentation
3. OpenCV Documentation
4. NumPy Documentation
5. Matplotlib Documentation
6. Scikit-learn Documentation
7. Hugging Face Hub Documentation

# Development and Training Platforms
1. Google Colab
2. Hugging Face


- Author

- Shashank B
- Satellite AI Research Intern Assignment
- GalaxEye Space
