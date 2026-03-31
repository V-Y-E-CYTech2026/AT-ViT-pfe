# AT-ViT: Area-Targeted Cross-Attention Vision Transformer

This repository provides an implementation of AT-ViT, a modified Vision Transformer using area-targeted attention toward plant regions in herbarium images.

## Repository Structure

```
AT-ViT-main/
├── AT-ViT
│   ├── config.py
│   ├── dataset.py
│   ├── main.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── Baseline
│   ├── config.py
│   ├── dataset.py
│   ├── main.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── samples
│   ├── IoU
│   ├── attention_maps
│   └── patch_embedding_gradcam
├── noise-generation.ipynb
└── models-links.md
```

## Installation

### Requirements

- Python 3.8 or higher
- CUDA (if using GPU)

### Install dependencies

```bash
pip install torch torchvision pandas matplotlib scikit-learn timm opencv-python seaborn dotenv grad-cam ipykernel
```

### Clone the repository

```bash
git clone https://github.com/takichehhat/AT-ViT.git
cd AT-ViT-main
```

## Data Preparation

 Place the datasets in a folder and specify their paths in the `.env` file  


## Usage

### Training and Evaluation

Run the full pipeline (training, evaluation, and visualizations) for each model:

- **AT-ViT**:
  ```bash
  python AT-ViT/main.py
  ```

- **Baseline CrossViT**:
  ```bash
  python Baseline/main.py
  ```  

The `main.py` script performs the following key tasks:

- Loads the dataset and applies transforms  
- Trains the model  
- Evaluates the model on both clean and noisy datasets  
- Generates attention maps and Grad-CAM visualizations  

## Synthetic Noise Generation

To reproduce robustness testing with artificial noise, refer to:

- `noise-generation.ipynb`

## Additional Documentation

- `models-links.md` provides download links to pretrained models 

## Citation

If you use this work, please cite the following:

```bibtex
@article{ATViT2025,
  title={AT-ViT: An Area-Targeted Cross-Attention Vision Transformer Directed Toward Plant Regions in Herbarium Images},
  author={Sedrat, Amani and Chehhat, Takieddine and Sklab, Youcef},
  year={2025},
  journal={IET Computer Vision}
}
```



## Ajouts pfe

J'ai rajouté un script ```recreateCSV.py``` pour recréer le fichier CSV nécessaire en plus du dataset. 