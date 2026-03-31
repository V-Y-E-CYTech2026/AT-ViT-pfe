import os
import random
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv

def setup_environment(seed=42):
    """Set up the environment with random seed and CUDA settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    config = {
        'base_dir': Path(os.getenv("BASE_DIR")),
        'csv_path': Path(os.getenv("BASE_DIR")) / os.getenv("CSV_FILENAME"),
        'original_img_dir': Path(os.getenv("ORIGINAL_IMG_DIR")),
        'segmented_img_dir': Path(os.getenv("SEGMENTED_IMG_DIR")),
        'results_dir': Path(os.getenv("BASE_DIR")) / os.getenv("RESULTS_DIR"),
        'target_variable': os.getenv("TARGET_VARIABLE"),
        'original_noisy_base_dir': Path(os.getenv("ORIGINAL_NOISY_BASE_DIR")),
        'segmented_noisy_base_dir': Path(os.getenv("SEGMENTED_NOISY_BASE_DIR")),
        'background_noise_dir': Path(os.getenv("ORIGINAL_NOISY_BASE_DIR")) / "background_only",
        'plant_noise_dir': Path(os.getenv("ORIGINAL_NOISY_BASE_DIR")) / "plant_only",
        'background_noise_seg_dir': Path(os.getenv("SEGMENTED_NOISY_BASE_DIR")) / "background_only",
        'plant_noise_seg_dir': Path(os.getenv("SEGMENTED_NOISY_BASE_DIR")) / "plant_only",
        'batch_size': 16,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'weight_decay': 0.1,
        # cuda 0 for personnal computer
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    }
    config['results_dir'].mkdir(exist_ok=True)
    return config