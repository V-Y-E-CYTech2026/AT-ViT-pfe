import os
import pandas as pd
from PIL import Image, ImageFile
import io
from torch.utils.data import Dataset
from tqdm import tqdm

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DualInputPlantTraitDataset(Dataset):
    """Dataset class for loading original and segmented plant images."""
    def __init__(self, dataframe, original_img_dir, segmented_img_dir, transform_orig=None, transform_seg=None, subset='train', target_variable='thorns'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'code', 'train_test_set', and target_variable columns.
            original_img_dir (str): Directory containing original images.
            segmented_img_dir (str): Directory containing segmented images.
            transform_orig (callable, optional): Transforms for original images.
            transform_seg (callable, optional): Transforms for segmented images.
            subset (str): 'train' or 'test' to filter the dataframe.
            target_variable (str): Name of the target column in the dataframe.
        """
        self.dataframe = dataframe[dataframe['train_test_set'] == subset].reset_index(drop=True)
        self.original_img_dir = original_img_dir
        self.segmented_img_dir = segmented_img_dir
        self.transform_orig = transform_orig
        self.transform_seg = transform_seg
        self.target_variable = target_variable
        
        # Get list of unique class labels and create mapping
        self.classes = sorted(self.dataframe[target_variable].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Validate images and filter out corrupt ones
        self._validate_images()
        
        print(f"Loaded {subset} dataset with {len(self.dataframe)} samples and {len(self.classes)} classes")
    
    def _validate_images(self):
        """Check for corrupt or problematic images and filter them out."""
        valid_indices = []
        corrupt_files = []
        
        print("Validating image files...")
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            code = row['code']
            orig_path = os.path.join(self.original_img_dir, f"{code}.jpg")
            seg_path = os.path.join(self.segmented_img_dir, f"{code}.jpg")
            
            try:
                with Image.open(orig_path) as img:
                    img.verify()
                with Image.open(seg_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except (IOError, OSError, Image.DecompressionBombError) as e:
                corrupt_files.append((code, str(e)))
        
        if corrupt_files:
            print(f"Found {len(corrupt_files)} corrupt or problematic image files:")
            for code, err in corrupt_files[:10]:
                print(f"- {code}: {err}")
            if len(corrupt_files) > 10:
                print(f"  (and {len(corrupt_files) - 10} more)")
            
            results_dir = os.path.dirname(self.original_img_dir)
            with open(os.path.join(results_dir, 'corrupt_images.txt'), 'w') as f:
                for code, err in corrupt_files:
                    f.write(f"{code}: {err}\n")
            
            self.dataframe = self.dataframe.iloc[valid_indices].reset_index(drop=True)
            print(f"Filtered dataset to {len(self.dataframe)} valid samples")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        code = self.dataframe.loc[idx, 'code']
        label = self.dataframe.loc[idx, self.target_variable]
        label_idx = self.class_to_idx[label]
        
        # Load original image
        orig_path = os.path.join(self.original_img_dir, f"{code}.jpg")
        try:
            with open(orig_path, 'rb') as f:
                orig_image = Image.open(io.BytesIO(f.read())).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Failed to load original image: {orig_path} ({str(e)})")
            orig_image = Image.new('RGB', (240, 240), (0, 0, 0))
            label_idx = self.class_to_idx[0.0]
        
        # Load segmented image
        seg_path = os.path.join(self.segmented_img_dir, f"{code}.jpg")
        try:
            with open(seg_path, 'rb') as f:
                seg_image = Image.open(io.BytesIO(f.read())).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Failed to load segmented image: {seg_path} ({str(e)})")
            seg_image = Image.new('RGB', (224, 224), (0, 0, 0))
            label_idx = self.class_to_idx[0.0]
        
        if self.transform_orig:
            orig_image = self.transform_orig(orig_image)
        if self.transform_seg:
            seg_image = self.transform_seg(seg_image)
        
        return {'original': orig_image, 'segmented': seg_image}, label_idx, code