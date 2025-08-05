
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path


# Color encoding mapping
COLOR_MAPPING = {
    'red': 0, 'green': 1, 'blue': 2, 'yellow': 3, 
    'orange': 4, 'purple': 5, 'cyan': 6, 'magenta': 7
}
class ColorConditionalDataset(Dataset):
    def __init__(self, dataset_path, split='training', image_size=(256, 256), augment=False):
        """
        Dataset that provides image, color, and target for color-conditional training
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Paths to inputs and outputs
        self.split_path = self.dataset_path / split
        self.inputs_path = self.split_path / 'inputs'
        self.outputs_path = self.split_path / 'outputs'
        
        # Load data.json
        json_path = self.split_path / 'data.json'
        if not json_path.exists():
            raise FileNotFoundError(f"data.json not found in {self.split_path}")
        
        with open(json_path, 'r') as f:
            self.data_mappings = json.load(f)
        
        # Create list of valid data pairs
        self.valid_pairs = []
        for mapping in self.data_mappings:
            input_path = self.inputs_path / mapping['input_polygon']
            output_path = self.outputs_path / mapping['output_image']
            color = mapping['colour']
            
            if input_path.exists() and output_path.exists() and color in COLOR_MAPPING:
                self.valid_pairs.append({
                    'input_path': input_path,
                    'output_path': output_path,
                    'color': color,
                    'color_idx': COLOR_MAPPING[color]
                })
        
        # Define transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        # Augmentation transforms
        if augment and split == 'training':
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            self.augment_transform = None
        
        print(f"Found {len(self.valid_pairs)} valid pairs in {split} set")
        self.print_dataset_stats()
    
    def print_dataset_stats(self):
        """Print statistics about the dataset"""
        shapes = {}
        colors = {}
        
        for pair in self.valid_pairs:
            shape = pair['input_path'].stem
            color = pair['color']
            
            shapes[shape] = shapes.get(shape, 0) + 1
            colors[color] = colors.get(color, 0) + 1
        
        print(f"Shapes: {dict(sorted(shapes.items()))}")
        print(f"Colors: {dict(sorted(colors.items()))}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # Load image and mask
        try:
            image = Image.open(pair['input_path']).convert('RGB')
            mask = Image.open(pair['output_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {pair['input_path']} or {pair['output_path']}: {e}")
            raise
        
        # Apply augmentation if specified
        if self.augment_transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            mask = self.augment_transform(mask)
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # Normalize mask to [0, 1] range
        mask = mask / 255.0 if mask.max() > 1.0 else mask
        
        # Return image, color index, mask, and color name for debugging
        return image, pair['color_idx'], mask, pair['color']