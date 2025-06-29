"""
Data preprocessing utilities for butterfly image classification
Handles dataset organization, augmentation, and preparation
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import json

class ButterflyDataPreprocessor:
    """
    Comprehensive data preprocessing for butterfly classification dataset
    
    Features:
    - Dataset organization and splitting
    - Image quality validation
    - Data distribution analysis
    - Automated directory structure creation
    """
    
    def __init__(self, base_data_dir, output_dir):
        self.base_data_dir = base_data_dir
        self.output_dir = output_dir
        self.species_info = {}
        
    def analyze_dataset(self):
        """
        Analyze the butterfly dataset structure and distribution
        
        Returns:
            dict: Dataset statistics and information
        """
        
        species_count = {}
        total_images = 0
        image_sizes = []
        corrupted_images = []
        
        print("Analyzing butterfly dataset...")
        
        for species_folder in os.listdir(self.base_data_dir):
            species_path = os.path.join(self.base_data_dir, species_folder)
            
            if not os.path.isdir(species_path):
                continue
                
            image_count = 0
            
            for image_file in os.listdir(species_path):
                image_path = os.path.join(species_path, image_file)
                
                try:
                    with Image.open(image_path) as img:
                        image_sizes.append(img.size)
                        image_count += 1
                        total_images += 1
                except Exception as e:
                    corrupted_images.append(image_path)
                    print(f"Corrupted image: {image_path} - {e}")
            
            species_count[species_folder] = image_count
        
        # Calculate statistics
        avg_images_per_species = np.mean(list(species_count.values()))
        min_images = min(species_count.values())
        max_images = max(species_count.values())
        
        # Most common image sizes
        size_counter = Counter(image_sizes)
        common_sizes = size_counter.most_common(5)
        
        stats = {
            'total_species': len(species_count),
            'total_images': total_images,
            'avg_images_per_species': avg_images_per_species,
            'min_images_per_species': min_images,
            'max_images_per_species': max_images,
            'species_distribution': species_count,
            'common_image_sizes': common_sizes,
            'corrupted_images': corrupted_images
        }
        
        self.species_info = stats
        return stats
    
    def visualize_distribution(self):
        """Visualize species distribution and dataset statistics"""
        
        if not self.species_info:
            self.analyze_dataset()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Species distribution histogram
        species_counts = list(self.species_info['species_distribution'].values())
        axes[0, 0].hist(species_counts, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Images per Species')
        axes[0, 0].set_xlabel('Number of Images')
        axes[0, 0].set_ylabel('Number of Species')
        
        # Top 20 species by image count
        sorted_species = sorted(
            self.species_info['species_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        species_names = [item[0][:15] + '...' if len(item[0]) > 15 else item[0] 
                        for item in sorted_species]
        counts = [item[1] for item in sorted_species]
        
        axes[0, 1].barh(species_names, counts, color='lightcoral')
        axes[0, 1].set_title('Top 20 Species by Image Count')
        axes[0, 1].set_xlabel('Number of Images')
        
        # Image size distribution
        sizes = [size[0] * size[1] for size in [size for size, _ in self.species_info['common_image_sizes']]]
        size_labels = [f"{size[0]}x{size[1]}" for size, _ in self.species_info['common_image_sizes']]
        
        axes[1, 0].pie(sizes, labels=size_labels, autopct='%1.1f%%')
        axes[1, 0].set_title('Common Image Sizes Distribution')
        
        # Dataset summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Dataset Summary:
        
        Total Species: {self.species_info['total_species']}
        Total Images: {self.species_info['total_images']:,}
        
        Images per Species:
        • Average: {self.species_info['avg_images_per_species']:.1f}
        • Minimum: {self.species_info['min_images_per_species']}
        • Maximum: {self.species_info['max_images_per_species']}
        
        Data Quality:
        • Corrupted Images: {len(self.species_info['corrupted_images'])}
        • Success Rate: {((self.species_info['total_images'] / (self.species_info['total_images'] + len(self.species_info['corrupted_images']))) * 100):.1f}%
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def create_balanced_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                            min_images_per_class=10):
        """
        Create balanced train/validation/test splits
        
        Args:
            train_ratio (float): Proportion for training set
            val_ratio (float): Proportion for validation set
            test_ratio (float): Proportion for test set
            min_images_per_class (int): Minimum images required per class
        """
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if not self.species_info:
            self.analyze_dataset()
        
        # Create output directories
        splits = ['train', 'validation', 'test']
        for split in splits:
            split_dir = os.path.join(self.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
        
        split_info = {'train': {}, 'validation': {}, 'test': {}}
        
        print("Creating balanced dataset splits...")
        
        for species_folder in os.listdir(self.base_data_dir):
            species_path = os.path.join(self.base_data_dir, species_folder)
            
            if not os.path.isdir(species_path):
                continue
            
            # Get all valid images for this species
            image_files = []
            for image_file in os.listdir(species_path):
                image_path = os.path.join(species_path, image_file)
                try:
                    with Image.open(image_path) as img:
                        image_files.append(image_file)
                except:
                    continue
            
            # Skip species with insufficient images
            if len(image_files) < min_images_per_class:
                print(f"Skipping {species_folder}: only {len(image_files)} images (min: {min_images_per_class})")
                continue
            
            # Calculate split sizes
            n_images = len(image_files)
            n_train = max(1, int(n_images * train_ratio))
            n_val = max(1, int(n_images * val_ratio))
            n_test = n_images - n_train - n_val
            
            # Randomly shuffle and split
            np.random.shuffle(image_files)
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Create species directories in each split
            for split in splits:
                species_split_dir = os.path.join(self.output_dir, split, species_folder)
                os.makedirs(species_split_dir, exist_ok=True)
            
            # Copy files to respective splits
            file_splits = [
                ('train', train_files),
                ('validation', val_files),
                ('test', test_files)
            ]
            
            for split_name, files in file_splits:
                split_info[split_name][species_folder] = len(files)
                
                for image_file in files:
                    src_path = os.path.join(species_path, image_file)
                    dst_path = os.path.join(self.output_dir, split_name, species_folder, image_file)
                    shutil.copy2(src_path, dst_path)
        
        # Save split information
        with open(os.path.join(self.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        # Print split summary
        print("\nDataset split completed!")
        for split_name in splits:
            total_images = sum(split_info[split_name].values())
            total_species = len(split_info[split_name])
            print(f"{split_name.capitalize()}: {total_images:,} images, {total_species} species")
        
        return split_info
    
    def validate_image_quality(self, min_size=(100, 100), max_size=(5000, 5000)):
        """
        Validate image quality and identify problematic images
        
        Args:
            min_size (tuple): Minimum acceptable image dimensions
            max_size (tuple): Maximum acceptable image dimensions
        
        Returns:
            dict: Quality validation results
        """
        
        quality_issues = {
            'too_small': [],
            'too_large': [],
            'corrupted': [],
            'wrong_format': [],
            'valid': []
        }
        
        print("Validating image quality...")
        
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # Check image format
                        if img.format not in ['JPEG', 'PNG', 'JPG']:
                            quality_issues['wrong_format'].append(file_path)
                            continue
                        
                        # Check image size
                        if width < min_size[0] or height < min_size[1]:
                            quality_issues['too_small'].append(file_path)
                        elif width > max_size[0] or height > max_size[1]:
                            quality_issues['too_large'].append(file_path)
                        else:
                            quality_issues['valid'].append(file_path)
                            
                except Exception as e:
                    quality_issues['corrupted'].append(file_path)
        
        # Print quality report
        total_images = sum(len(issues) for issues in quality_issues.values())
        print(f"\nImage Quality Report:")
        print(f"Total images processed: {total_images:,}")
        print(f"Valid images: {len(quality_issues['valid']):,} ({len(quality_issues['valid'])/total_images*100:.1f}%)")
        print(f"Too small: {len(quality_issues['too_small'])}")
        print(f"Too large: {len(quality_issues['too_large'])}")
        print(f"Wrong format: {len(quality_issues['wrong_format'])}")
        print(f"Corrupted: {len(quality_issues['corrupted'])}")
        
        return quality_issues
    
    def create_species_mapping(self):
        """
        Create species name mapping and metadata
        
        Returns:
            dict: Species mapping with scientific names and common names
        """
        
        # This would typically be loaded from a scientific database
        # For demonstration, we'll create a basic mapping
        species_mapping = {}
        
        for species_folder in os.listdir(self.base_data_dir):
            species_path = os.path.join(self.base_data_dir, species_folder)
            
            if not os.path.isdir(species_path):
                continue
            
            # Basic mapping (in real implementation, this would come from a database)
            species_mapping[species_folder] = {
                'common_name': species_folder.replace('_', ' ').title(),
                'scientific_name': f"Species {species_folder}",  # Placeholder
                'family': 'Lepidoptera',
                'image_count': len([f for f in os.listdir(species_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            }
        
        # Save species mapping
        with open(os.path.join(self.output_dir, 'species_mapping.json'), 'w') as f:
            json.dump(species_mapping, f, indent=2)
        
        return species_mapping

# Example usage
def main():
    """
    Example preprocessing pipeline for butterfly dataset
    """
    
    # Initialize preprocessor
    preprocessor = ButterflyDataPreprocessor(
        base_data_dir='raw_butterfly_data',
        output_dir='processed_butterfly_data'
    )
    
    # Analyze dataset
    stats = preprocessor.analyze_dataset()
    print(f"Found {stats['total_species']} species with {stats['total_images']:,} total images")
    
    # Visualize distribution
    preprocessor.visualize_distribution()
    
    # Validate image quality
    quality_results = preprocessor.validate_image_quality()
    
    # Create balanced splits
    split_info = preprocessor.create_balanced_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        min_images_per_class=10
    )
    
    # Create species mapping
    species_mapping = preprocessor.create_species_mapping()
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()