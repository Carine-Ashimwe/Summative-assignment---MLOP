"""
Data Preprocessing Module
Handles image preprocessing for the ML pipeline
"""

import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    """Class for preprocessing image data"""
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
        self.label_encoder = LabelEncoder()
        
    def load_images_from_directory(self, data_dir):
        """
        Load images from directory structure: data_dir/class_name/*.jpg
        
        Args:
            data_dir: Root directory containing class subdirectories
            
        Returns:
            images: List of image arrays
            labels: List of labels
            class_names: List of unique class names
        """
        images = []
        labels = []
        class_names = []
        
        # Get all subdirectories (each represents a class)
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory {data_dir} does not exist")
            
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            # Skip hidden files and non-directories
            if os.path.isdir(class_path) and not class_name.startswith('.'):
                class_names.append(class_name)
                print(f"Loading images from class: {class_name}")
                
                # Load images from this class directory
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                if len(image_files) == 0:
                    print(f"  Warning: No images found in {class_path}")
                    continue
                
                loaded_count = 0
                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = self.load_image(img_path)
                        images.append(img)
                        labels.append(class_name)
                        loaded_count += 1
                    except Exception as e:
                        print(f"  Error loading {img_file}: {e}")
                
                print(f"  Loaded {loaded_count}/{len(image_files)} images from {class_name}")
        
        if len(images) == 0:
            raise ValueError(f"No images were successfully loaded from {data_dir}")
        
        # Return sorted class names for consistency
        unique_class_names = sorted(set(class_names))
        print(f"\n✅ Successfully loaded {len(images)} images from {len(unique_class_names)} classes")
        
        return np.array(images), np.array(labels), unique_class_names
    
    def load_image(self, img_path):
        """
        Load and preprocess a single image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.target_size)
        
        # Normalize pixel values to [0, 1]
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def encode_labels(self, labels):
        """
        Encode string labels to integers
        
        Args:
            labels: List of label strings
            
        Returns:
            Encoded labels
        """
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels):
        """
        Decode integer labels back to strings
        
        Args:
            encoded_labels: Encoded label integers
            
        Returns:
            Decoded label strings
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def prepare_data(self, images, labels, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            images: Image arrays
            labels: Labels
            test_size: Proportion of data for test set
            validation_size: Proportion of train data for validation
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_generator(self, augment=True):
        """
        Create data generator for training with optional augmentation
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            ImageDataGenerator instance
        """
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()
        
        return datagen

