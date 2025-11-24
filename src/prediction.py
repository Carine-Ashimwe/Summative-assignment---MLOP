"""
Prediction Module
Handles model predictions for single and batch images
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import json
from typing import Union, List, Dict, Tuple


class ImagePredictor:
    """Class for making predictions with trained model"""
    
    def __init__(self, model_path=None, model=None, class_names=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file
            model: Loaded model object (alternative to model_path)
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        self.input_shape = (224, 224, 3)
        
        if model_path:
            self.load_model(model_path)
        elif model:
            self.model = model
        
        if self.model is None:
            raise ValueError("Either model_path or model must be provided")
        
        # Load metadata if available
        if model_path:
            self._load_metadata(model_path)
    
    def _load_metadata(self, model_path):
        """Load model metadata"""
        metadata_path = model_path.replace('.tf', '_metadata.json').replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', self.class_names)
                if 'input_shape' in metadata:
                    self.input_shape = tuple(metadata['input_shape'])
    
    def load_model(self, model_path):
        """Load model from file"""
        self.model = keras.models.load_model(model_path)
        self._load_metadata(model_path)
    
    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess a single image for prediction
        
        Args:
            image_input: Path to image file or image array
            
        Returns:
            Preprocessed image array
        """
        # Load image if path provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            image = image_input.copy()
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        height, width = self.input_shape[:2]
        image = cv2.resize(image, (width, height))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Expand dimensions for batch prediction
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_input: Union[str, np.ndarray], return_probabilities: bool = False) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_input: Path to image file or image array
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        if self.class_names:
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_index': int(predicted_class_idx)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                (self.class_names[i] if self.class_names else f"Class_{i}"): float(predictions[0][i])
                for i in range(len(predictions[0]))
            }
        
        return result
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                try:
                    processed_img = self.preprocess_image(img_path)
                    batch_images.append(processed_img[0])  # Remove batch dimension
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    results.append({
                        'image_path': img_path,
                        'error': str(e)
                    })
            
            if batch_images:
                # Stack into batch
                batch_array = np.array(batch_images)
                
                # Predict
                predictions = self.model.predict(batch_array, verbose=0)
                
                # Process results
                for j, pred in enumerate(predictions):
                    predicted_class_idx = np.argmax(pred)
                    confidence = float(pred[predicted_class_idx])
                    
                    if self.class_names:
                        predicted_class = self.class_names[predicted_class_idx]
                    else:
                        predicted_class = f"Class_{predicted_class_idx}"
                    
                    results.append({
                        'image_path': valid_paths[j],
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'class_index': int(predicted_class_idx)
                    })
        
        return results

