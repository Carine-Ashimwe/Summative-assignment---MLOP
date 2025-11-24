"""
Model Creation and Training Module
Handles model architecture, training, and saving
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from datetime import datetime


class ImageClassifierModel:
    """Image classification model using transfer learning"""
    
    def __init__(self, num_classes, input_shape=(224, 224, 3), model_name='image_classifier'):
        """
        Initialize model
        
        Args:
            num_classes: Number of classification classes
            input_shape: Input image shape (height, width, channels)
            model_name: Name for saving the model
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_names = None
        
    def build_model(self, use_transfer_learning=True, dropout_rate=0.5):
        """
        Build the model architecture
        
        Args:
            use_transfer_learning: Whether to use MobileNetV2 for transfer learning
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled model
        """
        if use_transfer_learning:
            # Use MobileNetV2 as base model for transfer learning
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers (optional - can fine-tune later)
            base_model.trainable = False
            
            # Build model
            inputs = keras.Input(shape=self.input_shape)
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            self.model = keras.Model(inputs, outputs)
        else:
            # Simple CNN model from scratch
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(dropout_rate),
                layers.Dense(512, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, validation_data=None,
              data_generator=None, callbacks=None):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data tuple
            data_generator: Optional data generator for augmentation
            callbacks: Optional list of callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare validation data
        if validation_data is None:
            validation_data = (X_val, y_val)
        
        # Setup callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train model
        if data_generator:
            train_gen = data_generator.flow(X_train, y_train, batch_size=batch_size)
            self.history = self.model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def _get_default_callbacks(self):
        """Get default training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels
            class_names: List of class names for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist(),
            'class_names': class_names
        }
        
        return results
    
    def save_model(self, model_dir='models', save_format='tf'):
        """
        Save the trained model
        
        Args:
            model_dir: Directory to save model
            save_format: 'tf' for SavedModel or 'h5' for HDF5
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        if save_format == 'tf':
            model_path = os.path.join(model_dir, f'{self.model_name}.tf')
            self.model.save(model_path, save_format='tf')
        else:
            model_path = os.path.join(model_dir, f'{self.model_name}.h5')
            self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, f'{self.model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to saved model
        """
        self.model = keras.models.load_model(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.tf', '_metadata.json').replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names')
                self.num_classes = metadata.get('num_classes')
                self.input_shape = tuple(metadata.get('input_shape'))
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def set_class_names(self, class_names):
        """Set class names for the model"""
        self.class_names = class_names

