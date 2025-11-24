"""
Standalone script to train the model
Run this script to train the model outside of the notebook
"""

import os
import sys
from src.preprocessing import ImagePreprocessor
from src.model import ImageClassifierModel

def main():
    """Main training function"""
    
    # Configuration
    DATA_DIR = 'data/train'
    MODELS_DIR = 'models'
    TARGET_SIZE = (224, 224)
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print("="*60)
    print("ML Image Classification - Model Training")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n Error: Training data directory not found at {DATA_DIR}")
        print("\nPlease organize your training images in the following structure:")
        print("data/train/")
        print("  ├── class1/")
        print("  │   ├── img1.jpg")
        print("  │   └── img2.jpg")
        print("  ├── class2/")
        print("  │   ├── img1.jpg")
        print("  │   └── img2.jpg")
        print("  └── ...")
        sys.exit(1)
    
    # Step 1: Load and preprocess data
    print("\n Step 1: Loading and preprocessing data...")
    preprocessor = ImagePreprocessor(target_size=TARGET_SIZE, normalize=True)
    
    try:
        images, labels, class_names = preprocessor.load_images_from_directory(DATA_DIR)
        print(f" Loaded {len(images)} images from {len(class_names)} classes")
        print(f"   Classes: {class_names}")
    except Exception as e:
        print(f" Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Encode labels and split data
    print("\n Step 2: Encoding labels and splitting data...")
    encoded_labels = preprocessor.encode_labels(labels)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        images, encoded_labels, test_size=0.2, validation_size=0.1, random_state=42
    )
    print(f" Data split:")
    print(f"   - Training: {len(X_train)} images")
    print(f"   - Validation: {len(X_val)} images")
    print(f"   - Test: {len(X_test)} images")
    
    # Step 3: Create model
    print("\n  Step 3: Creating model architecture...")
    model = ImageClassifierModel(
        num_classes=len(class_names),
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        model_name='image_classifier'
    )
    model.set_class_names(class_names)
    model.build_model(use_transfer_learning=True, dropout_rate=0.5)
    print(" Model created successfully")
    
    # Step 4: Train model
    print(f"\n Step 4: Training model for {EPOCHS} epochs...")
    datagen = preprocessor.get_data_generator(augment=True)
    
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        data_generator=datagen
    )
    
    # Step 5: Evaluate model
    print("\n Step 5: Evaluating model on test set...")
    eval_results = model.evaluate(X_test, y_test, class_names=class_names)
    
    print(f"\nTest Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    
    # Step 6: Save model
    print("\n Step 6: Saving model...")
    model_path = model.save_model(model_dir=MODELS_DIR, save_format='tf')
    
    print("\n" + "="*60)
    print(" Training completed successfully!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    print("\nYou can now run the Flask API with: python app.py")

if __name__ == '__main__':
    main()

