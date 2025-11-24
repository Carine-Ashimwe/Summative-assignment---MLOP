"""
Script to download and organize Flowers dataset for the ML pipeline
Downloads TensorFlow's flowers dataset and organizes it in the required structure
"""

import os
import shutil
import numpy as np
from pathlib import Path
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.utils import get_file
except ImportError:
    print("TensorFlow not installed. Please install: pip install tensorflow")
    exit(1)

def download_flowers_dataset():
    """Download and organize flowers dataset"""
    
    print("="*60)
    print("Flowers Dataset Downloader")
    print("="*60)
    
    # Dataset URL
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    
    # Create directories
    data_dir = Path('data/train')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print("\n📥 Downloading flowers dataset...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Download and extract
        dataset_path = get_file(
            'flower_photos',
            origin=dataset_url,
            untar=True,
            cache_dir='.',
            cache_subdir=''
        )
        
        # Move to data/train
        source_dir = Path('flower_photos')
        
        if source_dir.exists():
            print(f"\n✅ Dataset downloaded successfully!")
            
            # Get all class directories
            classes = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d)) 
                      and not d.startswith('.')]
            
            print(f"\n📁 Found {len(classes)} classes: {classes}")
            
            # Count images per class
            total_images = 0
            for cls in classes:
                cls_path = source_dir / cls
                images = [f for f in os.listdir(cls_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(images)
                print(f"   - {cls}: {len(images)} images")
            
            print(f"\n📊 Total images: {total_images}")
            
            # Move to data/train
            print("\n🔄 Organizing dataset structure...")
            for cls in classes:
                src = source_dir / cls
                dst = data_dir / cls
                
                if dst.exists():
                    print(f"   ⚠️  {cls} already exists, skipping...")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"   ✅ Moved {cls}")
            
            # Clean up
            if source_dir.exists():
                shutil.rmtree(source_dir)
            
            print("\n" + "="*60)
            print("✅ Dataset setup complete!")
            print("="*60)
            print(f"\n📁 Dataset location: {data_dir}")
            print(f"📊 Classes: {classes}")
            print(f"📈 Total images: {total_images}")
            print("\n🚀 You can now run: python train_model.py")
            
        else:
            print("❌ Error: Dataset directory not found after download")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz")
        print("\nThen extract and organize in data/train/")

def verify_dataset():
    """Verify dataset structure"""
    train_dir = Path('data/train')
    
    if not train_dir.exists():
        print("❌ data/train directory not found")
        return False
    
    classes = [d for d in os.listdir(train_dir) 
              if os.path.isdir(train_dir / d) and not d.startswith('.')]
    
    if len(classes) == 0:
        print("❌ No class directories found in data/train")
        print("   Please organize your images in subdirectories like:")
        print("   data/train/class1/")
        print("   data/train/class2/")
        return False
    
    print(f"\n✅ Dataset verified:")
    print(f"   Classes: {len(classes)}")
    total = 0
    for cls in classes:
        cls_path = train_dir / cls
        count = len([f for f in os.listdir(cls_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total += count
        print(f"   - {cls}: {count} images")
    print(f"   Total: {total} images")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_dataset()
    else:
        download_flowers_dataset()
        print("\n" + "="*60)
        print("Verifying dataset...")
        verify_dataset()

