"""
Simple script to download Flowers dataset without requiring TensorFlow
Uses urllib and tarfile directly
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path

def download_flowers_dataset():
    """Download and organize flowers dataset"""
    
    print("="*60)
    print("Flowers Dataset Downloader")
    print("="*60)
    
    # Dataset URL
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dataset_file = "flower_photos.tgz"
    
    # Create directories
    data_dir = Path('data/train')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (data_dir.exists() and 
        len([d for d in os.listdir(data_dir) 
             if os.path.isdir(data_dir / d) and not d.startswith('.')]) > 0):
        print("\n✅ Dataset already exists in data/train/")
        print("   Run with 'python download_flowers_simple.py verify' to check")
        return
    
    # Download dataset
    print("\n📥 Downloading flowers dataset...")
    print("   URL: " + dataset_url)
    print("   This may take a few minutes (dataset is ~218 MB)...")
    
    try:
        # Download
        print("\n   Downloading...")
        urllib.request.urlretrieve(dataset_url, dataset_file)
        print("   ✅ Download complete!")
        
        # Extract
        print("\n   Extracting archive...")
        with tarfile.open(dataset_file, 'r:gz') as tar:
            tar.extractall()
        print("   ✅ Extraction complete!")
        
        # Move to data/train
        source_dir = Path('flower_photos')
        
        if source_dir.exists():
            print(f"\n✅ Dataset downloaded successfully!")
            
            # Get all class directories (exclude LICENSE.txt)
            classes = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(source_dir / d) 
                      and not d.startswith('.')]
            
            print(f"\n📁 Found {len(classes)} classes: {classes}")
            
            # Count images per class
            total_images = 0
            for cls in classes:
                cls_path = source_dir / cls
                images = [f for f in os.listdir(cls_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
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
                # Remove LICENSE.txt if exists
                license_file = source_dir / 'LICENSE.txt'
                if license_file.exists():
                    license_file.unlink()
                try:
                    shutil.rmtree(source_dir)
                except:
                    pass
            
            # Remove tar file
            if os.path.exists(dataset_file):
                os.remove(dataset_file)
            
            print("\n" + "="*60)
            print("✅ Dataset setup complete!")
            print("="*60)
            print(f"\n📁 Dataset location: {data_dir}")
            print(f"📊 Classes: {sorted(classes)}")
            print(f"📈 Total images: {total_images}")
            print("\n🚀 You can now run: python train_model.py")
            
        else:
            print("❌ Error: Dataset directory not found after extraction")
            
    except urllib.error.URLError as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Please check your internet connection.")
        print("   Or download manually from:")
        print("   https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz")
        print("\n   Then extract and copy the class folders to data/train/")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz")
        print("\n   Then extract and copy the class folders to data/train/")

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
    class_counts = {}
    for cls in sorted(classes):
        cls_path = train_dir / cls
        count = len([f for f in os.listdir(cls_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        total += count
        class_counts[cls] = count
        print(f"   - {cls}: {count} images")
    print(f"   Total: {total} images")
    
    # Check if we have enough images
    min_images = min(class_counts.values())
    if min_images < 50:
        print(f"\n⚠️  Warning: Some classes have fewer than 50 images.")
        print(f"   Minimum: {min_images} images per class")
        print(f"   For best results, aim for at least 100-200 images per class.")
    
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

