"""
Helper script to download sample data for testing
This is optional - you can use your own dataset instead
"""

import os
import urllib.request
import zipfile
import shutil

def download_file(url, destination):
    """Download a file from URL"""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded to {destination}")

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def setup_sample_data():
    """Setup sample data structure"""
    
    print("="*60)
    print("Sample Data Setup Helper")
    print("="*60)
    print("\nThis script helps you set up sample data for testing.")
    print("\nOptions:")
    print("1. Create empty directory structure (for your own data)")
    print("2. Download CIFAR-10 dataset (10 classes, 60000 images)")
    print("3. Skip (use existing data)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Create empty structure
        classes = input("Enter class names (comma-separated, e.g., cat,dog,bird): ").strip().split(',')
        classes = [c.strip() for c in classes if c.strip()]
        
        if not classes:
            print("No classes provided. Exiting.")
            return
        
        base_dir = 'data/train'
        os.makedirs(base_dir, exist_ok=True)
        
        for cls in classes:
            cls_dir = os.path.join(base_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            print(f"Created directory: {cls_dir}")
        
        print(f"\n✅ Created {len(classes)} class directories.")
        print(f"📁 Please add your images to: {base_dir}/<class_name>/")
        
    elif choice == "2":
        print("\n⚠️  Note: CIFAR-10 download requires manual setup.")
        print("Please download from: https://www.cs.toronto.edu/~kriz/cifar.html")
        print("Or use TensorFlow/Keras built-in dataset:")
        print("\n  from tensorflow.keras.datasets import cifar10")
        print("  (x_train, y_train), (x_test, y_test) = cifar10.load_data()")
        
    elif choice == "3":
        print("Skipping data setup. Using existing data.")
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    setup_sample_data()

