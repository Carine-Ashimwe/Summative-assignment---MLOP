"""
Complete setup and testing script for ML Image Classification Pipeline
This script helps verify everything is working correctly
"""

import os
import sys
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("1. Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    print("✅ Python version OK")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("2. Checking Dependencies")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'flask': 'Flask',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'opencv-python': 'OpenCV',
        'pillow': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    for module, name in required_packages.items():
        try:
            if module == 'opencv-python':
                __import__('cv2')
            elif module == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True

def check_directory_structure():
    """Check project directory structure"""
    print_header("3. Checking Directory Structure")
    
    required_dirs = [
        'src',
        'data/train',
        'data/test',
        'models',
        'static',
        'templates',
        'notebook'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some directories are missing")
        print("   They should be created automatically")
    
    return all_exist

def check_dataset():
    """Check if dataset exists and is organized correctly"""
    print_header("4. Checking Dataset")
    
    train_dir = Path('data/train')
    
    if not train_dir.exists():
        print("❌ data/train directory not found")
        print("   Run: python download_flowers_simple.py")
        return False
    
    # Get class directories
    classes = [d for d in os.listdir(train_dir) 
              if os.path.isdir(train_dir / d) and not d.startswith('.')]
    
    if len(classes) == 0:
        print("❌ No class directories found in data/train")
        print("   Download dataset with: python download_flowers_simple.py")
        return False
    
    print(f"✅ Found {len(classes)} classes:")
    
    total_images = 0
    min_images = float('inf')
    for cls in sorted(classes):
        cls_path = train_dir / cls
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        count = len(images)
        total_images += count
        min_images = min(min_images, count)
        status = "✅" if count >= 50 else "⚠️"
        print(f"   {status} {cls}: {count} images")
    
    print(f"\n📊 Total images: {total_images}")
    
    if min_images < 50:
        print(f"⚠️  Warning: Some classes have fewer than 50 images")
        print(f"   Minimum: {min_images} images per class")
        print(f"   For best results, aim for 100+ images per class")
    
    if total_images < 200:
        print(f"⚠️  Warning: Total images is low ({total_images})")
        print(f"   Recommended: 500+ images for good results")
    
    return len(classes) >= 2

def check_model():
    """Check if trained model exists"""
    print_header("5. Checking Trained Model")
    
    model_paths = [
        'models/image_classifier.tf',
        'models/image_classifier.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Found model: {model_path}")
            
            # Check for metadata
            metadata_path = model_path.replace('.tf', '_metadata.json').replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                print(f"✅ Model metadata found")
            else:
                print(f"⚠️  Model metadata not found (optional)")
            
            return True
    
    print("❌ No trained model found")
    print("   Train model with: python train_model.py")
    print("   Or use the Jupyter notebook: notebook/ml_summative.ipynb")
    return False

def check_api_files():
    """Check API and UI files"""
    print_header("6. Checking API and UI Files")
    
    required_files = [
        'app.py',
        'static/style.css',
        'static/script.js',
        'templates/index.html'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_source_files():
    """Check source module files"""
    print_header("7. Checking Source Files")
    
    required_files = [
        'src/preprocessing.py',
        'src/model.py',
        'src/prediction.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def print_summary(results):
    """Print test summary"""
    print_header("Setup Summary")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 All checks passed! Your project is ready to use.")
        print("\nNext steps:")
        print("  1. If model not trained: python train_model.py")
        print("  2. Start API: python app.py")
        print("  3. Open browser: http://localhost:5000")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not results['Dependencies']:
            print("  - Install dependencies: pip install -r requirements.txt")
        if not results['Dataset']:
            print("  - Download dataset: python download_flowers_simple.py")
        if not results['Model']:
            print("  - Train model: python train_model.py")
    
    print("="*60 + "\n")
    
    return all_passed

def main():
    """Main setup and test function"""
    print("\n" + "="*60)
    print("  ML Image Classification Pipeline - Setup & Test")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Dataset': check_dataset(),
        'Model': check_model(),
        'API Files': check_api_files(),
        'Source Files': check_source_files()
    }
    
    print_summary(results)
    
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())

