"""
Quick Training Script - Simple wrapper for fast training
This makes training MUCH easier with sensible defaults
"""

import subprocess
import os
import sys
import shutil
from pathlib import Path

def quick_train():
    """Run quick training with optimal settings"""
    
    print("=" * 70)
    print("  🚀 QUICK TRAIN - Easy Model Training")
    print("=" * 70)
    print()
    
    # Check if data exists
    if not os.path.exists('data/real') or not os.path.exists('data/fake'):
        print("❌ ERROR: No training data found!")
        print()
        print("Please run first:")
        print("   python create_demo_dataset.py")
        print()
        return False
    
    # Count files
    real_files = list(Path('data/real').glob('*.wav'))
    fake_files = list(Path('data/fake').glob('*.wav'))
    
    print(f"📊 Dataset Status:")
    print(f"   Real audio: {len(real_files)} files")
    print(f"   Fake audio: {len(fake_files)} files")
    print()
    
    if len(real_files) < 10 or len(fake_files) < 10:
        print("⚠️  WARNING: Very small dataset. Training may not be accurate.")
        print("   Recommended: At least 50 files per class")
        print()
    
    # Training parameters
    epochs = 20
    batch_size = 16
    
    print(f"⚙️  Training Configuration:")
    print(f"   Model: ResNet-18")
    print(f"   Features: CQCC")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Device: CPU")
    print()
    
    print("🏃 Starting training...")
    print("   (This may take 5-15 minutes depending on your CPU)")
    print()
    print("-" * 70)
    
    # Run training
    cmd = [
        sys.executable,
        'main_train.py',
        '--data_folder', './data',
        '--feature_type', 'cqcc',
        '--model_type', 'resnet18',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', '0.001',
        '--device', 'cpu',
        '--export_model'
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("-" * 70)
        print()
        print("✅ Training completed successfully!")
        print()
        
        # Find the latest output folder
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            experiments = sorted(outputs_dir.glob('resnet18_cqcc_*'), key=os.path.getmtime)
            if experiments:
                latest_exp = experiments[-1]
                best_model = latest_exp / 'best_model.pth'
                
                if best_model.exists():
                    print(f"📁 Trained model saved to:")
                    print(f"   {best_model}")
                    print()
                    
                    # Copy to models folder
                    os.makedirs('models', exist_ok=True)
                    dest = Path('models/trained_model.pth')
                    shutil.copy(best_model, dest)
                    
                    print(f"✅ Model copied to: {dest}")
                    print()
                    print("🎯 NEXT STEPS:")
                    print()
                    print("1️⃣  Update app.py to use the trained model:")
                    print("   Change line ~55 from:")
                    print("      model_path = 'models/test_model.pth'")
                    print("   To:")
                    print("      model_path = 'models/trained_model.pth'")
                    print()
                    print("2️⃣  Restart the server:")
                    print("   python app.py")
                    print()
                    print("3️⃣  Test with your audio files!")
                    print("   Open: http://localhost:5000")
                    print()
                    
                    return True
        
        print("⚠️  Could not find trained model. Check outputs/ folder.")
        return False
        
    except subprocess.CalledProcessError as e:
        print()
        print("❌ Training failed!")
        print(f"   Error: {e}")
        return False
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user.")
        return False

if __name__ == '__main__':
    print()
    success = quick_train()
    print()
    
    if success:
        print("=" * 70)
        print("  🎉 SUCCESS! Your model is trained and ready to use!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("  ❌ Training did not complete successfully.")
        print("=" * 70)
    print()
