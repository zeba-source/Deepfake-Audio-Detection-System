"""
Fix the label swap issue by swapping folder contents
"""

import shutil
from pathlib import Path
import os

def swap_dataset_folders():
    """Swap the contents of data/real and data/fake folders"""
    
    print("\n" + "="*70)
    print("  🔧 FIXING LABEL SWAP ISSUE")
    print("="*70)
    
    real_folder = Path('data/real')
    fake_folder = Path('data/fake')
    temp_folder = Path('data/temp_swap')
    
    # Safety check
    if not real_folder.exists() or not fake_folder.exists():
        print("❌ Error: data/real or data/fake folder not found!")
        return False
    
    # Count files before swap
    real_files = list(real_folder.glob('*.wav'))
    fake_files = list(fake_folder.glob('*.wav'))
    
    print(f"\n📊 Current state:")
    print(f"   data/real/: {len(real_files)} files (currently labeled as REAL)")
    print(f"   data/fake/: {len(fake_files)} files (currently labeled as FAKE)")
    
    print(f"\n⚠️  Problem detected:")
    print(f"   Files in data/real/ are actually AI-generated (should be FAKE)")
    print(f"   Files in data/fake/ are actually human voices (should be REAL)")
    
    print(f"\n🔄 Swapping folder contents...")
    
    try:
        # Step 1: Move real folder to temp
        print(f"   1️⃣  Moving data/real/ → data/temp_swap/")
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        shutil.move(str(real_folder), str(temp_folder))
        
        # Step 2: Move fake folder to real
        print(f"   2️⃣  Moving data/fake/ → data/real/")
        shutil.move(str(fake_folder), str(real_folder))
        
        # Step 3: Move temp folder to fake
        print(f"   3️⃣  Moving data/temp_swap/ → data/fake/")
        shutil.move(str(temp_folder), str(fake_folder))
        
        print(f"\n✅ Swap completed successfully!")
        
        # Verify the swap
        new_real_files = list(real_folder.glob('*.wav'))
        new_fake_files = list(fake_folder.glob('*.wav'))
        
        print(f"\n📊 After swap:")
        print(f"   data/real/: {len(new_real_files)} files (NOW contains human voices)")
        print(f"   data/fake/: {len(new_fake_files)} files (NOW contains AI voices)")
        
        print(f"\n🎯 Sample files:")
        print(f"   data/real/: {[f.name for f in new_real_files[:3]]}")
        print(f"   data/fake/: {[f.name for f in new_fake_files[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during swap: {e}")
        return False

if __name__ == '__main__':
    success = swap_dataset_folders()
    
    if success:
        print(f"\n{'='*70}")
        print(f"  ✅ FIX APPLIED!")
        print(f"{'='*70}")
        print(f"\n🔄 NEXT STEPS:")
        print(f"   1. ❌ The current trained model has learned the WRONG labels")
        print(f"   2. ✅ Dataset folders are now correctly labeled")
        print(f"   3. 🔄 MUST RETRAIN the model with: python simple_train.py")
        print(f"   4. ✅ After retraining, predictions will be correct")
        print(f"\n{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"  ❌ FIX FAILED")
        print(f"{'='*70}\n")
