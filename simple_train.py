"""
Simple All-in-One Training Script
Extracts features AND trains the model in one go!
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import librosa
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.dataset import DeepfakeAudioDataset
from src.models.resnet_model import ResNetDeepfakeDetector

def extract_cqcc_features(audio_path, sr=16000):
    """Extract CQCC features from audio file"""
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=5.0)
        
        # Compute CQT (Constant-Q Transform)
        cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=512, n_bins=84))
        
        # Convert to dB scale
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Ensure consistent shape (pad or truncate)
        target_frames = 256
        if cqt_db.shape[1] < target_frames:
            pad_width = target_frames - cqt_db.shape[1]
            cqt_db = np.pad(cqt_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            cqt_db = cqt_db[:, :target_frames]
        
        return cqt_db.T  # Shape: (time, freq)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_audio_dataset(data_folder):
    """Load all audio files and extract features"""
    print("\n" + "="*70)
    print("  📊 LOADING AND EXTRACTING FEATURES")
    print("="*70)
    
    real_folder = Path(data_folder) / 'real'
    fake_folder = Path(data_folder) / 'fake'
    
    features_list = []
    labels_list = []
    
    # Load REAL audio files
    print("\n📂 Processing REAL audio files...")
    real_files = list(real_folder.glob('*.wav'))
    for audio_file in tqdm(real_files, desc="Real files"):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            features_list.append(features)
            labels_list.append(1)  # 1 = REAL
    
    print(f"✅ Loaded {len([l for l in labels_list if l == 1])} REAL files")
    
    # Load FAKE audio files
    print("\n📂 Processing FAKE audio files...")
    fake_files = list(fake_folder.glob('*.wav'))
    for audio_file in tqdm(fake_files, desc="Fake files"):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            features_list.append(features)
            labels_list.append(0)  # 0 = FAKE
    
    print(f"✅ Loaded {len([l for l in labels_list if l == 0])} FAKE files")
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(features)}")
    print(f"   Feature shape: {features[0].shape}")
    print(f"   Real samples: {np.sum(labels == 1)}")
    print(f"   Fake samples: {np.sum(labels == 0)}")
    
    return features, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model with per-class accuracy tracking"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = {0: 0, 1: 0}  # 0=FAKE, 1=REAL
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Track per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate per-class accuracies
    overall_acc = 100. * correct / total
    fake_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    real_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    return total_loss / len(dataloader), overall_acc, fake_acc, real_acc

def main():
    print("\n" + "="*70)
    print("  🚀 SIMPLE DEEPFAKE DETECTION TRAINING")
    print("="*70)
    
    # Configuration
    data_folder = './data'
    epochs = 20
    batch_size = 16
    learning_rate = 0.001
    device = 'cpu'
    
    print(f"\n⚙️  Configuration:")
    print(f"   Data folder: {data_folder}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    
    # Load and extract features
    features, labels = load_audio_dataset(data_folder)
    
    # Create dataset
    print("\n" + "="*70)
    print("  📦 CREATING DATASETS")
    print("="*70)
    
    dataset = DeepfakeAudioDataset(features, labels, train=True)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Split sizes:")
    print(f"   Training: {train_size} (70%)")
    print(f"   Validation: {val_size} (15%)")
    print(f"   Test: {test_size} (15%)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("\n" + "="*70)
    print("  🧠 CREATING MODEL")
    print("="*70)
    
    model = ResNetDeepfakeDetector()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created!")
    print(f"   Architecture: ResNet-18")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Calculate class weights
    print("\n" + "="*70)
    print("  ⚖️  CALCULATING CLASS WEIGHTS")
    print("="*70)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n⚖️  Class Weights:")
    print(f"   Class 0 (FAKE): {class_weights[0]:.4f}")
    print(f"   Class 1 (REAL): {class_weights[1]:.4f}")
    
    if abs(class_weights[0] - class_weights[1]) < 0.1:
        print(f"   ✅ Balanced dataset - equal weights")
    elif class_weights[0] > class_weights[1]:
        print(f"   → FAKE class weighted {class_weights[0]/class_weights[1]:.2f}x higher")
    else:
        print(f"   → REAL class weighted {class_weights[1]/class_weights[0]:.2f}x higher")
    
    # Setup training with weighted loss
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n✅ Using weighted CrossEntropyLoss")
    
    # Training loop
    print("\n" + "="*70)
    print("  🏋️  TRAINING")
    print("="*70)
    
    best_val_acc = 0
    best_model_state = None
    
    # For logging
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_fake_acc': [],
        'val_real_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_fake_acc, val_real_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_fake_acc'].append(val_fake_acc)
        training_history['val_real_acc'].append(val_real_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: Overall={val_acc:.2f}%, Real={val_real_acc:.2f}%, Fake={val_fake_acc:.2f}%")
        
        # Early warning for poor fake class performance
        if epoch >= 4 and val_fake_acc < 30:
            print(f"  ⚠️  WARNING: Fake class accuracy is very low ({val_fake_acc:.2f}%)")
            print(f"      Model may be ignoring the fake class!")
        
        # Warning for class imbalance in predictions
        if abs(val_real_acc - val_fake_acc) > 30:
            print(f"  ⚠️  WARNING: Large accuracy gap between classes!")
            print(f"      Real: {val_real_acc:.2f}% vs Fake: {val_fake_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ⭐ New best validation accuracy: {val_acc:.2f}%")
    
    # Test with best model
    print("\n" + "="*70)
    print("  🧪 TESTING BEST MODEL")
    print("="*70)
    
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_fake_acc, test_real_acc = validate(model, test_loader, criterion, device)
    
    print(f"\n✅ Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Overall Accuracy: {test_acc:.2f}%")
    print(f"   Real Accuracy: {test_real_acc:.2f}%")
    print(f"   Fake Accuracy: {test_fake_acc:.2f}%")
    
    # Check for class imbalance in final results
    if abs(test_real_acc - test_fake_acc) > 20:
        print(f"\n⚠️  WARNING: Significant accuracy imbalance detected!")
        print(f"   Model performs much better on {'REAL' if test_real_acc > test_fake_acc else 'FAKE'} class")
        print(f"   Consider retraining with adjusted class weights")
    
    # Save model
    print("\n" + "="*70)
    print("  💾 SAVING MODEL")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/trained_model.pth'
    
    torch.save({
        'model_state_dict': best_model_state,
        'test_accuracy': test_acc,
        'feature_type': 'cqcc',
        'input_shape': features[0].shape,
        'trained_date': datetime.now().isoformat()
    }, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    # Save training info with per-class metrics
    info = {
        'test_accuracy': float(test_acc),
        'test_real_accuracy': float(test_real_acc),
        'test_fake_accuracy': float(test_fake_acc),
        'best_val_accuracy': float(best_val_acc),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'class_weights': class_weights.tolist(),
        'training_history': training_history,
        'trained_date': datetime.now().isoformat()
    }
    
    with open('models/training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create visualization of training history
    print("\n📊 Creating training visualization...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = training_history['epoch']
    
    # Loss curves
    axes[0, 0].plot(epochs_range, training_history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, training_history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Overall accuracy
    axes[0, 1].plot(epochs_range, training_history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs_range, training_history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 1].set_title('Overall Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Per-class validation accuracy
    axes[1, 0].plot(epochs_range, training_history['val_real_acc'], 'g-o', label='Real (Human)', linewidth=2, markersize=6)
    axes[1, 0].plot(epochs_range, training_history['val_fake_acc'], 'r-s', label='Fake (AI)', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=30, color='orange', linestyle='--', linewidth=2, label='Warning Threshold (30%)')
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1, 0].set_title('Per-Class Validation Accuracy', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Final test results bar chart
    test_results = ['Overall', 'Real', 'Fake']
    test_scores = [test_acc, test_real_acc, test_fake_acc]
    colors = ['blue', 'green', 'red']
    
    bars = axes[1, 1].bar(test_results, test_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1, 1].set_title('Final Test Accuracy', fontweight='bold')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, test_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('Training Progress and Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: models/training_curves.png")
    
    print("\n" + "="*70)
    print("  ✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy (Overall): {test_acc:.2f}%")
    print(f"   Test Accuracy (Real): {test_real_acc:.2f}%")
    print(f"   Test Accuracy (Fake): {test_fake_acc:.2f}%")
    print(f"   Model saved to: {model_path}")
    print(f"   Training curves: models/training_curves.png")
    print()
    print("🎯 NEXT STEPS:")
    print("   1. Update app.py line 55:")
    print("      model_path = 'models/trained_model.pth'")
    print()
    print("   2. Restart the server:")
    print("      python app.py")
    print()
    print("   3. Open browser:")
    print("      http://localhost:5000")
    print()
    print("="*70)
    print()

if __name__ == '__main__':
    main()
