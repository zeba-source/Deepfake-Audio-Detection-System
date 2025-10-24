"""
Enhanced Training Script with Class Weighting
Improves handling of imbalanced datasets
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(data_folder='./data'):
    """
    Calculate class weights based on dataset distribution.
    Helps handle imbalanced datasets by weighting minority class higher.
    """
    
    print("\n" + "="*70)
    print("  ⚖️  CALCULATING CLASS WEIGHTS")
    print("="*70)
    
    # Count samples in each class
    real_folder = Path(data_folder) / 'real'
    fake_folder = Path(data_folder) / 'fake'
    
    num_real = len(list(real_folder.glob('*.wav')))
    num_fake = len(list(fake_folder.glob('*.wav')))
    total = num_real + num_fake
    
    print(f"\n📊 Dataset Distribution:")
    print(f"   Real samples: {num_real} ({num_real/total*100:.1f}%)")
    print(f"   Fake samples: {num_fake} ({num_fake/total*100:.1f}%)")
    print(f"   Total: {total}")
    
    # Create labels array (0=FAKE, 1=REAL)
    labels = np.array([0]*num_fake + [1]*num_real)
    
    # Compute class weights
    # sklearn computes weights as: n_samples / (n_classes * n_samples_per_class)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    # Convert to torch tensor
    weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"\n⚖️  Computed Class Weights:")
    print(f"   Class 0 (FAKE): {class_weights[0]:.4f}")
    print(f"   Class 1 (REAL): {class_weights[1]:.4f}")
    print(f"   Weight ratio: {class_weights[0]/class_weights[1]:.2f}:1")
    
    # Interpretation
    print(f"\n💡 Weight Interpretation:")
    if abs(class_weights[0] - class_weights[1]) < 0.1:
        print(f"   ✅ Dataset is well balanced - weights are equal")
        print(f"   → No special weighting needed")
    elif class_weights[0] > class_weights[1]:
        ratio = class_weights[0] / class_weights[1]
        print(f"   → FAKE class is weighted {ratio:.2f}x higher")
        print(f"   → Compensating for fewer FAKE samples")
        print(f"   → Model will pay more attention to FAKE samples")
    else:
        ratio = class_weights[1] / class_weights[0]
        print(f"   → REAL class is weighted {ratio:.2f}x higher")
        print(f"   → Compensating for fewer REAL samples")
        print(f"   → Model will pay more attention to REAL samples")
    
    # Show impact
    if total < 100:
        print(f"\n⚠️  Warning: Small dataset ({total} samples)")
        print(f"   → Class weights help, but more data is better")
    
    print("="*70)
    print()
    
    return weights_tensor, class_weights


def create_weighted_loss(class_weights_tensor, device='cpu'):
    """
    Create CrossEntropyLoss with class weights.
    
    Args:
        class_weights_tensor: Tensor of class weights [weight_class0, weight_class1]
        device: Device to move weights to
    
    Returns:
        Weighted CrossEntropyLoss criterion
    """
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    print(f"✅ Created weighted loss function")
    print(f"   Weights: {class_weights_tensor.cpu().numpy()}")
    
    return criterion


def demonstrate_class_weighting():
    """
    Demonstrate how class weights affect loss calculation.
    """
    
    print("\n" + "="*70)
    print("  📚 CLASS WEIGHTING DEMONSTRATION")
    print("="*70)
    
    # Calculate weights
    weights_tensor, weights = calculate_class_weights('./data')
    
    # Create example predictions
    print(f"\n🧪 Example Loss Calculation:")
    print(f"   Scenario: Model predicts wrong class for both REAL and FAKE")
    
    # Without weights
    criterion_unweighted = nn.CrossEntropyLoss()
    
    # Example: Model predicts FAKE (class 0) but actual is REAL (class 1)
    pred_fake = torch.tensor([[2.0, -1.0]])  # Strong prediction for class 0
    target_real = torch.tensor([1])  # Actual label is class 1
    
    loss_unweighted_1 = criterion_unweighted(pred_fake, target_real)
    
    # Example: Model predicts REAL (class 1) but actual is FAKE (class 0)
    pred_real = torch.tensor([[-1.0, 2.0]])  # Strong prediction for class 1
    target_fake = torch.tensor([0])  # Actual label is class 0
    
    loss_unweighted_2 = criterion_unweighted(pred_real, target_fake)
    
    print(f"\n   WITHOUT class weights:")
    print(f"   • Wrong REAL prediction loss: {loss_unweighted_1.item():.4f}")
    print(f"   • Wrong FAKE prediction loss: {loss_unweighted_2.item():.4f}")
    print(f"   • Both errors treated equally")
    
    # With weights
    criterion_weighted = nn.CrossEntropyLoss(weight=weights_tensor)
    
    loss_weighted_1 = criterion_weighted(pred_fake, target_real)
    loss_weighted_2 = criterion_weighted(pred_real, target_fake)
    
    print(f"\n   WITH class weights:")
    print(f"   • Wrong REAL prediction loss: {loss_weighted_1.item():.4f}")
    print(f"   • Wrong FAKE prediction loss: {loss_weighted_2.item():.4f}")
    
    if weights[1] > weights[0]:
        print(f"   • REAL errors penalized {weights[1]/weights[0]:.2f}x more")
    elif weights[0] > weights[1]:
        print(f"   • FAKE errors penalized {weights[0]/weights[1]:.2f}x more")
    else:
        print(f"   • Both errors weighted equally (balanced dataset)")
    
    print(f"\n💡 Impact on Training:")
    print(f"   → Model learns to focus on minority class")
    print(f"   → Prevents bias towards majority class")
    print(f"   → Improves balanced accuracy")
    
    print("="*70)
    print()
    
    return criterion_weighted


def usage_example():
    """
    Show how to integrate class weights into training loop.
    """
    
    print("\n" + "="*70)
    print("  📖 USAGE IN TRAINING SCRIPT")
    print("="*70)
    
    print("""
    
    # In your training script, add this code:
    
    # ===== STEP 1: Calculate class weights =====
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Count your samples
    num_real = len(list(Path('data/real').glob('*.wav')))
    num_fake = len(list(Path('data/fake').glob('*.wav')))
    
    # Create labels array
    labels = np.array([0]*num_fake + [1]*num_real)
    
    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: Real={class_weights[1]:.4f}, Fake={class_weights[0]:.4f}")
    
    
    # ===== STEP 2: Create weighted loss =====
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    
    # ===== STEP 3: Use in training loop (no changes needed!) =====
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # Loss automatically applies class weights
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
    
    
    # That's it! The loss function handles the weighting automatically.
    """)
    
    print("="*70)
    print()


if __name__ == '__main__':
    print("\n🎯 CLASS WEIGHTING ANALYSIS AND DEMONSTRATION")
    
    # Calculate weights for current dataset
    weights_tensor, weights = calculate_class_weights('./data')
    
    # Demonstrate effect
    criterion = demonstrate_class_weighting()
    
    # Show usage
    usage_example()
    
    print("\n✅ Analysis complete!")
    print(f"\n💡 Key Takeaway:")
    print(f"   Class weights help handle imbalanced datasets by penalizing")
    print(f"   mistakes on minority class more heavily.")
    print(f"\n   Your dataset: Real={weights[1]:.4f}, Fake={weights[0]:.4f}")
    print()
