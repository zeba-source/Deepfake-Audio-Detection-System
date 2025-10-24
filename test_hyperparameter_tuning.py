"""Demo script for hyperparameter tuning with Optuna."""
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.training.hyperparameter_tuning import hyperparameter_search


def create_synthetic_dataset(n_samples=500, feature_dim=20):
    """Create synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        feature_dim: Feature dimensionality
        
    Returns:
        Tuple of (features, labels)
    """
    # Generate random features
    features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    
    # Generate labels based on simple pattern (for demonstration)
    # Real samples: positive values in first feature
    # Fake samples: negative values in first feature
    labels = (features[:, 0] > 0).astype(int)
    
    # Add some noise to labels
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    labels[noise_idx] = 1 - labels[noise_idx]
    
    return features, labels


def main():
    """Run hyperparameter tuning demo."""
    print("="*70)
    print("HYPERPARAMETER TUNING DEMO")
    print("="*70)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    features, labels = create_synthetic_dataset(n_samples=500, feature_dim=20)
    print(f"   Total samples: {len(features)}")
    print(f"   Feature dimension: {features.shape[1]}")
    print(f"   Real samples: {np.sum(labels == 0)}")
    print(f"   Fake samples: {np.sum(labels == 1)}")
    
    # Split into train and validation
    print("\n2. Splitting into train/validation sets...")
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"   Training samples: {len(train_features)}")
    print(f"   Validation samples: {len(val_features)}")
    
    # Run hyperparameter search
    print("\n3. Running hyperparameter optimization...")
    print("   Note: Using 20 trials for demo (recommended: 50+ for real use)")
    print("   Each trial trains for 10 epochs\n")
    
    results, study = hyperparameter_search(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=20,  # Use fewer trials for demo
        output_dir='results/hyperparameter_tuning',
        device='cpu'  # Use CPU for demo
    )
    
    # Display summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    print(f"\n📊 Best Results:")
    print(f"   Validation Accuracy: {results['best_validation_accuracy']:.2f}%")
    print(f"   Best Trial Number: {results['best_trial_number']}")
    
    print(f"\n⚙️  Best Hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        if isinstance(value, float):
            if value < 0.01:
                print(f"   {param}: {value:.4e}")
            else:
                print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    print(f"\n📈 Trial Statistics:")
    stats = results['study_statistics']
    print(f"   Total trials: {stats['n_trials']}")
    print(f"   Completed: {stats['n_completed_trials']}")
    print(f"   Pruned: {stats['n_pruned_trials']}")
    
    print(f"\n📁 Output Files:")
    output_dir = 'results/hyperparameter_tuning'
    print(f"   {output_dir}/")
    print(f"   ├── best_hyperparameters.json")
    print(f"   ├── optimization_history.html")
    print(f"   ├── param_importances.html")
    print(f"   ├── param_slice.html")
    print(f"   └── optimization_analysis.png")
    
    print(f"\n✅ Demo complete!")
    print(f"\nTo view interactive plots, open the HTML files in a browser:")
    print(f"   - optimization_history.html: Trial-by-trial progress")
    print(f"   - param_importances.html: Which parameters matter most")
    print(f"   - param_slice.html: Parameter effects on accuracy")
    
    # Example: Use best parameters in training
    print(f"\n💡 Next Steps:")
    print(f"   Use these best parameters in main_train.py:")
    best_params = results['best_hyperparameters']
    print(f"""
   python main_train.py \\
       --lr {best_params.get('learning_rate', 0.001):.6f} \\
       --batch_size {best_params.get('batch_size', 32)} \\
       --dropout {best_params.get('dropout_rate', 0.5):.3f} \\
       --weight_decay {best_params.get('weight_decay', 0.0001):.6f}
   """)


if __name__ == '__main__':
    main()
