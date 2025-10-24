"""
Main Training Script for Deepfake Audio Detection

Integrates all components:
- Data loading and preprocessing
- Feature extraction (mel, cqcc, mfcc)
- Model training (ResNet, VGG, Simple CNN)
- Data augmentation
- Hyperparameter tuning (optional)
- Model evaluation and testing
- Model export for deployment
- Performance profiling

Usage:
    python main_train.py --data_folder ./data/audio --feature_type mel --model_type resnet18
    python main_train.py --data_folder ./data --feature_type cqcc --model_type resnet34 --epochs 50 --batch_size 32
    python main_train.py --help
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import DeepfakeAudioDataset
from src.models.resnet_model import ResNetDeepfakeDetector
from src.training.train import train_model
from src.testing.test import test_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Deepfake Audio Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to folder containing audio files (with real/ and fake/ subfolders)')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Proportion of data for testing')
    
    # Feature extraction arguments
    parser.add_argument('--feature_type', type=str, default='mel',
                        choices=['mel', 'cqcc', 'mfcc'],
                        help='Type of audio features to extract')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands')
    parser.add_argument('--n_mfcc', type=int, default=40,
                        help='Number of MFCC coefficients')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'simple_cnn'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (real/fake)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Augmentation arguments
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.5,
                        help='Probability of applying augmentation')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0=main thread)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--model_save_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (default: auto-generated)')
    
    # Additional features
    parser.add_argument('--export_model', action='store_true',
                        help='Export model to TorchScript after training')
    parser.add_argument('--profile_model', action='store_true',
                        help='Profile model performance after training')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device


def create_model(model_type, num_classes):
    """Create model based on type."""
    print(f"\nCreating model: {model_type}")
    
    # For now, only ResNet is supported (the one we have implemented)
    if model_type in ['resnet18', 'resnet34', 'resnet50']:
        model = ResNetDeepfakeDetector(num_classes=num_classes)
    else:
        # Fallback to resnet if unsupported model requested
        print(f"⚠️  Model type '{model_type}' not fully implemented yet, using ResNetDeepfakeDetector")
        model = ResNetDeepfakeDetector(num_classes=num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    return model


def create_dataloaders(args):
    """Create data loaders for train, val, and test sets."""
    print(f"\nLoading data from: {args.data_folder}")
    
    # Create dataset
    dataset = DeepfakeAudioDataset(
        data_folder=args.data_folder,
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        augmentation=args.augmentation,
        augmentation_prob=args.aug_prob
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset sizes:")
    print(f"  Total: {total_size}")
    print(f"  Training: {train_size} ({args.train_split*100:.0f}%)")
    print(f"  Validation: {val_size} ({args.val_split*100:.0f}%)")
    print(f"  Testing: {test_size} ({args.test_split*100:.0f}%)")
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )
    
    return train_loader, val_loader, test_loader


def setup_optimizer(model, args):
    """Setup optimizer."""
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def setup_scheduler(optimizer, args):
    """Setup learning rate scheduler."""
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    return scheduler


def save_config(args, output_dir):
    """Save experiment configuration."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    model_save_dir = Path(args.model_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model_type}_{args.feature_type}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(args, experiment_dir)
    
    # Setup device
    device = setup_device(args.device)
    
    # Print header
    print("\n" + "="*80)
    print(" " * 20 + "DEEPFAKE AUDIO DETECTION - TRAINING")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.model_type}")
    print(f"Features: {args.feature_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Augmentation: {'Yes' if args.augmentation else 'No'}")
    print("="*80)
    
    # Create data loaders
    print("\n" + "="*80)
    print("[STEP 1/5] Creating Data Loaders")
    print("="*80)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(args)
        print(f"✓ Data loaders created successfully")
    except Exception as e:
        print(f"\n❌ Error creating data loaders: {e}")
        print("\nExpected data structure:")
        print("  data_folder/")
        print("    ├── real/")
        print("    │   ├── audio1.wav")
        print("    │   └── audio2.wav")
        print("    └── fake/")
        print("        ├── audio1.wav")
        print("        └── audio2.wav")
        return 1
    
    # Create model
    print("\n" + "="*80)
    print("[STEP 2/5] Creating Model")
    print("="*80)
    model = create_model(args.model_type, args.num_classes)
    model = model.to(device)
    print(f"✓ Model created and moved to {device}")
    
    # Setup training
    optimizer = setup_optimizer(model, args)
    scheduler = setup_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✓ Optimizer: {args.optimizer}")
    print(f"✓ Scheduler: {args.scheduler}")
    print(f"✓ Loss function: CrossEntropyLoss")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\n⟳ Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Train model
    print("\n" + "="*80)
    print("[STEP 3/5] Training Model")
    print("="*80)
    
    best_model_path = model_save_dir / f"{args.experiment_name}_best.pth"
    
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device,
            save_path=str(best_model_path)
        )
        
        # Save training history
        history_path = experiment_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n✓ Training history saved to: {history_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        checkpoint_path = model_save_dir / f"{args.experiment_name}_interrupted.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': start_epoch
        }, checkpoint_path)
        print(f"✓ Model saved to: {checkpoint_path}")
        return 1
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test model
    print("\n" + "="*80)
    print("[STEP 4/5] Evaluating on Test Set")
    print("="*80)
    
    try:
        test_results = test_model(model=model, test_loader=test_loader, device=device)
        
        print(f"\n{'='*40}")
        print("TEST RESULTS")
        print(f"{'='*40}")
        print(f"Accuracy:  {test_results['accuracy']*100:>6.2f}%")
        print(f"Precision: {test_results['precision']:>6.4f}")
        print(f"Recall:    {test_results['recall']:>6.4f}")
        print(f"F1-Score:  {test_results['f1']:>6.4f}")
        print(f"{'='*40}")
        
        # Save test results
        test_results_path = experiment_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n✓ Test results saved to: {test_results_path}")
        
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        test_results = None
    
    # Export model
    if args.export_model:
        print("\n" + "="*80)
        print("[STEP 5/5] Exporting Model to TorchScript")
        print("="*80)
        
        try:
            from src.deployment.export_model import export_model
            
            # Get input shape from first batch
            sample_batch = next(iter(train_loader))
            input_shape = (1,) + tuple(sample_batch[0].shape[1:])
            
            export_path = model_save_dir / f"{args.experiment_name}_scripted.pt"
            export_info = export_model(
                model_path=str(best_model_path),
                output_path=str(export_path),
                model_type=args.model_type.replace('18', '').replace('34', '').replace('50', '').replace('16', ''),
                input_shape=input_shape,
                use_trace=False,
                device=str(device)
            )
            
            print(f"\n✓ Model exported to: {export_path}")
            print(f"  Speedup: {export_info['performance']['speedup']:.2f}x")
            print(f"  Size: {export_info['file_sizes']['scripted_mb']:.2f} MB")
            
        except Exception as e:
            print(f"\n⚠️  Model export failed: {e}")
    
    # Profile model
    if args.profile_model:
        print("\n" + "="*80)
        print("[OPTIONAL] Profiling Model Performance")
        print("="*80)
        
        try:
            from src.profiling.profile_model import profile_model, generate_performance_report
            
            sample_batch = next(iter(train_loader))
            input_shape = (1,) + tuple(sample_batch[0].shape[1:])
            profiling_dir = experiment_dir / 'profiling'
            
            profile_results = profile_model(
                model=model,
                input_shape=input_shape,
                model_name=args.experiment_name,
                devices=['cpu', 'cuda'] if device.type == 'cuda' else ['cpu'],
                batch_sizes=[1, 4, 8, 16, 32],
                num_iterations=100,
                output_dir=str(profiling_dir)
            )
            
            generate_performance_report(
                profile_results,
                output_path=str(profiling_dir / f"{args.experiment_name}_report.txt")
            )
            
            print(f"\n✓ Profiling results saved to: {profiling_dir}")
            
        except Exception as e:
            print(f"\n⚠️  Profiling failed: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print(" " * 30 + "TRAINING COMPLETE!")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Best model: {best_model_path}")
    if test_results:
        print(f"Test accuracy: {test_results['accuracy']*100:.2f}%")
        print(f"F1-Score: {test_results['f1']:.4f}")
    print(f"\nOutput directory: {experiment_dir}")
    print("\nGenerated files:")
    print(f"  ✓ {best_model_path.name}")
    print(f"  ✓ config.json")
    print(f"  ✓ training_history.json")
    if test_results:
        print(f"  ✓ test_results.json")
    if args.export_model:
        print(f"  ✓ {args.experiment_name}_scripted.pt")
    if args.profile_model:
        print(f"  ✓ profiling/ (performance metrics)")
    
    print("\n" + "="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
