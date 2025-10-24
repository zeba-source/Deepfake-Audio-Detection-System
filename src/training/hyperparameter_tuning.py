"""Hyperparameter tuning using Optuna for deepfake audio detection."""
import os
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.resnet_model import ResNetDeepfakeDetector
from src.models.dataset import DeepfakeAudioDataset


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        n_trials: int = 50,
        device: str = 'auto',
        output_dir: str = 'results/hyperparameter_tuning'
    ):
        """Initialize hyperparameter tuner.
        
        Args:
            train_features: Training features array
            train_labels: Training labels array
            val_features: Validation features array
            val_labels: Validation labels array
            n_trials: Number of Optuna trials to run
            device: Device to use ('cuda', 'cpu', or 'auto')
            output_dir: Directory to save results
        """
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.n_trials = n_trials
        self.output_dir = output_dir
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track best trial
        self.best_trial = None
        self.best_value = -np.inf
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function to minimize/maximize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (to maximize)
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # Additional hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{self.n_trials}")
        print(f"{'='*70}")
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {learning_rate:.2e}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Dropout rate: {dropout_rate:.3f}")
        print(f"  - Weight decay: {weight_decay:.2e}")
        print(f"  - Optimizer: {optimizer_name}")
        
        # Create datasets
        train_dataset = DeepfakeAudioDataset(
            self.train_features,
            self.train_labels,
            train=True  # Enable augmentation
        )
        
        val_dataset = DeepfakeAudioDataset(
            self.val_features,
            self.val_labels,
            train=False  # No augmentation
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Initialize model with suggested dropout
        model = ResNetDeepfakeDetector(
            pretrained=False,
            dropout_p=dropout_rate
        ).to(self.device)
        
        # Setup optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:  # SGD
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training for limited epochs (for faster tuning)
        n_epochs = 10
        best_val_acc = 0.0
        
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]', leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/len(train_loader):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            print(f"  Epoch {epoch+1}: Train Acc={100.*train_correct/train_total:.2f}%, "
                  f"Val Acc={val_acc:.2f}%")
            
            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                print(f"  Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def run_study(
        self,
        study_name: str = "deepfake_detection",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None
    ) -> optuna.Study:
        """Run Optuna hyperparameter optimization study.
        
        Args:
            study_name: Name of the study
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
            
        Returns:
            Completed Optuna study object
        """
        # Default sampler and pruner
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)
        
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3
            )
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize validation accuracy
            sampler=sampler,
            pruner=pruner
        )
        
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Study name: {study_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Sampler: {sampler.__class__.__name__}")
        print(f"Pruner: {pruner.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_features)}")
        print(f"Validation samples: {len(self.val_features)}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Store best trial
        self.best_trial = study.best_trial
        self.best_value = study.best_value
        
        return study
    
    def save_results(
        self,
        study: optuna.Study,
        filename: str = 'best_hyperparameters.json'
    ) -> Dict[str, Any]:
        """Save best hyperparameters to JSON file.
        
        Args:
            study: Completed Optuna study
            filename: Output filename
            
        Returns:
            Dictionary containing best parameters and results
        """
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_validation_accuracy': float(best_value),
            'best_trial_number': study.best_trial.number,
            'best_hyperparameters': best_params,
            'study_statistics': {
                'n_trials': len(study.trials),
                'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            },
            'all_trials': []
        }
        
        # Add all trial results
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value if trial.value is not None else None,
                'params': trial.params,
                'state': trial.state.name
            }
            results['all_trials'].append(trial_data)
        
        # Save to JSON
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Best hyperparameters saved to: {output_path}")
        print(f"\n📊 Best Hyperparameters:")
        print(f"  Validation Accuracy: {best_value:.2f}%")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4e}" if value < 0.01 else f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
        
        return results
    
    def plot_optimization_results(
        self,
        study: optuna.Study,
        save_plots: bool = True
    ) -> None:
        """Create and save optimization visualization plots.
        
        Args:
            study: Completed Optuna study
            save_plots: Whether to save plots to disk
        """
        print(f"\n📈 Creating optimization plots...")
        
        # 1. Optimization History
        try:
            fig = plot_optimization_history(study)
            fig.update_layout(
                title='Optimization History',
                xaxis_title='Trial Number',
                yaxis_title='Validation Accuracy (%)'
            )
            if save_plots:
                plot_path = os.path.join(self.output_dir, 'optimization_history.html')
                fig.write_html(plot_path)
                print(f"  ✓ Saved: optimization_history.html")
        except Exception as e:
            print(f"  ⚠️ Could not create optimization history plot: {e}")
        
        # 2. Parameter Importances
        try:
            fig = plot_param_importances(study)
            fig.update_layout(title='Hyperparameter Importances')
            if save_plots:
                plot_path = os.path.join(self.output_dir, 'param_importances.html')
                fig.write_html(plot_path)
                print(f"  ✓ Saved: param_importances.html")
        except Exception as e:
            print(f"  ⚠️ Could not create parameter importances plot: {e}")
        
        # 3. Slice Plot
        try:
            fig = plot_slice(study)
            fig.update_layout(title='Parameter Slice Plot')
            if save_plots:
                plot_path = os.path.join(self.output_dir, 'param_slice.html')
                fig.write_html(plot_path)
                print(f"  ✓ Saved: param_slice.html")
        except Exception as e:
            print(f"  ⚠️ Could not create slice plot: {e}")
        
        # 4. Custom matplotlib plots
        self._create_custom_plots(study, save_plots)
    
    def _create_custom_plots(
        self,
        study: optuna.Study,
        save_plots: bool = True
    ) -> None:
        """Create custom matplotlib visualization plots.
        
        Args:
            study: Completed Optuna study
            save_plots: Whether to save plots
        """
        # Extract trial data
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            print("  ⚠️ No completed trials to plot")
            return
        
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Optimization History with best value line
        ax = axes[0, 0]
        ax.plot(trial_numbers, trial_values, 'o-', alpha=0.6, label='Trial Value')
        best_values = [max(trial_values[:i+1]) for i in range(len(trial_values))]
        ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
        ax.axhline(y=study.best_value, color='g', linestyle='--', label=f'Best: {study.best_value:.2f}%')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Learning Rate vs Accuracy
        ax = axes[0, 1]
        learning_rates = [t.params['learning_rate'] for t in completed_trials]
        ax.scatter(learning_rates, trial_values, alpha=0.6, s=50)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Learning Rate vs Accuracy')
        ax.grid(True, alpha=0.3)
        
        # 3. Batch Size vs Accuracy
        ax = axes[1, 0]
        batch_sizes = [t.params['batch_size'] for t in completed_trials]
        unique_batch_sizes = sorted(set(batch_sizes))
        batch_size_means = []
        batch_size_stds = []
        for bs in unique_batch_sizes:
            values = [trial_values[i] for i, b in enumerate(batch_sizes) if b == bs]
            batch_size_means.append(np.mean(values))
            batch_size_stds.append(np.std(values))
        
        ax.errorbar(unique_batch_sizes, batch_size_means, yerr=batch_size_stds, 
                    fmt='o-', capsize=5, capthick=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Batch Size vs Accuracy (with std)')
        ax.grid(True, alpha=0.3)
        
        # 4. Dropout Rate vs Accuracy
        ax = axes[1, 1]
        dropout_rates = [t.params['dropout_rate'] for t in completed_trials]
        ax.scatter(dropout_rates, trial_values, alpha=0.6, s=50)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Dropout Rate vs Accuracy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'optimization_analysis.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: optimization_analysis.png")
        
        plt.close()


def hyperparameter_search(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    n_trials: int = 50,
    output_dir: str = 'results/hyperparameter_tuning',
    device: str = 'auto'
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run hyperparameter search using Optuna.
    
    This function:
    1. Defines search space for learning rate, batch size, dropout, weight decay
    2. Runs specified number of trials
    3. Optimizes for validation accuracy
    4. Saves best parameters to JSON
    5. Plots optimization history
    
    Args:
        train_features: Training features array
        train_labels: Training labels array
        val_features: Validation features array
        val_labels: Validation labels array
        n_trials: Number of optimization trials (default: 50)
        output_dir: Directory to save results
        device: Device to use ('cuda', 'cpu', or 'auto')
        
    Returns:
        Tuple of (best_params_dict, study_object)
    """
    # Create tuner
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=n_trials,
        device=device,
        output_dir=output_dir
    )
    
    # Run study
    study = tuner.run_study()
    
    # Save results
    results = tuner.save_results(study)
    
    # Create plots
    tuner.plot_optimization_results(study, save_plots=True)
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {study.best_value:.2f}%")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"\nResults saved to: {output_dir}")
    
    return results, study


if __name__ == '__main__':
    # Example usage with synthetic data
    print("Hyperparameter Search Example")
    print("="*70)
    
    # Create synthetic data
    n_train = 200
    n_val = 50
    feature_dim = 20
    
    train_features = np.random.randn(n_train, feature_dim).astype(np.float32)
    train_labels = np.random.randint(0, 2, n_train)
    val_features = np.random.randn(n_val, feature_dim).astype(np.float32)
    val_labels = np.random.randint(0, 2, n_val)
    
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print("\nRunning hyperparameter search with 10 trials (demo)...\n")
    
    # Run search with fewer trials for demo
    results, study = hyperparameter_search(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=10,
        output_dir='results/hyperparameter_tuning',
        device='cpu'
    )
    
    print("\n✅ Demo complete! Check 'results/hyperparameter_tuning' for outputs.")
