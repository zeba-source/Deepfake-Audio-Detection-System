# Hyperparameter Tuning Guide

## Overview
Automated hyperparameter optimization using Optuna to find optimal training configurations for deepfake audio detection models.

## Features

### Search Space
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Batch Size**: [16, 32, 64]
- **Dropout Rate**: 0.3 to 0.7
- **Weight Decay**: 1e-5 to 1e-3 (log scale)
- **Optimizer**: ['Adam', 'AdamW', 'SGD']

### Optimization Features
- ✅ TPE (Tree-structured Parzen Estimator) Sampler
- ✅ Median Pruning for early stopping of unpromising trials
- ✅ 10 epochs per trial for fast evaluation
- ✅ Maximizes validation accuracy
- ✅ Saves best parameters to JSON
- ✅ Generates interactive HTML plots
- ✅ Creates matplotlib summary visualizations

## Quick Start

### Basic Usage

```python
from src.training.hyperparameter_tuning import hyperparameter_search
import numpy as np

# Load your data
train_features = np.load('data/processed/train_features.npy')
train_labels = np.load('data/processed/train_labels.npy')
val_features = np.load('data/processed/val_features.npy')
val_labels = np.load('data/processed/val_labels.npy')

# Run hyperparameter search
results, study = hyperparameter_search(
    train_features=train_features,
    train_labels=train_labels,
    val_features=val_features,
    val_labels=val_labels,
    n_trials=50,
    output_dir='results/hyperparameter_tuning',
    device='auto'  # 'cuda', 'cpu', or 'auto'
)

# Access best parameters
best_params = results['best_hyperparameters']
best_accuracy = results['best_validation_accuracy']
print(f"Best validation accuracy: {best_accuracy:.2f}%")
```

### Using HyperparameterTuner Class

```python
from src.training.hyperparameter_tuning import HyperparameterTuner
import optuna

# Create tuner
tuner = HyperparameterTuner(
    train_features=train_features,
    train_labels=train_labels,
    val_features=val_features,
    val_labels=val_labels,
    n_trials=50,
    device='cuda',
    output_dir='results/hyperparameter_tuning'
)

# Run study with custom sampler
sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

study = tuner.run_study(
    study_name="deepfake_detection",
    sampler=sampler,
    pruner=pruner
)

# Save results
results = tuner.save_results(study)

# Create plots
tuner.plot_optimization_results(study, save_plots=True)
```

## Command Line Usage

### Run Demo Script
```bash
python test_hyperparameter_tuning.py
```

### Run with Custom Data
```python
# Create your own script
import numpy as np
from src.training.hyperparameter_tuning import hyperparameter_search

# Load data
train_features = np.load('your_train_features.npy')
train_labels = np.load('your_train_labels.npy')
val_features = np.load('your_val_features.npy')
val_labels = np.load('your_val_labels.npy')

# Run tuning
results, study = hyperparameter_search(
    train_features, train_labels,
    val_features, val_labels,
    n_trials=100,
    output_dir='results/my_tuning'
)
```

## Output Files

### 1. best_hyperparameters.json
Complete results in JSON format:
```json
{
    "best_validation_accuracy": 83.0,
    "best_trial_number": 15,
    "best_hyperparameters": {
        "learning_rate": 0.003722,
        "batch_size": 32,
        "dropout_rate": 0.459,
        "weight_decay": 0.000755,
        "optimizer": "Adam"
    },
    "study_statistics": {
        "n_trials": 20,
        "n_completed_trials": 11,
        "n_pruned_trials": 9
    },
    "all_trials": [...]
}
```

### 2. Interactive HTML Plots

#### optimization_history.html
- Trial-by-trial progress
- Shows value (validation accuracy) over trials
- Identifies best trial

#### param_importances.html
- Shows which hyperparameters matter most
- Based on fANOVA (functional ANOVA)
- Helps understand parameter impact

#### param_slice.html
- Visualizes effect of each parameter
- Shows relationship between parameter values and accuracy
- Useful for understanding optimal ranges

### 3. optimization_analysis.png
4-panel matplotlib visualization:
- **Top Left**: Optimization history with best value line
- **Top Right**: Learning rate vs accuracy (log scale)
- **Bottom Left**: Batch size vs accuracy with error bars
- **Bottom Right**: Dropout rate vs accuracy

## Integration with Training Pipeline

### Step 1: Run Hyperparameter Search
```python
from src.training.hyperparameter_tuning import hyperparameter_search
from src.features.feature_extractor import FeatureExtractor
from src.models.dataloaders import create_dataloaders

# Extract features
extractor = FeatureExtractor(feature_type='mel', sample_rate=16000)
features, labels, df = extractor.process_dataset(
    data_folder='data/raw',
    output_folder='data/processed'
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    features, labels,
    batch_size=32,  # Will be optimized
    train_split=0.8,
    val_split=0.1
)

# Get train/val data
train_features = features[train_loader.dataset.indices]
train_labels = labels[train_loader.dataset.indices]
val_features = features[val_loader.dataset.indices]
val_labels = labels[val_loader.dataset.indices]

# Run tuning
results, study = hyperparameter_search(
    train_features, train_labels,
    val_features, val_labels,
    n_trials=50
)
```

### Step 2: Use Best Parameters in Training
```python
import json
from src.training.train import train_model
from config.config import TrainingConfig

# Load best parameters
with open('results/hyperparameter_tuning/best_hyperparameters.json', 'r') as f:
    tuning_results = json.load(f)

best_params = tuning_results['best_hyperparameters']

# Create config with optimized parameters
config = TrainingConfig(
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    weight_decay=best_params['weight_decay'],
    epochs=100  # Use more epochs for final training
)

# Create model with optimized dropout
from src.models.resnet_model import ResNetDeepfakeDetector
model = ResNetDeepfakeDetector(dropout_p=best_params['dropout_rate'])

# Train with optimized parameters
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    save_dir='models/optimized'
)
```

### Step 3: Use in main_train.py
```bash
# Extract best parameters
python -c "
import json
with open('results/hyperparameter_tuning/best_hyperparameters.json', 'r') as f:
    params = json.load(f)['best_hyperparameters']
print(f'--lr {params[\"learning_rate\"]:.6f} --batch_size {params[\"batch_size\"]} --dropout {params[\"dropout_rate\"]:.3f} --weight_decay {params[\"weight_decay\"]:.6f}')
"

# Use parameters
python main_train.py \
    --data_folder data/raw \
    --output_folder data/processed \
    --model_save_dir models/optimized \
    --lr 0.003722 \
    --batch_size 32 \
    --dropout 0.459 \
    --weight_decay 0.000755 \
    --epochs 100
```

## Advanced Usage

### Custom Objective Function

```python
from src.training.hyperparameter_tuning import HyperparameterTuner
import optuna

class CustomTuner(HyperparameterTuner):
    def objective(self, trial):
        # Add custom hyperparameters
        scheduler_type = trial.suggest_categorical('scheduler', ['StepLR', 'ReduceLROnPlateau'])
        
        # Call parent objective
        val_acc = super().objective(trial)
        
        return val_acc

# Use custom tuner
tuner = CustomTuner(train_features, train_labels, val_features, val_labels)
study = tuner.run_study()
```

### Multi-Objective Optimization

```python
import optuna

def multi_objective(trial):
    # Optimize both accuracy and model size
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # ... train model ...
    
    val_accuracy = # ... get validation accuracy ...
    model_size = # ... get model parameter count ...
    
    return val_accuracy, -model_size  # Maximize accuracy, minimize size

# Create multi-objective study
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=50)

# Get Pareto front
pareto_trials = study.best_trials
```

### Resume Interrupted Study

```python
import optuna

# Save study to database
storage = optuna.storages.RDBStorage('sqlite:///optuna.db')
study = optuna.create_study(
    study_name='deepfake_tuning',
    storage=storage,
    load_if_exists=True  # Resume if exists
)

# Continue optimization
study.optimize(objective, n_trials=50)
```

## Best Practices

### 1. Number of Trials
- **Quick test**: 10-20 trials
- **Standard**: 50-100 trials
- **Thorough**: 200+ trials

### 2. Training Epochs per Trial
- Current: 10 epochs (fast evaluation)
- Increase for more accurate results
- Trade-off: speed vs accuracy

### 3. Pruning Strategy
```python
# Aggressive pruning (faster)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,
    n_warmup_steps=2
)

# Conservative pruning (more accurate)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=5
)
```

### 4. Search Space Refinement
After initial search, narrow ranges:
```python
# Initial broad search
learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

# Refined search (if initial best was ~1e-3)
learning_rate = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
```

### 5. Validation Data
- Use separate validation set (not test set)
- Ensure representative of full dataset
- Consider stratified sampling for imbalanced data

## Troubleshooting

### Issue: All trials get pruned
**Solution**: Reduce pruning aggressiveness
```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,  # Increase
    n_warmup_steps=5       # Increase
)
```

### Issue: Very slow optimization
**Solutions**:
1. Reduce epochs per trial
2. Use smaller training subset
3. Increase batch size
4. Use GPU if available
5. Enable more aggressive pruning

### Issue: Best parameters don't generalize
**Solutions**:
1. Increase validation set size
2. Use cross-validation
3. Add regularization to search space
4. Increase number of trials

### Issue: Out of memory
**Solutions**:
1. Reduce batch size range
2. Use gradient checkpointing
3. Clear CUDA cache between trials
4. Use CPU for tuning

## Performance Tips

### 1. Use GPU if Available
```python
results, study = hyperparameter_search(
    ...,
    device='cuda'  # Much faster than CPU
)
```

### 2. Parallel Optimization
```python
# Run multiple workers in parallel
study.optimize(objective, n_trials=50, n_jobs=4)
```

### 3. Use Smaller Subset for Quick Tuning
```python
# Use 20% of data for fast tuning
subset_size = len(train_features) // 5
indices = np.random.choice(len(train_features), subset_size, replace=False)

results, study = hyperparameter_search(
    train_features[indices],
    train_labels[indices],
    val_features,
    val_labels,
    n_trials=100
)
```

## Example Results

### Demo Output (20 trials)
```
Best validation accuracy: 83.00%
Best hyperparameters:
  - learning_rate: 0.003722
  - batch_size: 32
  - dropout_rate: 0.459
  - weight_decay: 0.000755
  - optimizer: Adam

Trial statistics:
  - Completed: 11
  - Pruned: 9
  - Time: ~7 minutes
```

### Typical Improvements
- Baseline (default params): 70-75% accuracy
- After tuning: 78-85% accuracy
- Improvement: +5-10% absolute accuracy

## See Also

- `src/training/train.py` - Training loop implementation
- `config/config.py` - Training configuration
- `main_train.py` - Complete training pipeline
- Optuna documentation: https://optuna.readthedocs.io/
