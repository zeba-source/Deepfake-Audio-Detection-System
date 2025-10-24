"""Unit tests for hyperparameter tuning module."""
import os
import tempfile
import shutil
import json

import pytest
import numpy as np
import optuna

from src.training.hyperparameter_tuning import (
    HyperparameterTuner,
    hyperparameter_search
)


@pytest.fixture
def sample_data():
    """Create sample training and validation data."""
    n_train = 100
    n_val = 20
    feature_dim = 20
    
    train_features = np.random.randn(n_train, feature_dim).astype(np.float32)
    train_labels = np.random.randint(0, 2, n_train)
    val_features = np.random.randn(n_val, feature_dim).astype(np.float32)
    val_labels = np.random.randint(0, 2, n_val)
    
    return train_features, train_labels, val_features, val_labels


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


# Test 1: HyperparameterTuner initialization
def test_tuner_initialization(sample_data, temp_output_dir):
    """Test that HyperparameterTuner initializes correctly."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=5,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    assert tuner.n_trials == 5
    assert tuner.device.type == 'cpu'
    assert tuner.output_dir == temp_output_dir
    assert len(tuner.train_features) == 100
    assert len(tuner.val_features) == 20


# Test 2: Objective function runs without errors
def test_objective_function(sample_data, temp_output_dir):
    """Test that objective function completes successfully."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=1,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    # Create a trial manually
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    
    # Run objective
    value = tuner.objective(trial)
    
    # Check that value is reasonable
    assert isinstance(value, float)
    assert 0 <= value <= 100  # Validation accuracy percentage


# Test 3: Run study with small number of trials
def test_run_study(sample_data, temp_output_dir):
    """Test that study runs and produces results."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=3,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    study = tuner.run_study(study_name='test_study')
    
    # Check study completed
    assert len(study.trials) == 3
    assert study.best_trial is not None
    assert study.best_value is not None
    
    # Check best parameters exist
    assert 'learning_rate' in study.best_params
    assert 'batch_size' in study.best_params
    assert 'dropout_rate' in study.best_params
    assert 'weight_decay' in study.best_params
    assert 'optimizer' in study.best_params


# Test 4: Save results to JSON
def test_save_results(sample_data, temp_output_dir):
    """Test that results are saved correctly to JSON."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=2,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    study = tuner.run_study()
    results = tuner.save_results(study)
    
    # Check results structure
    assert 'best_validation_accuracy' in results
    assert 'best_trial_number' in results
    assert 'best_hyperparameters' in results
    assert 'study_statistics' in results
    assert 'all_trials' in results
    
    # Check JSON file exists
    json_path = os.path.join(temp_output_dir, 'best_hyperparameters.json')
    assert os.path.exists(json_path)
    
    # Verify JSON can be loaded
    with open(json_path, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results['best_validation_accuracy'] == results['best_validation_accuracy']
    assert loaded_results['best_hyperparameters'] == results['best_hyperparameters']


# Test 5: Hyperparameter search function
def test_hyperparameter_search(sample_data, temp_output_dir):
    """Test the main hyperparameter_search function."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    results, study = hyperparameter_search(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=3,
        output_dir=temp_output_dir,
        device='cpu'
    )
    
    # Check results returned
    assert isinstance(results, dict)
    assert isinstance(study, optuna.Study)
    
    # Check results content
    assert 'best_validation_accuracy' in results
    assert 'best_hyperparameters' in results
    assert 'study_statistics' in results
    
    # Check study completed
    assert len(study.trials) == 3


# Test 6: Parameter ranges are correct
def test_parameter_ranges(sample_data, temp_output_dir):
    """Test that suggested parameters are within expected ranges."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=5,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    study = tuner.run_study()
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            params = trial.params
            
            # Check learning rate range
            assert 1e-5 <= params['learning_rate'] <= 1e-2
            
            # Check batch size is in allowed values
            assert params['batch_size'] in [16, 32, 64]
            
            # Check dropout rate range
            assert 0.3 <= params['dropout_rate'] <= 0.7
            
            # Check weight decay range
            assert 1e-5 <= params['weight_decay'] <= 1e-3
            
            # Check optimizer is valid
            assert params['optimizer'] in ['Adam', 'AdamW', 'SGD']


# Test 7: Pruning works correctly
def test_pruning_mechanism(sample_data, temp_output_dir):
    """Test that pruning mechanism works."""
    train_features, train_labels, val_features, val_labels = sample_data
    
    tuner = HyperparameterTuner(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_trials=10,
        device='cpu',
        output_dir=temp_output_dir
    )
    
    # Use aggressive pruning
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2,
        n_warmup_steps=2
    )
    
    study = tuner.run_study(pruner=pruner)
    
    # Check that some trials were pruned
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Should have at least some completed trials
    assert len(completed_trials) > 0
    
    # Pruning might or might not occur depending on performance
    # Just check that the mechanism doesn't crash
    assert len(study.trials) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
