"""
Centralized configuration file for DADA hybrid augmentation experiments.

This file contains all configuration parameters that can be imported by all notebooks.
"""

import os

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Experiment Type
EXPERIMENT_TYPE = 'hybrid'  # 'dada', 'hybrid', or 'both'

# Random Seeds
SEED = 42
RANDOM_SEED = 42

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset Settings
DATASET_CONFIG = {
    'name': 'cifar10',
    'img_size': 32,
    'num_classes': 10,
    'samples_per_class': 200,  # Low-data regime: 200 samples per class
    'val_samples': 5000,       # Validation samples from remaining training data
    'batch_size': 32,          # Batch size for training
    'data_dir': './data',      # Directory to save processed data
    'use_full_test_set': True, # Use full test set for evaluation
    # Note: Original DADA preprocessing: normalize to [-1, 1] range using (-127.5 + data) / 128
}

# CIFAR-10 Class Names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# =============================================================================
# DADA MODEL CONFIGURATION
# =============================================================================

# DADA Model Settings (matching original DADA exactly)
DADA_CONFIG = {
    'latent_dim': 100,         # Generator noise dimension (original DADA)
    'learning_rate': 0.0003,   # Learning rate for GAN training (original DADA default)
    'beta1': 0.5,             # Adam optimizer beta1
    'beta2': 0.999,           # Adam optimizer beta2
    'use_weight_norm': True,   # Use weight normalization
    'use_batch_norm': True,    # Use batch normalization
    'dropout_rate': 0.5,       # Dropout rate
    'gan_epochs': 200,         # GAN training epochs (original DADA)
    'total_epochs': 700,       # Total training epochs (original DADA)
}

# =============================================================================
# HYBRID AUGMENTATION CONFIGURATION
# =============================================================================

# Hybrid Experiment Settings
HYBRID_CONFIG = {
    'enable_mixup': True,      # Enable Mixup augmentation
    'enable_cutmix': True,     # Enable CutMix augmentation
    'mixup_alpha': 0.2,        # Mixup alpha parameter
    'cutmix_alpha': 1.0,       # CutMix alpha parameter
    'cutmix_prob': 0.5,        # Probability of applying CutMix
}

# =============================================================================
# CLASSIFIER CONFIGURATION
# =============================================================================

# Classifier Settings
CLASSIFIER_CONFIG = {
    'model_type': 'mobilenet',  # 'mobilenet', 'resnet', 'efficientnet', 'custom'
    'freeze_pretrained': True,  # Freeze pretrained layers
    'dropout_rate': 0.5,        # Dropout rate for classifier
    'learning_rate': 0.001,     # Learning rate for classifier training
    'epochs': 100,              # Number of training epochs
    'early_stopping_patience': 10,  # Early stopping patience
}

# =============================================================================
# DATA AUGMENTATION CONFIGURATION
# =============================================================================

# Traditional Data Augmentation (matching original DADA exactly)
DATA_AUG_CONFIG = {
    'enabled': True,           # Whether to use traditional data augmentation
    'rotation_range': 20,      # Rotation range in degrees (original DADA)
    'width_shift_range': 0.1,  # Width shift range (original DADA)
    'height_shift_range': 0.1, # Height shift range (original DADA)
    'horizontal_flip': True,   # Horizontal flip (original DADA)
    'zoom_range': 0.0,        # Zoom range (original DADA: no zoom)
    'shear_range': 0.0,       # Shear range (original DADA: no shear)
    'fill_mode': 'nearest',    # Fill mode (original DADA)
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training Settings
TRAINING_CONFIG = {
    'batch_size': 32,          # Batch size
    'epochs': 100,             # Number of epochs
    'learning_rate': 0.001,    # Learning rate
    'optimizer': 'adam',       # Optimizer
    'loss': 'sparse_categorical_crossentropy',  # Loss function
    'metrics': ['accuracy'],   # Metrics to track
    'validation_split': 0.2,   # Validation split
    'shuffle': True,           # Shuffle data
    'verbose': 1,              # Verbosity level
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Evaluation Settings
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'macro_f1'],  # Metrics to evaluate
    'adversarial_robustness': False,      # Whether to test adversarial robustness
    'fgsm_epsilon': 0.1,                 # FGSM attack epsilon
    'pgd_epsilon': 0.1,                  # PGD attack epsilon
    'pgd_steps': 10,                     # PGD attack steps
}

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

# File Paths
PATHS_CONFIG = {
    'data_dir': './saved_data',              # Data directory
    'models_dir': './models',          # Models directory
    'results_dir': './results',        # Results directory
    'logs_dir': './logs',              # Logs directory
    'figures_dir': './figures',        # Figures directory
}

# =============================================================================
# EXPERIMENTAL SETUPS
# =============================================================================

# Define different experimental setups
EXPERIMENTAL_SETUPS = {
    'dada_baseline': {
        'name': 'DADA Baseline',
        'description': 'Original DADA experiment setup',
        'samples_per_class': 200,
        'val_split': 0.0,
        'test_split': 0.0,
        'batch_size': 100,      # Original DADA batch size
        'random_seed': 1,       # Original DADA seed
        'data_augmentation': True,
        'use_full_test_set': True,
        'gan_epochs': 200,      # Original DADA GAN epochs
        'total_epochs': 700,    # Original DADA total epochs
        'learning_rate': 0.0003, # Original DADA learning rate
    },
    
    'hybrid': {
        'name': 'Hybrid Setup',
        'description': 'Hybrid DADA + Mixup/CutMix experiment',
        'samples_per_class': 200,
        'val_samples': 5000,
        'batch_size': 32,
        'random_seed': 42,
        'data_augmentation': True,
        'enable_mixup': True,
        'enable_cutmix': True,
        'classifier_model': 'mobilenet',
        'freeze_pretrained': True,
    },
    
    'comparison': {
        'name': 'Comparison Setup',
        'description': 'Setup for comparing all augmentation methods',
        'samples_per_class': 200,
        'val_samples': 5000,
        'batch_size': 32,
        'random_seed': 42,
        'data_augmentation': True,
        'enable_mixup': True,
        'enable_cutmix': True,
        'classifier_model': 'mobilenet',
        'freeze_pretrained': True,
    }
}

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_config(experiment_type='hybrid'):
    """
    Get the complete configuration for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment ('dada_baseline', 'hybrid', 'comparison')
    
    Returns:
        dict: Complete configuration dictionary
    """
    # Start with base configuration
    config = {
        'experiment_type': experiment_type,
        'dataset': DATASET_CONFIG.copy(),
        'dada': DADA_CONFIG.copy(),
        'hybrid': HYBRID_CONFIG.copy(),
        'classifier': CLASSIFIER_CONFIG.copy(),
        'data_aug': DATA_AUG_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'evaluation': EVALUATION_CONFIG.copy(),
        'paths': PATHS_CONFIG.copy(),
    }
    
    # Update with specific experimental setup
    if experiment_type in EXPERIMENTAL_SETUPS:
        setup = EXPERIMENTAL_SETUPS[experiment_type]
        config.update(setup)
    
    return config

def create_directories():
    """Create necessary directories for the experiment."""
    directories = [
        PATHS_CONFIG['data_dir'],
        PATHS_CONFIG['models_dir'],
        PATHS_CONFIG['results_dir'],
        PATHS_CONFIG['logs_dir'],
        PATHS_CONFIG['figures_dir'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def print_config(config):
    """Print the configuration in a readable format."""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    
    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    
    print("=" * 60)

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default configuration for hybrid experiment
CONFIG = get_config('hybrid')

# Create directories on import
create_directories()

# Export key variables for easy import
__all__ = [
    'CONFIG',
    'CIFAR10_CLASSES', 
    'get_config',
    'print_config',
    'create_directories',
    'DATASET_CONFIG',
    'DADA_CONFIG',
    'HYBRID_CONFIG',
    'CLASSIFIER_CONFIG',
    'DATA_AUG_CONFIG',
    'TRAINING_CONFIG',
    'EVALUATION_CONFIG',
    'PATHS_CONFIG',
    'EXPERIMENTAL_SETUPS'
]

if __name__ == "__main__":
    print_config(CONFIG) 