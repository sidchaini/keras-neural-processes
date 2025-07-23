"""
Optimized model factory with mixed precision and other performance enhancements.
"""

import keras
from keras import mixed_precision

from .src import CNP, NP, ANP


def create_optimized_model(model_type, use_mixed_precision=True, **kwargs):
    """
    Create an optimized neural process model with performance enhancements.
    
    Args:
        model_type: "CNP", "NP", or "ANP"
        use_mixed_precision: Whether to enable mixed precision training
        **kwargs: Additional arguments for the model
    
    Returns:
        Optimized model instance
    """
    # Enable mixed precision for significant speedup
    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    # Create model with optimized defaults
    if model_type.upper() == "CNP":
        model = CNP(**kwargs)
    elif model_type.upper() == "NP":
        model = NP(**kwargs)
    elif model_type.upper() == "ANP":
        # Use optimized defaults for ANP
        default_kwargs = {
            "encoder_sizes": [128, 128, 128, 128],
            "num_heads": 8,
            "dropout_rate": 0.1,
        }
        default_kwargs.update(kwargs)
        model = ANP(**default_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def configure_for_performance(model, optimizer=None):
    """
    Configure a model for optimal training performance.
    
    Args:
        model: Neural process model
        optimizer: Optimizer to use (defaults to Adam with mixed precision)
    
    Returns:
        Configured model
    """
    if optimizer is None:
        # Use optimizer optimized for mixed precision
        optimizer = keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Slightly larger epsilon for stability
        )
    
    model.compile(optimizer=optimizer)
    return model