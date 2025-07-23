#!/usr/bin/env python3
"""
Performance benchmark for Neural Process models.
"""

import time
import numpy as np
import keras
import keras_neural_processes as knp


def benchmark_model(model_class, model_name, inputs, num_runs=10):
    """Benchmark a model's forward pass performance."""
    print(f"\n=== Benchmarking {model_name} ===")
    
    # Create model
    model = model_class()
    
    # Warm-up run
    if model_name == "CNP":
        _ = model(inputs[:3], training=False)
    else:
        _ = model(inputs, training=False)
    
    # Benchmark forward passes
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        if model_name == "CNP":
            _ = model(inputs[:3], training=False)
        else:
            _ = model(inputs, training=False)
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Forward pass: {avg_time:.4f}s ± {std_time:.4f}s")
    
    return avg_time, std_time


def benchmark_training_step(model_class, model_name, inputs, num_runs=5):
    """Benchmark a model's training step performance."""
    print(f"Training step benchmark for {model_name}:")
    
    model = model_class()
    model.optimizer = optimizer
    
    if model_name == "CNP":
        context_x, context_y, target_x = inputs[:3]
        target_y = np.random.randn(*target_x.shape[:-1], 1).astype(np.float32)
        
        # Warm-up
        _ = model.train_step(context_x, context_y, target_x, target_y)
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.train_step(context_x, context_y, target_x, target_y)
            times.append(time.time() - start_time)
    else:
        context_x, context_y, target_x, target_y = inputs
        
        # Warm-up
        _ = model.train_step(context_x, context_y, target_x, target_y)
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.train_step(context_x, context_y, target_x, target_y)
            times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Training step: {avg_time:.4f}s ± {std_time:.4f}s")
    
    return avg_time, std_time


def main():
    """Run comprehensive benchmarks."""
    print("Neural Process Performance Benchmark")
    print("=====================================")
    
    # Test configuration
    batch_sizes = [16, 32]
    num_points = [25, 50]
    
    for batch_size in batch_sizes:
        for points in num_points:
            print(f"\nConfiguration: batch_size={batch_size}, num_points={points}")
            print("-" * 60)
            
            # Generate test data
            X_train = np.random.randn(batch_size, points, 2).astype(np.float32)
            y_train = np.random.randn(batch_size, points, 1).astype(np.float32)
            context_x = X_train[:, :points//3, :]  
            context_y = y_train[:, :points//3, :]
            target_x = X_train
            target_y = y_train
            
            inputs = (context_x, context_y, target_x, target_y)
            
            # Benchmark all models
            models = [
                (knp.CNP, "CNP"),
                (knp.NP, "NP"), 
                (knp.ANP, "ANP")
            ]
            
            forward_times = {}
            training_times = {}
            
            for model_class, model_name in models:
                try:
                    fwd_time, _ = benchmark_model(model_class, model_name, inputs)
                    train_time, _ = benchmark_training_step(model_class, model_name, inputs)
                    
                    forward_times[model_name] = fwd_time
                    training_times[model_name] = train_time
                    
                except Exception as e:
                    print(f"Error benchmarking {model_name}: {e}")
                    forward_times[model_name] = float('inf')
                    training_times[model_name] = float('inf')
            
            # Summary
            print(f"\nSummary for batch_size={batch_size}, num_points={points}:")
            print("Forward pass times:")
            for name, time_val in forward_times.items():
                print(f"  {name}: {time_val:.4f}s")
            
            print("Training step times:")
            for name, time_val in training_times.items():
                print(f"  {name}: {time_val:.4f}s")


if __name__ == "__main__":
    main()