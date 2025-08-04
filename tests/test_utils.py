import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from keras_neural_processes import utils


# Test fixtures for utilities
@pytest.fixture
def sample_gp_data():
    """Generate sample GP-like data for testing utilities."""
    batch_size = 3
    num_points = 30
    x_dim = 1
    y_dim = 1
    
    # Generate some function data
    x = np.linspace(-2, 2, num_points).reshape(1, num_points, x_dim)
    x = np.repeat(x, batch_size, axis=0)
    y = np.sin(x * np.pi) + 0.1 * np.random.randn(batch_size, num_points, y_dim)
    
    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def multi_channel_gp_data():
    """Generate multi-channel GP data for testing."""
    batch_size = 2
    num_points_per_channel = 20
    num_channels = 3
    total_points = num_points_per_channel * num_channels
    
    x = np.zeros((batch_size, total_points, 2))  # [time, channel_id]
    y = np.zeros((batch_size, total_points, 1))
    
    for batch_idx in range(batch_size):
        for ch in range(num_channels):
            start_idx = ch * num_points_per_channel
            end_idx = (ch + 1) * num_points_per_channel
            
            time_points = np.linspace(-2, 2, num_points_per_channel)
            x[batch_idx, start_idx:end_idx, 0] = time_points
            x[batch_idx, start_idx:end_idx, 1] = ch
            
            # Different function for each channel
            if ch == 0:
                y[batch_idx, start_idx:end_idx, 0] = np.sin(time_points * np.pi)
            elif ch == 1:
                y[batch_idx, start_idx:end_idx, 0] = np.cos(time_points * np.pi)
            else:
                y[batch_idx, start_idx:end_idx, 0] = time_points**2
    
    return x.astype(np.float32), y.astype(np.float32)


class TestContextSetGeneration:
    
    def test_get_context_set_basic_functionality(self, sample_gp_data):
        """Test basic functionality of get_context_set."""
        x, y = sample_gp_data
        num_context = 10
        
        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all"
        )
        
        assert context_x.shape == (x.shape[0], num_context, x.shape[2])
        assert context_y.shape == (y.shape[0], num_context, y.shape[2])
        
        # Check that context points are actually from the original data
        for batch_idx in range(x.shape[0]):
            for ctx_idx in range(num_context):
                ctx_point = context_x[batch_idx, ctx_idx]
                # Check if this point exists in the original data
                found = False
                for orig_idx in range(x.shape[1]):
                    if np.allclose(ctx_point, x[batch_idx, orig_idx]):
                        found = True
                        break
                assert found, "Context point not found in original data"
    
    def test_get_context_set_each_mode(self, multi_channel_gp_data):
        """Test get_context_set with 'each' mode for multi-channel data."""
        x, y = multi_channel_gp_data
        num_context = 5
        
        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="each"
        )
        
        # Should have num_context points from each of the 3 channels
        expected_total = num_context * 3
        assert context_x.shape == (x.shape[0], expected_total, x.shape[2])
        assert context_y.shape == (y.shape[0], expected_total, y.shape[2])
        
        # Check that we have the right number from each channel
        for batch_idx in range(x.shape[0]):
            channel_counts = {}
            for ctx_idx in range(expected_total):
                channel_id = int(context_x[batch_idx, ctx_idx, 1])
                channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1
            
            for channel_id in [0, 1, 2]:
                assert channel_counts[channel_id] == num_context
    
    def test_get_context_set_per_channel_list(self, multi_channel_gp_data):
        """Test get_context_set with different numbers per channel."""
        x, y = multi_channel_gp_data
        num_context = [3, 5, 7]  # Different for each channel
        
        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="each"
        )
        
        expected_total = sum(num_context)
        assert context_x.shape == (x.shape[0], expected_total, x.shape[2])
        assert context_y.shape == (y.shape[0], expected_total, y.shape[2])
    
    def test_get_context_set_with_seed(self, sample_gp_data):
        """Test that seeding produces reproducible results."""
        x, y = sample_gp_data
        num_context = 10
        seed = 42
        
        context_x1, context_y1 = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all", seed=seed
        )
        
        context_x2, context_y2 = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all", seed=seed
        )
        
        assert np.allclose(context_x1, context_x2)
        assert np.allclose(context_y1, context_y2)
    
    def test_get_context_set_error_cases(self, sample_gp_data):
        """Test error handling in get_context_set."""
        x, y = sample_gp_data
        
        # Test invalid mode
        with pytest.raises(AttributeError):
            utils.get_context_set(x, y, num_context=10, num_context_mode="invalid")
        
        # Test requesting too many points
        with pytest.raises(ValueError):
            utils.get_context_set(x, y, num_context=x.shape[1] + 10, num_context_mode="all")
        
        # Test list with wrong mode
        with pytest.raises(ValueError):
            utils.get_context_set(x, y, num_context=[5, 10], num_context_mode="all")


class TestTrainBatchGeneration:
    
    def test_get_train_batch_basic(self, sample_gp_data):
        """Test basic functionality of get_train_batch."""
        x, y = sample_gp_data
        batch_size = 2
        
        x_batch, y_batch = utils.get_train_batch(x, y, batch_size=batch_size)
        
        assert x_batch.shape == (batch_size, x.shape[1], x.shape[2])
        assert y_batch.shape == (batch_size, y.shape[1], y.shape[2])
    
    def test_get_train_batch_with_tensors(self, sample_gp_data):
        """Test get_train_batch with TensorFlow tensors."""
        x, y = sample_gp_data
        batch_size = 2
        
        x_tf = tf.constant(x)
        y_tf = tf.constant(y)
        
        x_batch, y_batch = utils.get_train_batch(x_tf, y_tf, batch_size=batch_size)
        
        assert isinstance(x_batch, tf.Tensor)
        assert isinstance(y_batch, tf.Tensor)
        assert x_batch.shape == (batch_size, x.shape[1], x.shape[2])
        assert y_batch.shape == (batch_size, y.shape[1], y.shape[2])
    
    def test_get_train_batch_larger_than_dataset(self, sample_gp_data):
        """Test get_train_batch when requesting more samples than available."""
        x, y = sample_gp_data
        batch_size = x.shape[0] + 5  # More than available
        
        # Should work with replacement (the function uses replace=False, so this will error)
        # Let's test that the function handles this correctly by expecting the error
        with pytest.raises(ValueError, match="Cannot take a larger sample than population"):
            utils.get_train_batch(x, y, batch_size=batch_size)


class TestNumContextSampling:
    
    def test_sample_num_context_simple_range(self):
        """Test _sample_num_context with simple range."""
        num_context_range = [5, 15]
        
        for _ in range(10):  # Test multiple times for randomness
            sampled = utils._sample_num_context(num_context_range)
            assert isinstance(sampled, (int, np.integer))
            assert 5 <= sampled <= 15
    
    def test_sample_num_context_per_channel_range(self):
        """Test _sample_num_context with per-channel ranges."""
        per_channel_range = [[3, 7], [10, 15], [5, 8]]
        
        for _ in range(10):
            sampled = utils._sample_num_context(per_channel_range)
            assert isinstance(sampled, list)
            assert len(sampled) == 3
            assert 3 <= sampled[0] <= 7
            assert 10 <= sampled[1] <= 15
            assert 5 <= sampled[2] <= 8
    
    def test_get_fixed_num_context_simple_range(self):
        """Test _get_fixed_num_context with simple range."""
        num_context_range = [4, 10]  # Mean should be 7
        fixed = utils._get_fixed_num_context(num_context_range)
        assert fixed == 7
    
    def test_get_fixed_num_context_per_channel_range(self):
        """Test _get_fixed_num_context with per-channel ranges."""
        per_channel_range = [[4, 6], [8, 12], [3, 7]]  # Means: [5, 10, 5]
        fixed = utils._get_fixed_num_context(per_channel_range)
        assert fixed == [5, 10, 5]
    
    def test_is_per_channel_range_detection(self):
        """Test _is_per_channel_range detection function."""
        # Simple range (not per-channel)
        assert not utils._is_per_channel_range([5, 10])
        
        # Per-channel format
        assert utils._is_per_channel_range([[5, 10], [3, 8]])
        assert utils._is_per_channel_range([[1, 2], [3, 4], [5, 6]])
        
        # Empty or invalid formats
        assert not utils._is_per_channel_range([])
        assert not utils._is_per_channel_range([5, [3, 8]])  # Mixed format
        assert not utils._is_per_channel_range("invalid")


class TestMetricsCalculation:
    
    def test_gplike_calculate_mymetrics_basic(self, sample_gp_data):
        """Test basic functionality of gplike_calculate_mymetrics."""
        x, y = sample_gp_data
        
        # Create mock predictions
        pred_x = x  # Same as target
        pred_y_mean = y + 0.05 * np.random.randn(*y.shape)  # Close to truth
        pred_y_std = 0.1 * np.ones_like(y)  # Small uncertainty
        
        # Get context set
        context_x, context_y = utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )
        
        metrics = utils.gplike_calculate_mymetrics(
            pred_x, pred_y_mean, pred_y_std,
            x, y,  # target
            context_x, context_y
        )
        
        # Check all expected metrics are present
        expected_keys = [
            'target_coverage',
            'fraction_context_close',
            'context_reconstruction_coverage',
            'mean_predictive_confidence'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))
        
        # Check reasonable ranges
        assert 0 <= metrics['target_coverage'] <= 1
        assert 0 <= metrics['fraction_context_close'] <= 1
        assert 0 <= metrics['context_reconstruction_coverage'] <= 1
        assert metrics['mean_predictive_confidence'] > 0
    
    def test_gplike_calculate_mymetrics_perfect_predictions(self, sample_gp_data):
        """Test metrics with perfect predictions."""
        x, y = sample_gp_data
        
        # Perfect predictions
        pred_x = x
        pred_y_mean = y  # Exact match
        pred_y_std = 0.001 * np.ones_like(y)  # Very small uncertainty
        
        context_x, context_y = utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )
        
        metrics = utils.gplike_calculate_mymetrics(
            pred_x, pred_y_mean, pred_y_std,
            x, y,
            context_x, context_y,
            closeness_thresh=0.1
        )
        
        # With perfect predictions, most metrics should be high
        assert metrics['target_coverage'] > 0.9  # Most points should be covered
        assert metrics['fraction_context_close'] == 1.0  # All context points should be close
    
    def test_gplike_calculate_mymetrics_with_parameters(self, sample_gp_data):
        """Test metrics calculation with different parameters."""
        x, y = sample_gp_data
        
        pred_x = x
        pred_y_mean = y + 0.2 * np.random.randn(*y.shape)
        pred_y_std = 0.3 * np.ones_like(y)
        
        context_x, context_y = utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )
        
        # Test with different closeness threshold
        metrics1 = utils.gplike_calculate_mymetrics(
            pred_x, pred_y_mean, pred_y_std,
            x, y, context_x, context_y,
            closeness_thresh=0.05  # Stricter threshold
        )
        
        metrics2 = utils.gplike_calculate_mymetrics(
            pred_x, pred_y_mean, pred_y_std,
            x, y, context_x, context_y,
            closeness_thresh=0.5  # More lenient threshold
        )
        
        # More lenient threshold should give higher "close" fraction
        assert metrics2['fraction_context_close'] >= metrics1['fraction_context_close']


class TestValidationUtilities:
    
    def test_gplike_fixed_sets(self, sample_gp_data):
        """Test gplike_fixed_sets function."""
        x, y = sample_gp_data
        pred_points = x.shape[1]
        seed = 42
        fixed_val_num_context = 8
        num_context_mode = "all"
        
        (context_x_fixed, context_y_fixed, target_x_fixed, 
         target_y_fixed, pred_x_fixed) = utils.gplike_fixed_sets(
            x, y, pred_points, seed, fixed_val_num_context, num_context_mode
        )
        
        # Check shapes
        assert context_x_fixed.shape == (x.shape[0], fixed_val_num_context, x.shape[2])
        assert context_y_fixed.shape == (y.shape[0], fixed_val_num_context, y.shape[2])
        assert target_x_fixed.shape == x.shape
        assert target_y_fixed.shape == y.shape
        assert pred_x_fixed.shape == x.shape  # Should be same as target_x
        
        # Check reproducibility with same seed
        (context_x_fixed2, context_y_fixed2, _, _, _) = utils.gplike_fixed_sets(
            x, y, pred_points, seed, fixed_val_num_context, num_context_mode
        )
        
        assert np.allclose(context_x_fixed, context_x_fixed2)
        assert np.allclose(context_y_fixed, context_y_fixed2)
    
    def test_gplike_new_sets(self, sample_gp_data):
        """Test gplike_new_sets function."""
        x, y = sample_gp_data
        pred_points = x.shape[1]
        num_context_range = [5, 15]
        num_context_mode = "all"
        
        (context_x, context_y, target_x, 
         target_y, pred_x) = utils.gplike_new_sets(
            x, y, pred_points, num_context_range, num_context_mode
        )
        
        # Check that we get valid shapes
        assert context_x.shape[0] == x.shape[0]  # Same batch size
        assert context_x.shape[2] == x.shape[2]  # Same feature dim
        assert 5 <= context_x.shape[1] <= 15  # Context size in range
        
        assert target_x.shape == x.shape
        assert target_y.shape == y.shape
        assert pred_x.shape == x.shape


class TestTensorCompatibility:
    
    def test_get_context_set_tensor_consistency(self, sample_gp_data):
        """Test that get_context_set preserves tensor types."""
        x, y = sample_gp_data
        num_context = 10
        seed = 42  # Use same seed for reproducible results
        
        # Test with numpy arrays
        context_x_np, context_y_np = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all", seed=seed
        )
        assert isinstance(context_x_np, np.ndarray)
        assert isinstance(context_y_np, np.ndarray)
        
        # Test with TensorFlow tensors
        x_tf = tf.constant(x)
        y_tf = tf.constant(y)
        
        context_x_tf, context_y_tf = utils.get_context_set(
            x_tf, y_tf, num_context=num_context, num_context_mode="all", seed=seed
        )
        assert isinstance(context_x_tf, tf.Tensor)
        assert isinstance(context_y_tf, tf.Tensor)
        
        # Results should be equivalent when using the same seed
        assert np.allclose(ops.convert_to_numpy(context_x_tf), context_x_np)
        assert np.allclose(ops.convert_to_numpy(context_y_tf), context_y_np)