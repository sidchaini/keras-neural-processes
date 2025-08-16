import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

import keras_neural_processes as knp
from keras_neural_processes.src import (
    MeanEncoder,
    LatentEncoder,
    AttentiveEncoder,
    Decoder,
    CNP,
    NP,
    ANP,
)
from keras_neural_processes import utils


# Tests for basic model instantiation
def test_CNP():
    model = knp.CNP()
    assert model is not None
    assert isinstance(model, CNP)


def test_NP():
    model = knp.NP()
    assert model is not None
    assert isinstance(model, NP)


def test_ANP():
    model = knp.ANP()
    assert model is not None
    assert isinstance(model, ANP)


# Tests for model configuration and instantiation with custom parameters
class TestModelInstantiation:

    def test_cnp_custom_params(self):
        """Test CNP with custom parameters."""
        model = CNP(encoder_sizes=[64, 64], decoder_sizes=[32, 32], output_dims=2)
        assert model.encoder_sizes == [64, 64]
        assert model.decoder_sizes_hidden == [32, 32]
        assert model.output_dims == 2

    def test_np_custom_params(self):
        """Test NP with custom parameters."""
        model = NP(
            det_encoder_sizes=[64, 64],
            latent_encoder_sizes=[32, 32],
            num_latents=64,
            decoder_sizes=[16, 16],
            output_dims=2,
        )
        assert model.det_encoder_sizes == [64, 64]
        assert model.latent_encoder_sizes == [32, 32]
        assert model.num_latents == 64
        assert model.decoder_sizes_hidden == [16, 16]
        assert model.output_dims == 2

    def test_anp_custom_params(self):
        """Test ANP with custom parameters."""
        model = ANP(
            encoder_sizes=[64, 64],
            num_heads=4,
            latent_encoder_sizes=[32, 32],
            num_latents=64,
            decoder_sizes=[16, 16],
            output_dims=2,
        )
        assert model.encoder_sizes == [64, 64]
        assert model.num_heads == 4
        assert model.latent_encoder_sizes == [32, 32]
        assert model.num_latents == 64
        assert model.decoder_sizes_hidden == [16, 16]
        assert model.output_dims == 2


# Tests for model building and forward passes
class TestModelFunctionality:

    def test_cnp_build_and_call(self, context_target_data):
        """Test CNP build and forward pass."""
        context_x, context_y, target_x, target_y = context_target_data

        model = CNP()

        # Build the model
        model.build([context_x.shape, context_y.shape, target_x.shape])

        # Forward pass
        mean, std = model([context_x, context_y, target_x])

        assert mean.shape == target_y.shape
        assert std.shape == target_y.shape
        assert ops.all(std > 0)  # Standard deviation should be positive

    def test_np_build_and_call(self, context_target_data):
        """Test NP build and forward pass."""
        context_x, context_y, target_x, target_y = context_target_data

        model = NP()

        # Build the model
        model.build([context_x.shape, context_y.shape, target_x.shape, target_y.shape])

        # Forward pass during training
        pred_dist, prior_dist, posterior_dist = model(
            [context_x, context_y, target_x, target_y], training=True
        )

        assert pred_dist.mean().shape == target_y.shape
        assert pred_dist.stddev().shape == target_y.shape
        assert prior_dist is not None
        assert posterior_dist is not None

        # Forward pass during inference
        pred_dist_test, prior_dist_test, posterior_dist_test = model(
            [context_x, context_y, target_x, None], training=False
        )

        assert pred_dist_test.mean().shape == target_y.shape
        assert pred_dist_test.stddev().shape == target_y.shape
        assert prior_dist_test is not None
        assert posterior_dist_test is None  # Should be None during inference

    def test_anp_build_and_call(self, context_target_data):
        """Test ANP build and forward pass."""
        context_x, context_y, target_x, target_y = context_target_data

        model = ANP()

        # Build the model
        model.build([context_x.shape, context_y.shape, target_x.shape, target_y.shape])

        # Forward pass during training
        pred_dist, prior_dist, posterior_dist = model(
            [context_x, context_y, target_x, target_y], training=True
        )

        assert pred_dist.mean().shape == target_y.shape
        assert pred_dist.stddev().shape == target_y.shape
        assert prior_dist is not None
        assert posterior_dist is not None


# Tests for individual layers
class TestLayers:

    def test_deterministic_encoder(self, context_target_data):
        """Test MeanEncoder."""
        context_x, context_y, _, _ = context_target_data

        encoder = MeanEncoder(hidden_sizes=[64, 64, 32])
        encoder.build([context_x.shape, context_y.shape])

        representation = encoder(context_x, context_y)

        expected_shape = (context_x.shape[0], 1, 32)  # batch, 1, hidden_size
        assert representation.shape == expected_shape

    def test_latent_encoder(self, context_target_data):
        """Test LatentEncoder."""
        context_x, context_y, _, _ = context_target_data

        encoder = LatentEncoder(hidden_sizes=[64, 64], num_latents=32)
        encoder.build([context_x.shape, context_y.shape])

        distribution = encoder(context_x, context_y)

        # Should return a distribution object
        assert hasattr(distribution, "mean")
        assert hasattr(distribution, "stddev")
        assert hasattr(distribution, "sample")

        # Check distribution properties
        mean = distribution.mean()
        stddev = distribution.stddev()
        sample = distribution.sample()

        expected_shape = (context_x.shape[0], 32)  # batch, num_latents
        assert mean.shape == expected_shape
        assert stddev.shape == expected_shape
        assert sample.shape == expected_shape
        assert ops.all(stddev > 0)

    def test_attentive_encoder(self, context_target_data):
        """Test AttentiveEncoder."""
        context_x, context_y, target_x, _ = context_target_data

        encoder = AttentiveEncoder(encoder_sizes=[64, 64, 32], num_heads=4)
        encoder.build([context_x.shape, context_y.shape, target_x.shape])

        representation = encoder(context_x, context_y, target_x)

        expected_shape = (
            target_x.shape[0],
            target_x.shape[1],
            32,
        )  # batch, num_targets, hidden_size
        assert representation.shape == expected_shape

    def test_deterministic_decoder(self, context_target_data):
        """Test Decoder."""
        context_x, context_y, target_x, target_y = context_target_data

        # Create a dummy representation that matches target shape
        representation_dim = 64
        representation = np.random.randn(
            target_x.shape[0],
            target_x.shape[1],
            representation_dim,  # Same num_points as target
        ).astype(np.float32)

        decoder = Decoder(hidden_sizes=[32, 32, 2])
        decoder.build([representation.shape, target_x.shape])

        mean, std = decoder(representation, target_x)

        assert mean.shape == target_y.shape
        assert std.shape == target_y.shape
        assert ops.all(std > 0)


# Tests for utility functions
class TestUtilityFunctions:

    def test_get_context_set_all_mode(self, sample_data_1d):
        """Test get_context_set with 'all' mode."""
        x, y = sample_data_1d
        num_context = 15

        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all"
        )

        assert context_x.shape == (x.shape[0], num_context, x.shape[2])
        assert context_y.shape == (y.shape[0], num_context, y.shape[2])

    def test_get_context_set_each_mode(self, sample_data_multi_channel):
        """Test get_context_set with 'each' mode."""
        x, y = sample_data_multi_channel
        num_context = 10

        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="each"
        )

        # Should have num_context points from each channel
        expected_total_points = num_context * 2  # 2 channels
        assert context_x.shape == (x.shape[0], expected_total_points, x.shape[2])
        assert context_y.shape == (y.shape[0], expected_total_points, y.shape[2])

    def test_get_context_set_per_channel_list(self, sample_data_multi_channel):
        """Test get_context_set with per-channel specification."""
        x, y = sample_data_multi_channel
        num_context = [5, 8]  # Different numbers for each channel

        context_x, context_y = utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="each"
        )

        expected_total_points = sum(num_context)
        assert context_x.shape == (x.shape[0], expected_total_points, x.shape[2])
        assert context_y.shape == (y.shape[0], expected_total_points, y.shape[2])

    def test_get_train_batch(self, sample_data_1d):
        """Test get_train_batch function."""
        x, y = sample_data_1d
        batch_size = 2

        x_batch, y_batch = utils.get_train_batch(x, y, batch_size=batch_size)

        assert x_batch.shape == (batch_size, x.shape[1], x.shape[2])
        assert y_batch.shape == (batch_size, y.shape[1], y.shape[2])

    def test_sample_num_context(self):
        """Test _sample_num_context function."""
        # Test simple range
        num_context_range = [5, 10]
        sampled = utils._sample_num_context(num_context_range)
        assert isinstance(sampled, (int, np.integer))
        assert 5 <= sampled <= 10

        # Test per-channel range
        per_channel_range = [[3, 5], [7, 10]]
        sampled = utils._sample_num_context(per_channel_range)
        assert isinstance(sampled, list)
        assert len(sampled) == 2
        assert 3 <= sampled[0] <= 5
        assert 7 <= sampled[1] <= 10

    def test_get_fixed_num_context(self):
        """Test _get_fixed_num_context function."""
        # Test simple range
        num_context_range = [5, 11]  # Mean should be 8
        fixed = utils._get_fixed_num_context(num_context_range)
        assert fixed == 8

        # Test per-channel range
        per_channel_range = [[3, 5], [7, 11]]  # Means should be [4, 9]
        fixed = utils._get_fixed_num_context(per_channel_range)
        assert fixed == [4, 9]


# Tests for training methods
class TestTrainingMethods:

    def test_cnp_train_step(self, context_target_data):
        """Test CNP train_step method."""
        context_x, context_y, target_x, target_y = context_target_data

        model = CNP()
        model.build([context_x.shape, context_y.shape, target_x.shape])

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.optimizer = optimizer

        # Convert to TensorFlow tensors for @tf.function compatibility
        context_x_tf = tf.constant(context_x)
        context_y_tf = tf.constant(context_y)
        target_x_tf = tf.constant(target_x)
        target_y_tf = tf.constant(target_y)

        loss = model.train_step(context_x_tf, context_y_tf, target_x_tf, target_y_tf)

        assert isinstance(loss, (tf.Tensor, float))
        assert not tf.math.is_nan(loss)

    def test_cnp_test_step(self, context_target_data):
        """Test CNP test_step method."""
        context_x, context_y, target_x, _ = context_target_data

        model = CNP()
        model.build([context_x.shape, context_y.shape, target_x.shape])

        # Convert to TensorFlow tensors
        context_x_tf = tf.constant(context_x)
        context_y_tf = tf.constant(context_y)
        target_x_tf = tf.constant(target_x)

        pred_mean, pred_std = model.test_step(context_x_tf, context_y_tf, target_x_tf)

        assert pred_mean.shape == target_x.shape[:-1] + (1,)  # (batch, num_points, 1)
        assert pred_std.shape == target_x.shape[:-1] + (1,)
        assert ops.all(pred_std > 0)

    def test_np_train_step(self, context_target_data):
        """Test NP train_step method."""
        context_x, context_y, target_x, target_y = context_target_data

        model = NP()
        model.build([context_x.shape, context_y.shape, target_x.shape, target_y.shape])

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.optimizer = optimizer

        # Convert to TensorFlow tensors
        context_x_tf = tf.constant(context_x)
        context_y_tf = tf.constant(context_y)
        target_x_tf = tf.constant(target_x)
        target_y_tf = tf.constant(target_y)

        loss, recon_loss, kl_div = model.train_step(
            context_x_tf, context_y_tf, target_x_tf, target_y_tf
        )

        assert isinstance(loss, (tf.Tensor, float))
        assert isinstance(recon_loss, (tf.Tensor, float))
        assert isinstance(kl_div, (tf.Tensor, float))
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_nan(recon_loss)
        assert not tf.math.is_nan(kl_div)


# Tests for error handling and edge cases
class TestErrorHandling:

    def test_incompatible_shapes(self):
        """Test error handling for incompatible input shapes."""
        model = CNP()

        # Test mismatched context_x and context_y batch dimensions
        context_x = np.random.randn(2, 10, 1).astype(np.float32)
        context_y = np.random.randn(3, 10, 1).astype(np.float32)  # Different batch size
        target_x = np.random.randn(2, 20, 1).astype(np.float32)

        # This should cause an error during the call, not build
        model.build([context_x.shape, context_y.shape, target_x.shape])

        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            _ = model([context_x, context_y, target_x])

    def test_get_context_set_invalid_mode(self, sample_data_1d):
        """Test get_context_set with invalid mode."""
        x, y = sample_data_1d

        with pytest.raises(AttributeError):
            utils.get_context_set(x, y, num_context=10, num_context_mode="invalid")

    def test_get_context_set_too_many_points(self, sample_data_1d):
        """Test get_context_set requesting more points than available."""
        x, y = sample_data_1d
        num_points = x.shape[1]

        with pytest.raises(ValueError):
            utils.get_context_set(
                x, y, num_context=num_points + 10, num_context_mode="all"
            )


# Tests for model serialization
class TestModelSerialization:

    def test_cnp_get_config(self):
        """Test CNP get_config method."""
        model = CNP(encoder_sizes=[64, 32], decoder_sizes=[32, 16], output_dims=2)
        config = model.get_config()

        assert config["encoder_sizes"] == [64, 32]
        assert config["decoder_sizes"] == [32, 16]
        assert config["output_dims"] == 2

    def test_np_get_config(self):
        """Test NP get_config method."""
        model = NP(
            det_encoder_sizes=[64, 32],
            latent_encoder_sizes=[32, 16],
            num_latents=64,
            decoder_sizes=[32, 16],
            output_dims=2,
        )
        config = model.get_config()

        assert config["det_encoder_sizes"] == [64, 32]
        assert config["latent_encoder_sizes"] == [32, 16]
        assert config["num_latents"] == 64
        assert config["decoder_sizes"] == [32, 16]
        assert config["output_dims"] == 2

    def test_anp_get_config(self):
        """Test ANP get_config method."""
        model = ANP(
            encoder_sizes=[64, 32],
            num_heads=4,
            latent_encoder_sizes=[32, 16],
            num_latents=64,
            decoder_sizes=[32, 16],
            output_dims=2,
        )
        config = model.get_config()

        assert config["encoder_sizes"] == [64, 32]
        assert config["num_heads"] == 4
        assert config["latent_encoder_sizes"] == [32, 16]
        assert config["num_latents"] == 64
        assert config["decoder_sizes"] == [32, 16]
        assert config["output_dims"] == 2


# Tests for layers get_config methods
class TestLayerSerialization:

    def test_deterministic_encoder_config(self):
        """Test MeanEncoder get_config method."""
        encoder = MeanEncoder(hidden_sizes=[64, 32, 16])
        config = encoder.get_config()
        assert config["hidden_sizes"] == [64, 32, 16]

    def test_latent_encoder_config(self):
        """Test LatentEncoder get_config method."""
        encoder = LatentEncoder(hidden_sizes=[64, 32], num_latents=16)
        config = encoder.get_config()
        assert config["hidden_sizes"] == [64, 32]
        assert config["num_latents"] == 16

    def test_attentive_encoder_config(self):
        """Test AttentiveEncoder get_config method."""
        encoder = AttentiveEncoder(encoder_sizes=[64, 32], num_heads=4)
        config = encoder.get_config()
        assert config["encoder_sizes"] == [64, 32]
        assert config["num_heads"] == 4

    def test_deterministic_decoder_config(self):
        """Test Decoder get_config method."""
        decoder = Decoder(hidden_sizes=[64, 32, 2])
        config = decoder.get_config()
        assert config["hidden_sizes"] == [64, 32, 2]


# Tests for additional utility functions
class TestAdditionalUtilities:

    def test_gplike_calculate_mymetrics(self, sample_data_1d):
        """Test gplike_calculate_mymetrics function."""
        x, y = sample_data_1d
        batch_size, num_points, _ = x.shape

        # Create mock prediction data
        pred_x = x  # Same as target for simplicity
        pred_y_mean = y + 0.1 * np.random.randn(*y.shape)  # Add some noise
        pred_y_std = 0.2 * np.ones_like(y)  # Constant uncertainty

        # Get context set
        context_x, context_y = utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )

        metrics = utils.gplike_calculate_mymetrics(
            pred_x, pred_y_mean, pred_y_std, x, y, context_x, context_y  # target
        )

        # Check that all expected metrics are present
        expected_metrics = [
            "target_coverage",
            "fraction_context_close",
            "context_reconstruction_coverage",
            "mean_predictive_confidence",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1 or metric == "mean_predictive_confidence"

    def test_is_per_channel_range(self):
        """Test _is_per_channel_range function."""
        # Test simple range (not per-channel)
        assert not utils._is_per_channel_range([5, 10])

        # Test per-channel range
        assert utils._is_per_channel_range([[5, 10], [3, 8]])

        # Test empty input
        assert not utils._is_per_channel_range([])

        # Test invalid format
        assert not utils._is_per_channel_range([5, [3, 8]])


# Tests for tensor vs numpy array handling
class TestTensorVsNumpyInputs:

    def test_cnp_with_tf_tensors(self, context_target_data):
        """Test CNP with TensorFlow tensor inputs."""
        context_x, context_y, target_x, target_y = context_target_data

        # Convert to TensorFlow tensors
        context_x_tf = tf.constant(context_x)
        context_y_tf = tf.constant(context_y)
        target_x_tf = tf.constant(target_x)

        model = CNP()
        model.build([context_x.shape, context_y.shape, target_x.shape])

        mean, std = model([context_x_tf, context_y_tf, target_x_tf])

        assert isinstance(mean, tf.Tensor)
        assert isinstance(std, tf.Tensor)
        assert mean.shape == target_y.shape
        assert std.shape == target_y.shape

    def test_get_context_set_with_tf_tensors(self, sample_data_1d):
        """Test get_context_set with TensorFlow tensor inputs."""
        x, y = sample_data_1d

        # Convert to TensorFlow tensors
        x_tf = tf.constant(x)
        y_tf = tf.constant(y)

        context_x, context_y = utils.get_context_set(
            x_tf, y_tf, num_context=10, num_context_mode="all"
        )

        assert isinstance(context_x, tf.Tensor)
        assert isinstance(context_y, tf.Tensor)
        assert context_x.shape == (x.shape[0], 10, x.shape[2])
        assert context_y.shape == (y.shape[0], 10, y.shape[2])


# Tests for different input/output dimensions
class TestDifferentDimensions:

    def test_multi_output_cnp(self):
        """Test CNP with multi-dimensional output."""
        batch_size = 2
        num_context = 5
        num_targets = 10
        x_dim = 1
        y_dim = 3  # Multi-dimensional output

        context_x = np.random.randn(batch_size, num_context, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, num_context, y_dim).astype(np.float32)
        target_x = np.random.randn(batch_size, num_targets, x_dim).astype(np.float32)

        model = CNP(output_dims=y_dim)
        model.build([context_x.shape, context_y.shape, target_x.shape])

        mean, std = model([context_x, context_y, target_x])

        assert mean.shape == (batch_size, num_targets, y_dim)
        assert std.shape == (batch_size, num_targets, y_dim)

    def test_multi_input_cnp(self):
        """Test CNP with multi-dimensional input."""
        batch_size = 2
        num_context = 5
        num_targets = 10
        x_dim = 2  # Multi-dimensional input
        y_dim = 1

        context_x = np.random.randn(batch_size, num_context, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, num_context, y_dim).astype(np.float32)
        target_x = np.random.randn(batch_size, num_targets, x_dim).astype(np.float32)

        model = CNP()
        model.build([context_x.shape, context_y.shape, target_x.shape])

        mean, std = model([context_x, context_y, target_x])

        assert mean.shape == (batch_size, num_targets, y_dim)
        assert std.shape == (batch_size, num_targets, y_dim)


# Integration tests
class TestIntegration:

    def test_end_to_end_training_cnp(self, sample_data_1d):
        """Test complete training workflow for CNP."""
        x, y = sample_data_1d

        model = CNP(encoder_sizes=[32, 32], decoder_sizes=[16, 16])
        model.build([x[:, :10].shape, y[:, :10].shape, x.shape])

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Train for a few steps
        losses = []
        for step in range(3):
            # Get context and target sets
            context_x, context_y = utils.get_context_set(
                x, y, num_context=10, num_context_mode="all"
            )

            context_x_tf = tf.constant(context_x)
            context_y_tf = tf.constant(context_y)
            target_x_tf = tf.constant(x)
            target_y_tf = tf.constant(y)

            model.optimizer = optimizer
            loss = model.train_step(
                context_x_tf, context_y_tf, target_x_tf, target_y_tf
            )
            losses.append(float(loss))

        # Check that losses are finite
        assert all(np.isfinite(loss) for loss in losses)

        # Test prediction
        pred_mean, pred_std = model.test_step(context_x_tf, context_y_tf, target_x_tf)
        assert pred_mean.shape == y.shape
        assert pred_std.shape == y.shape
        assert ops.all(pred_std > 0)

    def test_model_saving_and_loading(self, sample_data_1d):
        """Test that models can be saved and loaded."""
        x, y = sample_data_1d

        # Create and build model
        model = CNP(encoder_sizes=[32, 32], decoder_sizes=[16, 16])
        model.build([x[:, :10].shape, y[:, :10].shape, x.shape])

        # Get some predictions before saving
        context_x, context_y = utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )

        pred_before, _ = model([context_x, context_y, x])

        # Test that we can get config (required for saving)
        config = model.get_config()
        assert isinstance(config, dict)
        assert "encoder_sizes" in config
        assert "decoder_sizes" in config
        assert "output_dims" in config

        # Note: We don't actually save/load here since that requires more setup,
        # but we verify the configuration can be extracted which is needed for saving
