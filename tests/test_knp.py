import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

import keras_neural_processes as knp


# Tests for basic model instantiation
def test_model_instantiation(model_class):
    """Test that all models can be instantiated."""
    model = model_class()
    assert model is not None
    assert isinstance(model, model_class)


# Tests for model configuration and instantiation with custom parameters
class TestModelInstantiation:

    def test_cnp_custom_params(self):
        """Test CNP with custom parameters."""
        model = knp.CNP(
            encoder_sizes=[64, 64], decoder_sizes=[32, 32], x_dims=2, y_dims=2
        )
        assert model.encoder_sizes == [64, 64]
        assert model.decoder_sizes == [32, 32]
        assert model.x_dims == 2
        assert model.y_dims == 2

    def test_np_custom_params(self):
        """Test NP with custom parameters."""
        model = knp.NP(
            det_encoder_sizes=[64, 64],
            latent_encoder_sizes=[32, 32],
            num_latents=64,
            decoder_sizes=[16, 16],
            x_dims=2,
            y_dims=2,
        )
        assert model.det_encoder_sizes == [64, 64]
        assert model.latent_encoder_sizes == [32, 32]
        assert model.num_latents == 64
        assert model.decoder_sizes == [16, 16]
        assert model.x_dims == 2
        assert model.y_dims == 2

    def test_anp_custom_params(self):
        """Test ANP with custom parameters."""
        model = knp.ANP(
            att_encoder_sizes=[64, 64],
            num_heads=4,
            latent_encoder_sizes=[32, 32],
            num_latents=64,
            decoder_sizes=[16, 16],
            x_dims=2,
            y_dims=2,
        )
        assert model.att_encoder_sizes == [64, 64]
        assert model.num_heads == 4
        assert model.latent_encoder_sizes == [32, 32]
        assert model.num_latents == 64
        assert model.decoder_sizes == [16, 16]
        assert model.x_dims == 2
        assert model.y_dims == 2


# Tests for model building and forward passes
class TestModelFunctionality:
    def test_model_build_and_call(self, model_class, context_target_data):
        """Test model build and forward pass for all model types."""
        context_x, context_y, target_x, target_y = context_target_data
        model = model_class(x_dims=1)

        if model_class == knp.CNP:
            # CNP returns mean and std directly
            mean, std = model([context_x, context_y, target_x])
            assert mean.shape == target_y.shape
            assert std.shape == target_y.shape
            assert ops.all(std > 0)
        else:
            # NP and ANP return distributions
            # Test training pass
            pred_dist_train, prior_train, posterior_train = model(
                [context_x, context_y, target_x, target_y], training=True
            )
            assert pred_dist_train.mean().shape == target_y.shape
            assert pred_dist_train.stddev().shape == target_y.shape
            assert prior_train is not None
            assert posterior_train is not None

            # Test inference pass
            pred_dist_test, prior_test, posterior_test = model(
                [context_x, context_y, target_x, None], training=False
            )
            assert pred_dist_test.mean().shape == target_y.shape
            assert pred_dist_test.stddev().shape == target_y.shape
            assert prior_test is not None
            # Posterior should be None during inference
            assert posterior_test is None


# Tests for individual layers
class TestLayers:

    def test_deterministic_encoder(self, context_target_data):
        """Test MeanEncoder."""
        context_x, context_y, _, _ = context_target_data

        encoder = knp.MeanEncoder(sizes=[64, 64, 32])
        encoder.build([context_x.shape, context_y.shape])

        representation = encoder(context_x, context_y)

        expected_shape = (context_x.shape[0], 1, 32)  # batch, 1, hidden_size
        assert representation.shape == expected_shape

    def test_latent_encoder(self, context_target_data):
        """Test LatentEncoder."""
        context_x, context_y, _, _ = context_target_data

        encoder = knp.LatentEncoder(sizes=[64, 64], num_latents=32)
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

        encoder = knp.AttentiveEncoder(encoder_sizes=[64, 64, 32], num_heads=4)
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

        decoder = knp.Decoder(sizes=[32, 32, 2])
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

        context_x, context_y = knp.utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="all"
        )

        assert context_x.shape == (x.shape[0], num_context, x.shape[2])
        assert context_y.shape == (y.shape[0], num_context, y.shape[2])

    def test_get_context_set_each_mode(self, sample_data_multi_channel):
        """Test get_context_set with 'each' mode."""
        x, y = sample_data_multi_channel
        num_context = 10

        context_x, context_y = knp.utils.get_context_set(
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

        context_x, context_y = knp.utils.get_context_set(
            x, y, num_context=num_context, num_context_mode="each"
        )

        expected_total_points = sum(num_context)
        assert context_x.shape == (x.shape[0], expected_total_points, x.shape[2])
        assert context_y.shape == (y.shape[0], expected_total_points, y.shape[2])

    def test_get_train_batch(self, sample_data_1d):
        """Test get_train_batch function."""
        x, y = sample_data_1d
        batch_size = 2

        x_batch, y_batch = knp.utils.get_train_batch(x, y, batch_size=batch_size)

        assert x_batch.shape == (batch_size, x.shape[1], x.shape[2])
        assert y_batch.shape == (batch_size, y.shape[1], y.shape[2])

    def test_sample_num_context(self):
        """Test _sample_num_context function."""
        # Test simple range
        num_context_range = [5, 10]
        sampled = knp.utils._sample_num_context(num_context_range)
        assert isinstance(sampled, (int, np.integer))
        assert 5 <= sampled <= 10

        # Test per-channel range
        per_channel_range = [[3, 5], [7, 10]]
        sampled = knp.utils._sample_num_context(per_channel_range)
        assert isinstance(sampled, list)
        assert len(sampled) == 2
        assert 3 <= sampled[0] <= 5
        assert 7 <= sampled[1] <= 10

    def test_get_fixed_num_context(self):
        """Test _get_fixed_num_context function."""
        # Test simple range
        num_context_range = [5, 11]  # Mean should be 8
        fixed = knp.utils._get_fixed_num_context(num_context_range)
        assert fixed == 8

        # Test per-channel range
        per_channel_range = [[3, 5], [7, 11]]  # Means should be [4, 9]
        fixed = knp.utils._get_fixed_num_context(per_channel_range)
        assert fixed == [4, 9]


# Tests for training methods
class TestTrainingMethods:
    def test_model_train_step(self, model_class, context_target_data):
        """Test the train_step method for all model types."""
        context_x, context_y, target_x, target_y = context_target_data
        model = model_class(x_dims=1)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.optimizer = optimizer

        # Manually prepare data for the raw train_step
        context_x_prep = model._prepare_x(context_x)
        context_y_prep = model._prepare_y(context_y)
        target_x_prep = model._prepare_x(target_x)
        target_y_prep = model._prepare_y(target_y)

        logs = model.train_step(
            context_x_prep, context_y_prep, target_x_prep, target_y_prep
        )

        assert "loss" in logs
        assert not tf.math.is_nan(logs["loss"])

        if model_class in [knp.NP, knp.ANP]:
            assert "recon_loss" in logs
            assert "kl_div" in logs
            assert not tf.math.is_nan(logs["recon_loss"])
            assert not tf.math.is_nan(logs["kl_div"])

    def test_cnp_test_step(self, context_target_data):
        """Test CNP test_step/predict method."""
        context_x, context_y, target_x, _ = context_target_data

        model = knp.CNP(x_dims=1)

        pred_mean, pred_std = model.predict(context_x, context_y, target_x)

        assert pred_mean.shape == target_x.shape
        assert pred_std.shape == target_x.shape
        assert ops.all(pred_std > 0)


# Tests for error handling and edge cases
class TestErrorHandling:
    def test_incompatible_shapes_predict(self):
        """Test error handling for incompatible shapes in predict."""
        model = knp.CNP(x_dims=1)

        # Test mismatched context_x and context_y batch dimensions
        context_x = np.random.randn(2, 10, 1).astype(np.float32)
        context_y = np.random.randn(3, 10, 1).astype(np.float32)  # Different batch size
        pred_x = np.random.randn(2, 20, 1).astype(np.float32)

        with pytest.raises(ValueError, match="same number of samples and points"):
            _ = model.predict(context_x, context_y, pred_x)

    def test_incompatible_shapes_train(self):
        """Test error handling for incompatible shapes in train."""
        model = knp.CNP(x_dims=1)
        optimizer = keras.optimizers.Adam()

        # Mismatched samples
        x_train = np.random.randn(2, 10, 1).astype(np.float32)
        y_train = np.random.randn(3, 10, 1).astype(np.float32)

        with pytest.raises(ValueError, match="same number of samples and points"):
            model.train(x_train, y_train, epochs=1, optimizer=optimizer)

    def test_invalid_x_shape_strict(self):
        """Test error for invalid x shape when auto-reshaping is off."""
        model = knp.CNP(x_dims=3)  # Model expects 3D x
        context_x = np.random.randn(2, 10, 2).astype(np.float32)  # Data is 2D
        context_y = np.random.randn(2, 10, 1).astype(np.float32)
        pred_x = np.random.randn(2, 20, 2).astype(np.float32)

        with pytest.raises(ValueError, match="must have shape \\(..., 3\\)"):
            model.predict(context_x, context_y, pred_x)

    def test_invalid_y_shape(self):
        """Test error for invalid y shape (not matching y_dims)."""
        model = knp.CNP(x_dims=1, y_dims=1)
        context_x = np.random.randn(2, 10, 1).astype(np.float32)
        context_y = np.random.randn(2, 10, 2).astype(np.float32)  # y is 2D
        pred_x = np.random.randn(2, 20, 1).astype(np.float32)

        with pytest.raises(
            ValueError,
            match=r"context_y must have shape \(num_samples, num_points, 1\)",
        ):
            model.predict(context_x, context_y, pred_x)

    def test_get_context_set_invalid_mode(self, sample_data_1d):
        """Test get_context_set with invalid mode."""
        x, y = sample_data_1d

        with pytest.raises(AttributeError):
            knp.utils.get_context_set(x, y, num_context=10, num_context_mode="invalid")

    def test_get_context_set_too_many_points(self, sample_data_1d):
        """Test get_context_set requesting more points than available."""
        x, y = sample_data_1d
        num_points = x.shape[1]

        with pytest.raises(ValueError):
            knp.utils.get_context_set(
                x, y, num_context=num_points + 10, num_context_mode="all"
            )


# Tests for model serialization
class TestModelSerialization:

    def test_cnp_get_config(self):
        """Test CNP get_config method."""
        model = knp.CNP(
            encoder_sizes=[64, 32], decoder_sizes=[32, 16], x_dims=2, y_dims=2
        )
        config = model.get_config()

        assert config["encoder_sizes"] == [64, 32]
        assert config["decoder_sizes"] == [32, 16]
        assert config["x_dims"] == 2
        assert config["y_dims"] == 2

    def test_np_get_config(self):
        """Test NP get_config method."""
        model = knp.NP(
            det_encoder_sizes=[64, 32],
            latent_encoder_sizes=[32, 16],
            num_latents=64,
            decoder_sizes=[32, 16],
            x_dims=2,
            y_dims=2,
        )
        config = model.get_config()

        assert config["det_encoder_sizes"] == [64, 32]
        assert config["latent_encoder_sizes"] == [32, 16]
        assert config["num_latents"] == 64
        assert config["decoder_sizes"] == [32, 16]
        assert config["x_dims"] == 2
        assert config["y_dims"] == 2

    def test_anp_get_config(self):
        """Test ANP get_config method."""
        model = knp.ANP(
            att_encoder_sizes=[64, 32],
            num_heads=4,
            latent_encoder_sizes=[32, 16],
            num_latents=64,
            decoder_sizes=[32, 16],
            x_dims=2,
            y_dims=2,
        )
        config = model.get_config()

        assert config["att_encoder_sizes"] == [64, 32]
        assert config["num_heads"] == 4
        assert config["latent_encoder_sizes"] == [32, 16]
        assert config["num_latents"] == 64
        assert config["decoder_sizes"] == [32, 16]
        assert config["x_dims"] == 2
        assert config["y_dims"] == 2


# Tests for layers get_config methods
class TestLayerSerialization:

    def test_deterministic_encoder_config(self):
        """Test MeanEncoder get_config method."""
        encoder = knp.MeanEncoder(sizes=[64, 32, 16])
        config = encoder.get_config()
        assert config["sizes"] == [64, 32, 16]

    def test_latent_encoder_config(self):
        """Test LatentEncoder get_config method."""
        encoder = knp.LatentEncoder(sizes=[64, 32], num_latents=16)
        config = encoder.get_config()
        assert config["sizes"] == [64, 32]
        assert config["num_latents"] == 16

    def test_attentive_encoder_config(self):
        """Test AttentiveEncoder get_config method."""
        encoder = knp.AttentiveEncoder(encoder_sizes=[64, 32], num_heads=4)
        config = encoder.get_config()
        assert config["encoder_sizes"] == [64, 32]
        assert config["num_heads"] == 4

    def test_deterministic_decoder_config(self):
        """Test Decoder get_config method."""
        decoder = knp.Decoder(sizes=[64, 32, 2])
        config = decoder.get_config()
        assert config["sizes"] == [64, 32, 2]


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
        context_x, context_y = knp.utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )

        metrics = knp.utils.gplike_calculate_mymetrics(
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
        assert not knp.utils._is_per_channel_range([5, 10])

        # Test per-channel range
        assert knp.utils._is_per_channel_range([[5, 10], [3, 8]])

        # Test empty input
        assert not knp.utils._is_per_channel_range([])

        # Test invalid format
        assert not knp.utils._is_per_channel_range([5, [3, 8]])


# Tests for tensor vs numpy array handling
class TestTensorVsNumpyInputs:

    def test_cnp_with_tf_tensors(self, context_target_data):
        """Test CNP with TensorFlow tensor inputs."""
        context_x, context_y, target_x, target_y = context_target_data

        # Convert to TensorFlow tensors
        context_x_tf = tf.constant(context_x)
        context_y_tf = tf.constant(context_y)
        target_x_tf = tf.constant(target_x)

        xdim, ydim = context_x.shape[2], context_y.shape[2]
        model = knp.CNP(x_dims=xdim, y_dims=ydim)

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

        context_x, context_y = knp.utils.get_context_set(
            x_tf, y_tf, num_context=10, num_context_mode="all"
        )

        assert isinstance(context_x, tf.Tensor)
        assert isinstance(context_y, tf.Tensor)
        assert context_x.shape == (x.shape[0], 10, x.shape[2])
        assert context_y.shape == (y.shape[0], 10, y.shape[2])


# Tests for different input/output dimensions
class TestDifferentDimensions:
    def test_multi_output(self, model_class):
        """Test models with multi-dimensional output."""
        batch_size = 2
        num_context = 5
        num_targets = 10
        x_dim = 1
        y_dim = 3  # Multi-dimensional output

        context_x = np.random.randn(batch_size, num_context, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, num_context, y_dim).astype(np.float32)
        target_x = np.random.randn(batch_size, num_targets, x_dim).astype(np.float32)
        target_y = np.random.randn(batch_size, num_targets, y_dim).astype(np.float32)

        model = model_class(x_dims=x_dim, y_dims=y_dim)
        optimizer = keras.optimizers.Adam()
        model.optimizer = optimizer

        # Test predict
        mean, std = model.predict(context_x, context_y, target_x)
        assert mean.shape == (batch_size, num_targets, y_dim)
        assert std.shape == (batch_size, num_targets, y_dim)

        # Test train_step for models that support it in this configuration
        if model_class in [knp.NP, knp.ANP]:
            context_x_prep = model._prepare_x(context_x)
            context_y_prep = model._prepare_y(context_y)
            target_x_prep = model._prepare_x(target_x)
            target_y_prep = model._prepare_y(target_y)
            logs = model.train_step(
                context_x_prep, context_y_prep, target_x_prep, target_y_prep
            )
            assert "loss" in logs

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

        model = knp.CNP(x_dims=x_dim, y_dims=y_dim)

        mean, std = model.predict(context_x, context_y, target_x)

        assert mean.shape == (batch_size, num_targets, y_dim)
        assert std.shape == (batch_size, num_targets, y_dim)

    def test_multi_input_cnp_strict(self):
        """Test CNP with multi-dimensional input and strict checking."""
        batch_size = 2
        num_context = 5
        num_targets = 10
        x_dim = 3  # Not 1 or 2, so no auto-reshaping
        y_dim = 1

        context_x = np.random.randn(batch_size, num_context, x_dim).astype(np.float32)
        context_y = np.random.randn(batch_size, num_context, y_dim).astype(np.float32)
        target_x = np.random.randn(batch_size, num_targets, x_dim).astype(np.float32)

        model = knp.CNP(x_dims=x_dim, y_dims=y_dim)

        # This should work fine
        mean, std = model.predict(context_x, context_y, target_x)
        assert mean.shape == (batch_size, num_targets, y_dim)
        assert std.shape == (batch_size, num_targets, y_dim)

        # This should fail because the last dim is not x_dim
        bad_context_x = np.random.randn(batch_size, num_context, 1).astype(np.float32)
        with pytest.raises(ValueError, match=f"must have shape \\(..., {x_dim}\\)"):
            model.predict(bad_context_x, context_y, target_x)


# Integration tests
class TestIntegration:

    def test_end_to_end_training_cnp(self, sample_data_1d):
        """Test complete training workflow for CNP."""
        x, y = sample_data_1d

        model = knp.CNP(encoder_sizes=[32, 32], decoder_sizes=[16, 16], x_dims=1)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        history = model.train(
            x,
            y,
            epochs=3,
            optimizer=optimizer,
            batch_size=2,
            pbar=False,
            plotcb=False,
            num_context_mode="all",  # Use 'all' for 1D x data
        )

        # Check that losses are finite
        assert all(np.isfinite(loss) for loss in history.history["loss"])

        # Test prediction
        context_x, context_y = knp.utils.get_context_set(
            x, y, num_context=10, num_context_mode="all"
        )
        pred_mean, pred_std = model.predict(context_x, context_y, x)
        assert pred_mean.shape == y.shape
        assert pred_std.shape == y.shape
        assert ops.all(pred_std > 0)

    def test_model_saving_and_loading(self, tmp_path):
        """Test that models can be saved and loaded."""
        model = knp.CNP(
            encoder_sizes=[32, 32], decoder_sizes=[16, 16], x_dims=1, y_dims=1
        )
        model.save(tmp_path / "model.keras")
        loaded_model = keras.saving.load_model(tmp_path / "model.keras")

        assert loaded_model.encoder_sizes == [32, 32]
        assert loaded_model.decoder_sizes == [16, 16]
        assert loaded_model.x_dims == 1
        assert loaded_model.y_dims == 1

    def test_model_saving_and_loading_custom_dims(self, tmp_path):
        """Test saving/loading with custom x_dims and y_dims."""
        model = knp.CNP(
            encoder_sizes=[32, 32], decoder_sizes=[16, 16], x_dims=3, y_dims=4
        )
        model.save(tmp_path / "model.keras")
        loaded_model = keras.saving.load_model(tmp_path / "model.keras")

        assert loaded_model.x_dims == 3
        assert loaded_model.y_dims == 4


class TestModelBehavior:
    def test_model_training_flag(self, model_class, context_target_data):
        """Test that the training flag is passed correctly in all models."""
        context_x, context_y, target_x, target_y = context_target_data
        model = model_class(x_dims=1)

        # Mock the call method to check the training argument
        original_call = model.call
        call_args = {}

        def mocked_call(*args, **kwargs):
            call_args.update(kwargs)
            return original_call(*args, **kwargs)

        model.call = mocked_call

        # Test during training step
        model.optimizer = keras.optimizers.Adam()
        context_x_prep = model._prepare_x(context_x)
        context_y_prep = model._prepare_y(context_y)
        target_x_prep = model._prepare_x(target_x)
        target_y_prep = model._prepare_y(
            np.random.randn(*target_x.shape).astype(np.float32)
        )
        model.train_step(context_x_prep, context_y_prep, target_x_prep, target_y_prep)
        assert call_args.get("training") is True

        # Test during predict
        model.predict(context_x, context_y, target_x)
        assert call_args.get("training") is False
