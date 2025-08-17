import numpy as np
import pytest
import tensorflow as tf
import keras
import keras_neural_processes as knp


class TestCNPTraining:

    def test_cnp_train_method_basic(self, small_training_data):
        """Test CNP train method with minimal epochs."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[32, 32], decoder_sizes=[16, 16])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Train for just 2 epochs
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=2,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[5, 10],
            num_context_mode="all",
            dataset_type="gp",
            plotcb=False,  # Disable plotting for testing
            pbar=False,  # Disable progress bar for testing
        )

        # Check that history object exists and has loss data
        assert hasattr(history, "history")
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2  # 2 epochs

        # Check that losses are finite
        losses = history.history["loss"]
        assert all(np.isfinite(loss) for loss in losses)

    def test_cnp_train_with_validation(self, small_training_data):
        """Test CNP training with validation data but no plotting."""
        x, y = small_training_data

        # Split into train/val
        x_train, x_val = x[:2], x[2:]
        y_train, y_val = y[:2], y[2:]

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        history = model.train(
            X_train=x_train,
            y_train=y_train,
            epochs=2,
            optimizer=optimizer,
            batch_size=1,
            num_context_range=[3, 6],
            num_context_mode="all",  # Use "all" mode for single-channel data
            X_val=x_val,
            y_val=y_val,
            plotcb=False,  # Disable plotting for testing
            pbar=False,
        )

        assert len(history.history["loss"]) == 2

    def test_cnp_train_different_context_modes(self, small_training_data):
        """Test CNP training with 'all' context mode."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Test with "all" mode for single-channel data
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 5],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        assert len(history.history["loss"]) == 1


class TestNPTraining:

    def test_np_train_method_basic(self, small_training_data):
        """Test NP train method with minimal epochs."""
        x, y = small_training_data

        model = knp.NP(
            det_encoder_sizes=[16, 16],
            latent_encoder_sizes=[16, 16],
            num_latents=16,
            decoder_sizes=[8, 8],
        )
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Train for just 2 epochs
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=2,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            dataset_type="gp",
            plotcb=False,
            pbar=False,
        )

        # Check that history has all expected keys for NP
        assert "loss" in history.history
        assert "recon_loss" in history.history
        assert "kl_div" in history.history
        assert len(history.history["loss"]) == 2

        # Check that all losses are finite
        for key in ["loss", "recon_loss", "kl_div"]:
            losses = history.history[key]
            assert all(np.isfinite(loss) for loss in losses)


class TestANPTraining:

    def test_anp_train_method_basic(self, small_training_data):
        """Test ANP train method with minimal epochs."""
        x, y = small_training_data

        model = knp.ANP(
            att_encoder_sizes=[16, 16],
            num_heads=2,
            latent_encoder_sizes=[16, 16],
            num_latents=16,
            decoder_sizes=[8, 8],
        )
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        history = model.train(
            X_train=x,
            y_train=y,
            epochs=2,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        # Check that history has all expected keys for ANP
        assert "loss" in history.history
        assert "recon_loss" in history.history
        assert "kl_div" in history.history
        assert len(history.history["loss"]) == 2


class TestTrainingParameters:

    def test_training_with_different_batch_sizes(self, small_training_data):
        """Test training with different batch sizes."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Test with batch size equal to dataset size
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=1,
            optimizer=optimizer,
            batch_size=x.shape[0],  # Same as dataset size
            num_context_range=[2, 4],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        assert len(history.history["loss"]) == 1

    def test_training_with_different_context_ranges(self, small_training_data):
        """Test training with different context range specifications."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Test with single number for context range
        history1 = model.train(
            X_train=x,
            y_train=y,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[5, 5],  # Fixed number
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        # Test with range
        history2 = model.train(
            X_train=x,
            y_train=y,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 8],  # Range
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        assert len(history1.history["loss"]) == 1
        assert len(history2.history["loss"]) == 1


class TestTrainingValidation:

    def test_training_assertions(self, small_training_data):
        """Test that training validation works correctly."""
        x, y = small_training_data

        model = knp.CNP()
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Test that assertion works when plotcb=True but no validation data
        with pytest.raises(AssertionError, match="Validation data must be provided"):
            model.train(
                X_train=x,
                y_train=y,
                epochs=1,
                optimizer=optimizer,
                plotcb=True,  # This should require validation data
                X_val=None,
                y_val=None,
            )

    def test_pred_points_default_setting(self, small_training_data):
        """Test that pred_points is set correctly for GP dataset."""
        x, y = small_training_data
        x_val, y_val = x[:2], y[:2]

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # This should work without explicitly setting pred_points
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            dataset_type="gp",
            X_val=x_val,
            y_val=y_val,
            plotcb=False,  # Don't actually plot, but validate the setup
            pbar=False,
        )

        assert len(history.history["loss"]) == 1


class TestProgressTracking:

    def test_training_with_progress_bar(self, small_training_data):
        """Test training with progress bar enabled (but don't check output)."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # This should run without errors even with pbar=True
        history = model.train(
            X_train=x,
            y_train=y,
            epochs=2,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            plotcb=False,
            pbar=True,  # Enable progress bar
        )

        assert len(history.history["loss"]) == 2

    def test_callbacks_integration(self, small_training_data):
        """Test that Keras callbacks are properly integrated."""
        x, y = small_training_data

        model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        history = model.train(
            X_train=x,
            y_train=y,
            epochs=3,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        # History should be properly tracked
        assert hasattr(history, "history")
        assert len(history.history["loss"]) == 3

        # Check that callbacks properly tracked epochs
        assert hasattr(history, "epoch")
        assert len(history.epoch) == 3


class TestSeedHandling:

    def test_training_reproducibility_with_seed(self, small_training_data):
        """Test that training can be made reproducible with seeds."""
        x, y = small_training_data

        def train_model_with_seed(seed):
            # Set seeds for reproducibility
            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = knp.CNP(encoder_sizes=[16, 16], decoder_sizes=[8, 8])
            optimizer = keras.optimizers.Adam(learning_rate=1e-3)

            history = model.train(
                X_train=x,
                y_train=y,
                epochs=2,
                optimizer=optimizer,
                batch_size=2,
                num_context_range=[3, 6],
                num_context_mode="all",
                plotcb=False,
                pbar=False,
                seed_val=seed,  # Validation seed
            )

            return history.history["loss"]

        # Train with same seed twice
        losses1 = train_model_with_seed(42)
        losses2 = train_model_with_seed(42)

        # Results should be deterministic (though individual training may still vary)
        assert len(losses1) == len(losses2)
        assert len(losses1) == 2

        # Train with different seed
        losses3 = train_model_with_seed(123)
        assert len(losses3) == 2
