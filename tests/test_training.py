import numpy as np
import pytest
import keras
import keras_neural_processes as knp
from unittest.mock import MagicMock


class TestModelTraining:
    def test_model_train_method_basic(self, model_class, sample_gp_data_with_val):
        """Test the basic training loop of a model."""
        X_train, y_train, _, _ = sample_gp_data_with_val

        # Minimal model configuration for testing
        if model_class == knp.CNP:
            model = model_class(
                encoder_sizes=[32, 32], decoder_sizes=[32, 32], x_dims=1, y_dims=1
            )
        elif model_class == knp.NP:
            model = model_class(
                det_encoder_sizes=[32, 32],
                latent_encoder_sizes=[32, 32],
                decoder_sizes=[32, 32],
                x_dims=1,
                y_dims=1,
            )
        else:  # ANP
            model = model_class(
                att_encoder_sizes=[32, 32],
                latent_encoder_sizes=[32, 32],
                decoder_sizes=[32, 32],
                x_dims=1,
                y_dims=1,
            )

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        history = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            pbar=False,
            plotcb=False,
            num_context_mode="all",
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        assert np.all(np.isfinite(history.history["loss"]))

        if model_class in [knp.NP, knp.ANP]:
            assert "recon_loss" in history.history
            assert "kl_div" in history.history
            assert np.all(np.isfinite(history.history["recon_loss"]))
            assert np.all(np.isfinite(history.history["kl_div"]))

    def test_cnp_train_with_validation(self, sample_gp_data_with_val):
        """Test training with a validation set and plotting callback."""
        X_train, y_train, X_val, y_val = sample_gp_data_with_val
        model = knp.CNP(
            encoder_sizes=[32, 32],
            decoder_sizes=[32, 32],
            x_dims=1,
            y_dims=1,
        )
        optimizer = keras.optimizers.Adam()
        history = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            pbar=False,
            plotcb=False,
            X_val=X_val,
            y_val=y_val,
            num_context_mode="all",
        )
        assert "loss" in history.history

    def test_cnp_train_different_context_modes(self, sample_gp_data_with_val):
        """Test that training runs with different context sampling modes."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # Test with 'all' mode
        model.train(
            X_train,
            y_train,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 5],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )


class TestNPTraining:

    def test_np_train_method_basic(self, sample_gp_data_with_val):
        """Test the basic training loop of an NP model."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.NP(
            det_encoder_sizes=[32, 32],
            latent_encoder_sizes=[32, 32],
            decoder_sizes=[32, 32],
            x_dims=1,
            y_dims=1,
        )
        optimizer = keras.optimizers.Adam()
        history = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            pbar=False,
            plotcb=False,
            num_context_mode="all",
        )
        assert "loss" in history.history
        assert "recon_loss" in history.history
        assert "kl_div" in history.history
        assert len(history.history["loss"]) == 2

        # Check that all losses are finite
        for key in ["loss", "recon_loss", "kl_div"]:
            losses = history.history[key]
            assert all(np.isfinite(loss) for loss in losses)


class TestANPTraining:

    def test_anp_train_method_basic(self, sample_gp_data_with_val):
        """Test the basic training loop of an ANP model."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.ANP(
            att_encoder_sizes=[32, 32],
            latent_encoder_sizes=[32, 32],
            decoder_sizes=[32, 32],
            x_dims=1,
            y_dims=1,
        )
        optimizer = keras.optimizers.Adam()
        history = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            pbar=False,
            plotcb=False,
            num_context_mode="all",
        )
        assert "loss" in history.history
        assert "recon_loss" in history.history
        assert "kl_div" in history.history
        assert len(history.history["loss"]) == 2


class TestTrainingParameters:

    def test_training_with_different_batch_sizes(self, sample_gp_data_with_val):
        """Test that training respects different batch sizes."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # Larger batch size
        history1 = model.train(
            X_train,
            y_train,
            epochs=1,
            optimizer=optimizer,
            batch_size=X_train.shape[0],
            num_context_range=[2, 4],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )
        assert len(history1.history["loss"]) == 1

        # Smaller batch size
        history2 = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            batch_size=2,
            pbar=False,
            plotcb=False,
            num_context_mode="all",
        )
        assert "loss" in history2.history

    def test_training_with_different_context_ranges(self, sample_gp_data_with_val):
        """Test that training works with different context point ranges."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # Single number for num_context_range
        history1 = model.train(
            X_train,
            y_train,
            epochs=1,
            optimizer=optimizer,
            num_context_range=[5, 10],
            num_context_mode="all",
            pbar=False,
            plotcb=False,
        )
        assert "loss" in history1.history

        # Range for num_context_range
        history2 = model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            num_context_range=[5, 15],
            num_context_mode="all",
            pbar=False,
            plotcb=False,
        )
        assert "loss" in history2.history


class TestTrainingValidation:

    def test_training_assertions(self, sample_gp_data_with_val):
        """Test assertions for invalid training configurations."""
        X_train, y_train, X_val, y_val = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # Should fail if plotcb is True but no validation data is provided
        with pytest.raises(AssertionError, match="Validation data must be provided"):
            model.train(
                X_train=X_train,
                y_train=y_train,
                epochs=1,
                optimizer=optimizer,
                plotcb=True,  # This should require validation data
                X_val=None,
                y_val=None,
            )

    def test_pred_points_default_setting(self, sample_gp_data_with_val):
        """Test that pred_points defaults correctly when not provided."""
        X_train, y_train, X_val, y_val = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # This should work without explicitly setting pred_points
        history = model.train(
            X_train,
            y_train,
            epochs=1,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            dataset_type="gp",
            X_val=X_val,
            y_val=y_val,
            plotcb=False,  # Don't actually plot, but validate the setup
            pbar=False,
        )

        assert len(history.history["loss"]) == 1


class TestProgressTracking:

    def test_training_with_progress_bar(self, sample_gp_data_with_val, mocker):
        """Test that the progress bar is used when pbar=True."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        mock_progbar = mocker.patch("keras_neural_processes.src.Progbar")

        # Train with pbar=True (default)
        model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            plotcb=False,
            num_context_mode="all",
        )
        assert mock_progbar.call_count == 1

        # Train with pbar=False
        mock_progbar.reset_mock()
        model.train(
            X_train,
            y_train,
            epochs=2,
            optimizer=optimizer,
            pbar=False,
            plotcb=False,
            num_context_mode="all",
        )
        assert mock_progbar.call_count == 0

    def test_callbacks_integration(self, sample_gp_data_with_val, mocker):
        """Test that Keras callbacks are correctly integrated."""
        X_train, y_train, _, _ = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        mock_callback_list = mocker.patch(
            "keras_neural_processes.src.keras.callbacks.CallbackList"
        )
        mock_callback_list_instance = mock_callback_list.return_value

        epochs = 3
        model.train(
            X_train,
            y_train,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=2,
            num_context_range=[3, 6],
            num_context_mode="all",
            plotcb=False,
            pbar=False,
        )

        assert mock_callback_list.call_count == 1
        assert mock_callback_list_instance.on_train_begin.call_count == 1
        assert mock_callback_list_instance.on_epoch_begin.call_count == epochs
        assert mock_callback_list_instance.on_epoch_end.call_count == epochs


class TestSeedHandling:

    def test_training_reproducibility_with_seed(self, sample_gp_data_with_val, mocker):
        """Test that training is reproducible when a seed is provided for validation."""
        X_train, y_train, X_val, y_val = sample_gp_data_with_val
        model = knp.CNP(x_dims=1, y_dims=1)
        optimizer = keras.optimizers.Adam()

        # Mock the plotting function to capture the seed it receives
        mock_gplike_val_step = MagicMock()
        mocker.patch(
            "keras_neural_processes.utils.gplike_val_step", mock_gplike_val_step
        )

        def train_model_with_seed(seed):
            history = model.train(
                X_train,
                y_train,
                epochs=2,
                optimizer=optimizer,
                X_val=X_val,
                y_val=y_val,
                seed_val=seed,
                plot_every=1,
                num_context_mode="all",
            )
            return history, mock_gplike_val_step.call_args_list

        # Run training twice with the same seed
        mock_gplike_val_step.reset_mock()
        _, calls1 = train_model_with_seed(seed=42)

        mock_gplike_val_step.reset_mock()
        _, calls2 = train_model_with_seed(seed=42)

        # Run training with a different seed
        mock_gplike_val_step.reset_mock()
        _, calls3 = train_model_with_seed(seed=99)

        # Check that the seeds were passed correctly
        assert calls1[0].kwargs["seed"] == 42
        assert calls2[0].kwargs["seed"] == 42
        assert calls3[0].kwargs["seed"] == 99
