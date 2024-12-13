import keras
import tensorflow as tf
import tensorflow_probability as tfp
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
import warnings

# from tqdm.auto import tqdm
from keras.utils import Progbar

from ..layers import encoders, decoders
from ..utils.data import get_train_batch, get_context_set
from ..utils.metrics import calculate_mymetrics
from ..utils.plotting import plot_functions

#### NOTE
#### In the reference below, see how the GAN can take an input model for the encoder and decoder.
#### Aim to do that for a CNP and a (Latent) NP once the code below works well.
#### You can default to the encoder and decoder currently used below (from
#### encoders.DeterministicEncoder and decoders. DeterministicDecoder) if None is provided.
#### Ref: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to-end_gan_example


# Full CNP model
class CNP(keras.Model):
    # Init is for all input-independent initialization
    def __init__(self, encoder_sizes=[128, 128, 128, 128], decoder_sizes=[128, 128, 2]):
        super().__init__()
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes

    # Build is for all initialization that depends on input_shape
    def build(self, input_shape):
        context_x_shape, context_y_shape, _ = input_shape

        xdims = context_x_shape[-1]  # No. of channels / dims in x
        ydims = context_y_shape[-1]  # No. of channels / dims in y
        if xdims > 1:
            warnings.warn(
                "Note: The encoder implementation doesn't implicitly "
                "depend on xdims. Ensure you use an encoder size that is "
                "appropriate for your data.",
                UserWarning,
            )

        # Final layer of decoder should
        # have a mu and sigma for each ydim
        self.decoder_sizes.append(2 * ydims)

        # Set encoder and decoder
        self.encoder = encoders.DeterministicEncoder(self.encoder_sizes)
        self.decoder = decoders.DeterministicDecoder(self.decoder_sizes)

    # Call is for the forward computation
    def call(self, inputs):  # double check training default
        context_x, context_y, target_x = inputs

        # Get representation from encoder
        representation = self.encoder(context_x, context_y)

        # Get predictions from decoder
        mean, std = self.decoder(representation, target_x)

        return mean, std

    @tf.function(jit_compile=True)
    def train_step(self, context_x, context_y, target_x, target_y):
        with tf.GradientTape() as tape:
            # Forward pass
            mean, std = self((context_x, context_y, target_x), training=True)
            # Compute negative log likelihood loss
            dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            loss_value = -ops.mean(dist.log_prob(target_y))
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)
        return loss_value

    @tf.function(jit_compile=True)
    def test_step(self, context_x, context_y, pred_x):
        # Get model predictions and uncertainty estimates
        pred_y_mean, pred_y_std = self((context_x, context_y, pred_x))
        return pred_y_mean, pred_y_std

    def train(
        self,
        X_train,
        y_train,
        epochs,
        optimizer,
        batch_size=64,
        X_val=None,
        y_val=None,
        plotcb=True,
        pbar=True,
        pred_points=400,
        plot_every=1000,
    ):
        """Custom training loop for the CNP model.

        Args:
            optimizer: Keras optimizer instance
            epochs: Number of training epochs
        """

        # assert batch_size == "all" # batching not supported yet
        self.optimizer = optimizer
        if pbar:
            progbar = Progbar(epochs)
        if plotcb:
            assert X_val is not None and y_val is not None

        # Initialize callbacks
        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)

        # Configure callbacks
        callbacks.on_train_begin()  # Called at the start of training

        for epoch in range(1, epochs + 1):
            # Reset metrics at start of each epoch
            callbacks.on_epoch_begin(epoch)  # Called at start of epoch

            # batching strategy: sample randomly from big train set
            X_train_batch, y_train_batch = get_train_batch(
                X_train, y_train, batch_size=batch_size
            )

            # sample context set from fixed target set
            context_x_train, context_y_train = get_context_set(
                X_train_batch,
                y_train_batch,
                num_context=np.random.randint(low=3, high=10 + 1),
            )

            target_x_train = X_train_batch
            target_y_train = y_train_batch

            loss = self.train_step(
                context_x_train, context_y_train, target_x_train, target_y_train
            )

            # Update metrics with both loss and mae if reqd.
            logs = {
                "loss": float(loss),
            }
            callbacks.on_epoch_end(epoch, logs)  # Called at end of epoch

            if pbar:
                progbar.update(epoch, values=[("loss", float(loss))])

            if plotcb and (epoch % plot_every == 0):
                context_x_val, context_y_val = get_context_set(
                    X_val, y_val, num_context=np.random.randint(low=2, high=10 + 1)
                )
                target_x_val, target_y_val = X_val, y_val

                # Create evenly spaced points for prediction
                pred_x_val = np.linspace(-2, 2, pred_points).reshape(
                    target_x_val.shape[0], pred_points, target_x_val.shape[2]
                )

                # Get model predictions and uncertainty estimates
                pred_y_mean, pred_y_std = self.test_step(
                    context_x_val, context_y_val, pred_x_val
                )

                # Calc and print val_loss
                dist = tfp.distributions.MultivariateNormalDiag(pred_y_mean, pred_y_std)
                val_loss = -ops.mean(dist.log_prob(target_y_val))
                print(f"\nEpoch{epoch}, val_loss={val_loss} (lower is better)")

                # Calc and print other metrics
                mymetrics = calculate_mymetrics(
                    pred_x_val,
                    pred_y_mean,
                    pred_y_std,
                    target_x_val,
                    target_y_val,
                    context_x_val,
                    context_y_val,
                )
                for metric_name, value in mymetrics.items():
                    print(f"{metric_name}: {value:.1%} (higher is better)")
                print(f"num_context_points = {context_x_val.shape[1]}")
                # Visualize predictions
                plot_functions(
                    target_x=target_x_val,
                    target_y=target_y_val,
                    context_x=context_x_val,
                    context_y=context_y_val,
                    pred_y=pred_y_mean,
                    std=pred_y_std,
                )
                plt.title(f"Epoch {epoch}")
                plt.legend()
                plt.show()
                plt.close()

        return history
