import keras
import tensorflow as tf
import tensorflow_probability as tfp
from keras import ops
import warnings

from ..layers.decoders import DeterministicDecoder
from ..layers.encoders import DeterministicEncoder

#### NOTE
#### In the reference below, see how the GAN can take an input model for the encoder and decoder.
#### Aim to do that for a CNP and a (Latent) NP once the code below works well.
#### You can default to the encoder and decoder currently used below (from
#### encoders.DeterministicEncoder and decoders. DeterministicDecoder) if None is provided.
#### Ref: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to-end_gan_example


# Full CNP model
class CNP(keras.Model):
    # Init is for all input-independent initialization
    def __init__(self, encoder_sizes=[128, 128, 128, 128], decoder_sizes=[128, 128]):
        super().__init__()

        # A loss tracker and an MAE
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
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
        self.encoder = DeterministicEncoder(self.encoder_sizes)
        self.decoder = DeterministicDecoder(self.decoder_sizes)

    # Call is for the forward computation
    def call(self, inputs):
        context_x, context_y, target_x = inputs

        # Get representation from encoder
        representation = self.encoder(context_x, context_y)

        # Get predictions from decoder
        mean, std = self.decoder(representation, target_x)

        return mean, std

    # We are overriding the training step for probability calculation
    def train_step(self, data):
        # Refer https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#going_lower-level
        (context_x, context_y, target_x), target_y = data

        with tf.GradientTape() as tape:
            # Forward pass
            mean, std = self((context_x, context_y, target_x), training=True)

            # Compute our own negative log likelihood loss
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            loss = -ops.mean(dist.log_prob(target_y))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(target_y, mean)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric]
