import warnings

import keras
from keras import layers, ops
from keras.utils import Progbar
import tensorflow as tf
import tensorflow_probability as tfp

from . import utils


# =============================================================================
# 1. COMPONENT LAYERS (THE BUILDING BLOCKS)
# =============================================================================


# First, let's define the aggregator layers for composability
@keras.saving.register_keras_serializable()
class MeanAggregator(layers.Layer):
    """
    Aggregates point representations by taking their mean.
    Suitable for deterministic encoders that treat all points equally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch, num_points, features) or (batch, features)
        Returns:
            Tensor of shape (batch, 1, features)
        """
        if len(ops.shape(inputs)) == 2:
            # Input is already aggregated, just add the points dimension
            return ops.expand_dims(inputs, axis=1)
        else:
            # Input has points dimension, aggregate over it
            return ops.mean(inputs, axis=1, keepdims=True)


@keras.saving.register_keras_serializable()
class AttentionAggregator(layers.Layer):
    """
    Aggregates context representations using attention mechanism.
    Provides target-specific representations based on cross-attention.
    """

    def __init__(self, num_heads, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        # These will be initialized when we know the representation dimension
        self.self_attention = None
        self.cross_attention = None
        self.self_attention_norm = layers.LayerNormalization()
        self.cross_attention_norm = layers.LayerNormalization()

    def build(self, input_shape):
        # input_shape should be [(batch, num_context, rep_dim), (batch, num_targets, x_dim)]
        context_shape, target_shape = input_shape
        rep_dim = context_shape[-1]

        if self.key_dim is None:
            self.key_dim = rep_dim

        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )

        # Query projection for target_x
        self.query_projection = layers.Dense(rep_dim)

        super().build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: Tuple of (context_representations, target_x)
                context_representations: (batch, num_context, rep_dim)
                target_x: (batch, num_targets, x_dim)
        Returns:
            Tensor of shape (batch, num_targets, rep_dim)
        """
        context_representations, target_x = inputs

        # Apply self-attention to context representations
        self_att_output = self.self_attention(
            query=context_representations,
            value=context_representations,
            key=context_representations,
        )
        context_representations = self.self_attention_norm(
            context_representations + self_att_output
        )

        # Project target_x to create queries for cross-attention
        queries = self.query_projection(target_x)

        # Apply cross-attention
        cross_att_output = self.cross_attention(
            query=queries,
            value=context_representations,
            key=context_representations,
        )

        # Final target-specific representations
        final_representations = self.cross_attention_norm(queries + cross_att_output)
        return final_representations

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config


# Now the compositional encoders that use processors + aggregators
@keras.saving.register_keras_serializable()
class DeterministicEncoder(layers.Layer):
    """
    A compositional deterministic encoder that combines a processor and aggregator.

    Args:
        processor: A Keras model/layer that processes (context_x, context_y) -> representations
        aggregator: An aggregator layer (MeanAggregator or AttentionAggregator)
        target_x_required: Whether the aggregator needs target_x (True for AttentionAggregator)
    """

    def __init__(self, processor, aggregator, target_x_required=False, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.aggregator = aggregator
        self.target_x_required = target_x_required

    def call(self, inputs):
        if self.target_x_required:
            # For attention-based aggregators
            context_x, context_y, target_x = inputs
            # Concatenate context inputs
            context_input = ops.concatenate([context_x, context_y], axis=-1)
            # Process through the network
            context_representations = self.processor(context_input)
            # Aggregate with target_x information
            return self.aggregator([context_representations, target_x])
        else:
            # For mean-based aggregators
            context_x, context_y = inputs
            # Concatenate context inputs
            context_input = ops.concatenate([context_x, context_y], axis=-1)
            # Process through the network
            context_representations = self.processor(context_input)
            # Aggregate
            return self.aggregator(context_representations)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "processor": self.processor,
                "aggregator": self.aggregator,
                "target_x_required": self.target_x_required,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class CompositionalLatentEncoder(layers.Layer):
    """
    A compositional latent encoder for variational models.

    Args:
        processor: A Keras model/layer that processes (x, y) -> hidden representations
        num_latents: Dimensionality of the latent space
        aggregator: Optional aggregator (defaults to MeanAggregator)
    """

    def __init__(self, processor, num_latents, aggregator=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.num_latents = num_latents
        self.aggregator = aggregator if aggregator is not None else MeanAggregator()

        # Latent parameter layers
        self.mu_layer = layers.Dense(num_latents)
        self.log_sigma_layer = layers.Dense(num_latents)

    def call(self, inputs):
        x, y = inputs
        # Concatenate inputs
        encoder_input = ops.concatenate([x, y], axis=-1)
        # Process through the network
        hidden = self.processor(encoder_input)
        # Aggregate (remove keepdims for latent encoding)
        aggregated = self.aggregator(hidden)
        if len(ops.shape(aggregated)) == 3:
            aggregated = ops.squeeze(aggregated, axis=1)

        # Compute latent parameters
        mu = self.mu_layer(aggregated)
        log_sigma = self.log_sigma_layer(aggregated)

        # Bound the variance
        sigma = 0.1 + 0.9 * ops.sigmoid(log_sigma)

        return tfp.distributions.Normal(loc=mu, scale=sigma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "processor": self.processor,
                "num_latents": self.num_latents,
                "aggregator": self.aggregator,
            }
        )
        return config


# Keep the old encoders for backward compatibility, but mark them as deprecated
@keras.saving.register_keras_serializable()
class MeanEncoder(layers.Layer):
    def __init__(self, sizes=None, network=None, network_aggregates=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = kwargs.get("activation", "relu")
        self.network_aggregates = network_aggregates

        if network is not None:
            self.mlp = network
            self.sizes = None
        elif sizes is not None:
            self.sizes = list(sizes)
            mlp_layers = []
            for size in self.sizes[:-1]:
                mlp_layers.append(layers.Dense(size, activation=self.activation))
            if self.sizes:
                mlp_layers.append(layers.Dense(self.sizes[-1]))
            self.mlp = keras.Sequential(mlp_layers, name="mlp")
        else:
            raise ValueError("Either 'sizes' or a 'network' must be provided.")

    def call(self, context_x, context_y):
        # Concatenate x and y to create input features for each context point
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)

        # Pass through the MLP
        representation = self.mlp(encoder_input)

        if not self.network_aggregates:
            representation = ops.mean(representation, axis=1, keepdims=True)

        # Ensure output is 3D for broadcasting, e.g. (batch, 1, features)
        if len(ops.shape(representation)) == 2:
            representation = ops.expand_dims(representation, axis=1)

        return representation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sizes": self.sizes,
                "activation": self.activation,
                "network_aggregates": self.network_aggregates,
            }
        )
        if self.sizes is None:
            config["network"] = self.mlp
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class LatentEncoder(layers.Layer):
    def __init__(
        self,
        sizes=None,
        num_latents=None,
        network=None,
        network_aggregates=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = kwargs.get("activation", "relu")
        self.network_aggregates = network_aggregates

        if network is not None:
            self.mlp = network
            self.sizes = None
            if num_latents is None:
                raise ValueError(
                    "num_latents must be provided if a custom network is used."
                )
            self.num_latents = num_latents
        elif sizes is not None and num_latents is not None:
            self.sizes = list(sizes)
            self.num_latents = num_latents
            mlp_layers = []
            for size in self.sizes:
                mlp_layers.append(layers.Dense(size, activation=self.activation))
            self.mlp = keras.Sequential(mlp_layers, name="mlp")
        else:
            raise ValueError(
                "Either ('sizes' and 'num_latents') or ('network' and 'num_latents') must be provided."
            )

        # Two final layers for mean and log_sigma
        self.mu_layer = layers.Dense(self.num_latents)
        self.log_sigma_layer = layers.Dense(self.num_latents)

    def call(self, x, y):
        # Concatenate x and y along feature dimension
        encoder_input = ops.concatenate([x, y], axis=-1)

        # Pass through MLP
        hidden = self.mlp(encoder_input)

        # Aggregate by taking the mean over all points, if the network doesn't.
        if not self.network_aggregates:
            hidden = ops.mean(hidden, axis=1)

        # Get mean and log_sigma for the latent distribution
        mu = self.mu_layer(hidden)
        log_sigma = self.log_sigma_layer(hidden)

        # Bound the variance
        sigma = 0.1 + 0.9 * ops.sigmoid(log_sigma)

        # Return a distribution object
        return tfp.distributions.Normal(loc=mu, scale=sigma)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sizes": self.sizes,
                "num_latents": self.num_latents,
                "activation": self.activation,
                "network_aggregates": self.network_aggregates,
            }
        )
        if self.sizes is None:
            config["network"] = self.mlp
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class AttentiveEncoder(layers.Layer):
    def __init__(
        self,
        encoder_sizes=None,
        num_heads=None,
        context_network=None,
        query_network=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.activation = kwargs.get("activation", "relu")

        if context_network is not None:
            self.context_mlp = context_network
            self.encoder_sizes = None
            self.rep_dim = None  # Defer rep_dim initialization
        elif encoder_sizes is not None:
            self.encoder_sizes = list(encoder_sizes)
            self.rep_dim = self.encoder_sizes[-1]
            context_mlp_layers = []
            for size in self.encoder_sizes[:-1]:
                context_mlp_layers.append(
                    layers.Dense(size, activation=self.activation)
                )
            context_mlp_layers.append(layers.Dense(self.rep_dim))
            self.context_mlp = keras.Sequential(context_mlp_layers, name="context_mlp")
        else:
            raise ValueError(
                "Either 'encoder_sizes' or 'context_network' must be provided."
            )

        if query_network is not None:
            self.query_mlp = query_network
        else:
            self.query_mlp = (
                layers.Dense(self.rep_dim) if self.rep_dim is not None else None
            )

        if num_heads is None:
            raise ValueError("'num_heads' must be provided.")

        if self.rep_dim is not None:
            self.self_attention = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.rep_dim
            )
            self.cross_attention = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.rep_dim
            )
        else:
            self.self_attention = None
            self.cross_attention = None

        self.self_attention_norm = layers.LayerNormalization()
        self.cross_attention_norm = layers.LayerNormalization()

    def call(self, context_x, context_y, target_x):
        # Process context points individually
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)
        context_representation = self.context_mlp(encoder_input)

        if self.rep_dim is None:
            self.rep_dim = context_representation.shape[-1]
            if self.query_mlp is None:
                self.query_mlp = layers.Dense(self.rep_dim)
            self.self_attention = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.rep_dim
            )
            self.cross_attention = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.rep_dim
            )

        # Apply self-attention to the context representations
        self_att_output = self.self_attention(
            query=context_representation,
            value=context_representation,
            key=context_representation,
        )
        context_representation = self.self_attention_norm(
            context_representation + self_att_output
        )

        # Project target_x to create the query for cross-attention
        query = self.query_mlp(target_x)

        # Apply cross-attention
        cross_att_output = self.cross_attention(
            query=query, value=context_representation, key=context_representation
        )

        # The final representation is target-specific.
        # Shape: (batch, num_targets, features)
        final_representation = self.cross_attention_norm(query + cross_att_output)

        return final_representation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "num_heads": self.num_heads,
                "activation": self.activation,
            }
        )
        if self.encoder_sizes is None:
            config["context_network"] = self.context_mlp
            config["query_network"] = self.query_mlp
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, sizes=None, network=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = kwargs.get("activation", "relu")
        self.network_checked = False

        if network is not None:
            self.mlp = network
            self.sizes = None
        elif sizes is not None:
            self.sizes = list(sizes)
            mlp_layers = []
            for size in self.sizes[:-1]:
                mlp_layers.append(layers.Dense(size, activation=self.activation))
            mlp_layers.append(layers.Dense(self.sizes[-1]))
            self.mlp = keras.Sequential(mlp_layers, name="mlp")
            output_dim = self.sizes[-1]

            if output_dim % 2 != 0:
                warnings.warn(
                    "Warning: The last layer size of the decoder should be an even number, "
                    "as it's split into a mean and a standard deviation for each output "
                    f"dimension. Got size {output_dim}.",
                    UserWarning,
                )
        else:
            raise ValueError("Either 'sizes' or a 'network' must be provided.")

    def call(self, representation, target_x):
        # Concatenate representation and target_x
        decoder_input = ops.concatenate([representation, target_x], axis=-1)

        # Pass through MLP
        hidden = self.mlp(decoder_input)

        if not self.network_checked and self.sizes is None:
            output_dim = hidden.shape[-1]
            if output_dim % 2 != 0:
                warnings.warn(
                    "Warning: The custom network for the decoder should have an even number "
                    "of outputs, as it's split into a mean and a standard deviation. "
                    f"Got size {output_dim}.",
                    UserWarning,
                )
            self.network_checked = True

        # Split into mean and log_std
        mean, log_std = ops.split(hidden, 2, axis=-1)

        # Bound the variance
        std = 0.1 + 0.9 * ops.softplus(log_std)

        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update({"sizes": self.sizes, "activation": self.activation})
        if self.sizes is None:
            config["network"] = self.mlp
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================================================================
# 2. BASE AND MIXIN CLASSES (THE SKELETONS)
# =============================================================================


@keras.saving.register_keras_serializable()
class BaseNeuralProcess(keras.Model):
    """
    An abstract base class for all Neural Process models.

    This is now a standard Keras model that can be used with model.fit().
    Use the neural_process_generator from utils to create appropriate training data.

    Subclasses must implement:
    - `__init__(self, ...)`: To define their specific encoders and decoders.
    - `call(self, inputs, training=False)`: To define the forward pass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_x(self, X, name="X"):
        """
        Validate and prepare an X tensor.
        Handles optional 1D to 2D conversion.
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        if len(X.shape) != 3:
            raise ValueError(
                f"{name} must be a 3D tensor, but got {len(X.shape)} dims."
            )
        return X

    def _prepare_y(self, y, name="y"):
        """Validate and prepare a y tensor."""
        y = ops.convert_to_tensor(y, dtype="float32")
        if len(y.shape) != 3 or y.shape[-1] != self.y_dims:
            raise ValueError(
                f"{name} must have shape (num_samples, num_points, {self.y_dims}), "
                f"but got {y.shape}."
            )
        return y

    def _validate_data_shapes(self, X, y, name="train"):
        """Helper to validate the shape of the input data."""
        X = self._prepare_x(X, name=f"X_{name}")
        y = self._prepare_y(y, name=f"y_{name}")

        if not ops.all(ops.shape(X)[:2] == ops.shape(y)[:2]):
            raise ValueError(
                f"X_{name} and y_{name} must have the same number of samples and points, "
                f"but got {ops.shape(X)[:2]} and {ops.shape(y)[:2]}."
            )
        return X, y

    def predict(self, context_x, context_y, pred_x):
        """
        Make predictions with the Neural Process.

        Args:
            context_x: Context points, x-values.
            context_y: Context points, y-values.
            pred_x: Target points for prediction, x-values.

        Returns:
            A tuple of (mean, std) for the predictions.
        """
        context_x = self._prepare_x(context_x, name="context_x")
        context_y = self._prepare_y(context_y, name="context_y")
        pred_x = self._prepare_x(pred_x, name="pred_x")

        if not ops.all(ops.shape(context_x)[:2] == ops.shape(context_y)[:2]):
            raise ValueError(
                "context_x and context_y must have the same number of samples "
                f"and points, but got shapes {ops.shape(context_x)} "
                f"and {ops.shape(context_y)}."
            )

        # For prediction, we call the model directly and extract mean/std
        # The exact behavior depends on the model type (CNP vs NP/ANP)
        if hasattr(self, "_is_latent_model") and self._is_latent_model:
            # For latent models (NP/ANP), use a dummy target_y
            shape = ops.shape(pred_x)
            dummy_target_y = ops.zeros(
                (shape[0], shape[1], self.y_dims), dtype=pred_x.dtype
            )
            pred_dist, _, _ = self(
                (context_x, context_y, pred_x, dummy_target_y), training=False
            )
            return pred_dist.mean(), pred_dist.stddev()
        else:
            # For deterministic models (CNP)
            mean, std = self((context_x, context_y, pred_x), training=False)
            return mean, std


class ConditionalModelMixin:
    """
    Mixin for deterministic models like CNP.
    Provides the core training and testing logic.
    """

    def train_step(self, data):
        """Standard Keras train_step for use with model.fit()."""
        (context_x, context_y, target_x), target_y = data

        with tf.GradientTape() as tape:
            mean, std = self((context_x, context_y, target_x), training=True)
            # The loss is the negative log-likelihood of the targets
            # under the predicted distribution.
            dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            loss_value = -ops.mean(dist.log_prob(target_y))

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss_value}

    def test_step(self, data):
        """Standard Keras test_step for use with model.evaluate()."""
        (context_x, context_y, target_x), target_y = data

        mean, std = self((context_x, context_y, target_x), training=False)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        loss_value = -ops.mean(dist.log_prob(target_y))
        return {"loss": loss_value}


class LatentModelMixin:
    """
    A mixin class providing the shared ELBO-based logic
    for latent variable models like NP and ANP.
    """

    def train_step(self, data):
        """Standard Keras train_step for use with model.fit()."""
        (context_x, context_y, target_x), target_y = data

        with tf.GradientTape() as tape:
            # The `call` method of NP/ANP returns the predictive distribution,
            # and the prior/posterior distributions for the latent variable z.
            pred_dist, prior, posterior = self(
                (context_x, context_y, target_x, target_y), training=True
            )

            # 1. Reconstruction Loss (the log-likelihood term in ELBO)
            log_likelihood = pred_dist.log_prob(target_y)
            reconstruction_loss = -ops.mean(log_likelihood)

            # 2. KL Divergence (the regularization term in ELBO)
            # This encourages the context-based prior to be close to the
            # target-based posterior.
            kl_div = tfp.distributions.kl_divergence(posterior, prior)
            kl_div = ops.mean(ops.sum(kl_div, axis=-1))

            # 3. Final ELBO Loss to minimize
            # We want to maximize ELBO = log_likelihood - kl_div
            # So we minimize Loss = -ELBO = -log_likelihood + kl_div
            # The KL term is often weighted to balance the two losses.
            num_targets_float = ops.cast(ops.shape(target_x)[1], "float32")
            loss_value = reconstruction_loss + kl_div / num_targets_float

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss_value, "recon_loss": reconstruction_loss, "kl_div": kl_div}

    def test_step(self, data):
        """Standard Keras test_step for use with model.evaluate()."""
        (context_x, context_y, target_x), target_y = data

        pred_dist, prior, posterior = self(
            (context_x, context_y, target_x, target_y), training=False
        )

        # Compute reconstruction loss for evaluation
        log_likelihood = pred_dist.log_prob(target_y)
        reconstruction_loss = -ops.mean(log_likelihood)

        # At test time, posterior is None, so we can't compute true KL divergence
        # We'll just return the reconstruction loss as the main metric
        return {"loss": reconstruction_loss, "recon_loss": reconstruction_loss}


# =============================================================================
# 3. FLAVOURED MODEL IMPLEMENTATIONS
# =============================================================================


@keras.saving.register_keras_serializable()
class CNP(ConditionalModelMixin, BaseNeuralProcess):
    """
    Conditional Neural Process (CNP).
    A deterministic model that uses a mean-aggregated representation.
    """

    def __init__(
        self,
        encoder_sizes=[128, 128, 128, 128],
        decoder_sizes=[128, 128],
        y_dims=1,
        encoder=None,
        decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_sizes = list(encoder_sizes)
        self.decoder_sizes = list(decoder_sizes)
        self.y_dims = y_dims
        self._is_latent_model = False  # CNP is deterministic
        full_decoder_sizes = self.decoder_sizes + [
            2 * self.y_dims
        ]  # adding 2 per dim: mu and sigma

        # Use new compositional architecture internally
        if encoder is not None:
            self.encoder = encoder
        else:
            # Create MLP processor
            mlp_layers = []
            for size in self.encoder_sizes[:-1]:
                mlp_layers.append(layers.Dense(size, activation="relu"))
            mlp_layers.append(layers.Dense(self.encoder_sizes[-1]))
            processor = keras.Sequential(mlp_layers, name="encoder_mlp")

            # Use new compositional encoder
            self.encoder = DeterministicEncoder(processor, MeanAggregator())

        self.decoder = decoder if decoder is not None else Decoder(full_decoder_sizes)

    def call(self, inputs, training=False):
        context_x, context_y, target_x = inputs

        # 1. Encode context to a single, global representation vector
        # Check if using new compositional encoder or old encoder for backward compatibility
        if isinstance(self.encoder, DeterministicEncoder):
            representation = self.encoder([context_x, context_y])
        else:
            # Old encoder API (backward compatibility)
            representation = self.encoder(context_x, context_y)

        # 2. Repeat the representation for each target point to match dimensions
        num_targets = ops.shape(target_x)[1]
        representation = ops.broadcast_to(
            representation,
            (ops.shape(representation)[0], num_targets, ops.shape(representation)[-1]),
        )  # Shape: (batch, num_targets, dim)

        # 3. Decode to get predicted distribution
        mean, std = self.decoder(representation, target_x)
        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "decoder_sizes": self.decoder_sizes,
                "y_dims": self.y_dims,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class NP(LatentModelMixin, BaseNeuralProcess):
    """
    Neural Process (NP).
    A probabilistic model with a deterministic path (using mean-aggregation)
    and a latent path to model global uncertainty.
    """

    def __init__(
        self,
        det_encoder_sizes=[128, 128, 128, 128],
        latent_encoder_sizes=[128, 128],
        num_latents=128,
        decoder_sizes=[128, 128],
        y_dims=1,
        det_encoder=None,
        latent_encoder=None,
        decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.det_encoder_sizes = list(det_encoder_sizes)
        self.latent_encoder_sizes = list(latent_encoder_sizes)
        self.num_latents = num_latents
        self.decoder_sizes = list(decoder_sizes)
        self.y_dims = y_dims
        self._is_latent_model = True  # NP has latent variables
        full_decoder_sizes = self.decoder_sizes + [
            2 * self.y_dims
        ]  # adding 2 per dim: mu and sigma

        # Use new compositional architecture internally
        if det_encoder is not None:
            self.det_encoder = det_encoder
        else:
            # Create MLP processor for deterministic encoder
            det_mlp_layers = []
            for size in self.det_encoder_sizes[:-1]:
                det_mlp_layers.append(layers.Dense(size, activation="relu"))
            det_mlp_layers.append(layers.Dense(self.det_encoder_sizes[-1]))
            det_processor = keras.Sequential(det_mlp_layers, name="det_encoder_mlp")

            # Use new compositional deterministic encoder
            self.det_encoder = DeterministicEncoder(det_processor, MeanAggregator())

        if latent_encoder is not None:
            self.latent_encoder = latent_encoder
        else:
            # Create MLP processor for latent encoder
            lat_mlp_layers = []
            for size in self.latent_encoder_sizes:
                lat_mlp_layers.append(layers.Dense(size, activation="relu"))
            lat_processor = keras.Sequential(lat_mlp_layers, name="latent_encoder_mlp")

            # Use new compositional latent encoder
            self.latent_encoder = CompositionalLatentEncoder(
                lat_processor, self.num_latents
            )

        self.decoder = decoder if decoder is not None else Decoder(full_decoder_sizes)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # Latent Path: determine prior and posterior distributions for z
        # Check if using new compositional encoder or old encoder for backward compatibility
        if isinstance(self.latent_encoder, CompositionalLatentEncoder):
            prior_dist = self.latent_encoder([context_x, context_y])
            if training:
                posterior_dist = self.latent_encoder([target_x, target_y])
                z = posterior_dist.sample()
            else:
                posterior_dist = None
                z = prior_dist.sample()
        else:
            # Old encoder API (backward compatibility)
            prior_dist = self.latent_encoder(context_x, context_y)
            if training:
                posterior_dist = self.latent_encoder(target_x, target_y)
                z = posterior_dist.sample()
            else:
                posterior_dist = None
                z = prior_dist.sample()

        num_targets = ops.shape(target_x)[1]
        z_rep = ops.broadcast_to(
            ops.expand_dims(z, axis=1),
            (ops.shape(z)[0], num_targets, ops.shape(z)[-1]),
        )

        # Deterministic Path (using mean encoder)
        # Check if using new compositional encoder or old encoder for backward compatibility
        if isinstance(self.det_encoder, DeterministicEncoder):
            det_rep = self.det_encoder([context_x, context_y])
        else:
            # Old encoder API (backward compatibility)
            det_rep = self.det_encoder(context_x, context_y)
        det_rep = ops.broadcast_to(
            det_rep,
            (ops.shape(det_rep)[0], num_targets, ops.shape(det_rep)[-1]),
        )  # Shape: (batch, num_targets, dim)

        # Combine and Decode
        representation = ops.concatenate([det_rep, z_rep], axis=-1)
        mean, std = self.decoder(representation, target_x)
        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "det_encoder_sizes": self.det_encoder_sizes,
                "latent_encoder_sizes": self.latent_encoder_sizes,
                "num_latents": self.num_latents,
                "decoder_sizes": self.decoder_sizes,
                "y_dims": self.y_dims,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class ANP(LatentModelMixin, BaseNeuralProcess):
    """
    Attentive Neural Process (ANP).
    A probabilistic model that uses an attention mechanism in its deterministic
    path to avoid underfitting, combined with a latent path for global uncertainty.
    """

    def __init__(
        self,
        att_encoder_sizes=[128, 128, 128, 128],
        num_heads=8,
        latent_encoder_sizes=[128, 128],
        num_latents=128,
        decoder_sizes=[128, 128],
        y_dims=1,
        att_encoder=None,
        latent_encoder=None,
        decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.att_encoder_sizes = list(att_encoder_sizes)
        self.num_heads = num_heads
        self.latent_encoder_sizes = list(latent_encoder_sizes)
        self.num_latents = num_latents
        self.decoder_sizes = list(decoder_sizes)
        self.y_dims = y_dims
        self._is_latent_model = True  # ANP has latent variables

        full_decoder_sizes = self.decoder_sizes + [
            2 * self.y_dims
        ]  # adding 2 per dim: mu and sigma

        # Use new compositional architecture internally
        if att_encoder is not None:
            self.att_encoder = att_encoder
        else:
            # Create MLP processor for attention encoder
            att_mlp_layers = []
            for size in self.att_encoder_sizes[:-1]:
                att_mlp_layers.append(layers.Dense(size, activation="relu"))
            att_mlp_layers.append(layers.Dense(self.att_encoder_sizes[-1]))
            att_processor = keras.Sequential(att_mlp_layers, name="att_encoder_mlp")

            # Use new compositional deterministic encoder with attention
            self.att_encoder = DeterministicEncoder(
                att_processor,
                AttentionAggregator(num_heads=self.num_heads),
                target_x_required=True,
            )

        if latent_encoder is not None:
            self.latent_encoder = latent_encoder
        else:
            # Create MLP processor for latent encoder
            lat_mlp_layers = []
            for size in self.latent_encoder_sizes:
                lat_mlp_layers.append(layers.Dense(size, activation="relu"))
            lat_processor = keras.Sequential(lat_mlp_layers, name="latent_encoder_mlp")

            # Use new compositional latent encoder
            self.latent_encoder = CompositionalLatentEncoder(
                lat_processor, self.num_latents
            )

        self.decoder = decoder if decoder is not None else Decoder(full_decoder_sizes)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # Latent Path (same as NP, provides global context)
        # Check if using new compositional encoder or old encoder for backward compatibility
        if isinstance(self.latent_encoder, CompositionalLatentEncoder):
            prior_dist = self.latent_encoder([context_x, context_y])
            if training:
                posterior_dist = self.latent_encoder([target_x, target_y])
                z = posterior_dist.sample()
            else:
                posterior_dist = None
                z = prior_dist.sample()
        else:
            # Old encoder API (backward compatibility)
            prior_dist = self.latent_encoder(context_x, context_y)
            if training:
                posterior_dist = self.latent_encoder(target_x, target_y)
                z = posterior_dist.sample()
            else:
                posterior_dist = None
                z = prior_dist.sample()

        num_targets = ops.shape(target_x)[1]
        z_rep = ops.broadcast_to(
            ops.expand_dims(z, axis=1),
            (ops.shape(z)[0], num_targets, ops.shape(z)[-1]),
        )

        # Deterministic Path (uses attention to get target-specific context)
        # Note: The output `det_rep` already has shape (batch, num_targets, features),
        # so it does NOT need to be broadcasted like in CNP/NP.
        # Check if using new compositional encoder or old encoder for backward compatibility
        if isinstance(self.att_encoder, DeterministicEncoder):
            det_rep = self.att_encoder([context_x, context_y, target_x])
        else:
            # Old encoder API (backward compatibility)
            det_rep = self.att_encoder(context_x, context_y, target_x)

        # Combine and Decode
        representation = ops.concatenate([det_rep, z_rep], axis=-1)
        mean, std = self.decoder(representation, target_x)
        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "att_encoder_sizes": self.att_encoder_sizes,
                "num_heads": self.num_heads,
                "latent_encoder_sizes": self.latent_encoder_sizes,
                "num_latents": self.num_latents,
                "decoder_sizes": self.decoder_sizes,
                "y_dims": self.y_dims,
            }
        )
        return config
