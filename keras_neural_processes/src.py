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
@keras.saving.register_keras_serializable()
class MeanEncoder(layers.Layer):
    def __init__(self, hidden_sizes, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = kwargs.get("activation", "relu")
        self.mlp_layers = []

        for size in self.hidden_sizes[:-1]:
            self.mlp_layers.append(layers.Dense(size, activation=self.activation))
        # Final layer without activation
        self.mlp_layers.append(layers.Dense(hidden_sizes[-1]))

    def call(self, context_x, context_y):
        # Concatenate x and y to create input features for each context point
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)

        # Pass through the MLP
        hidden = encoder_input
        for layer in self.mlp_layers:
            hidden = layer(hidden)

        # Aggregate representations by taking the mean across the context points
        # Shape: (batch, num_context, features) -> (batch, 1, features)
        representation = ops.mean(hidden, axis=1, keepdims=True)
        return representation

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_sizes": self.hidden_sizes})
        config.update({"activation": self.activation})
        return config


@keras.saving.register_keras_serializable()
class LatentEncoder(layers.Layer):
    def __init__(self, hidden_sizes, num_latents, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.num_latents = num_latents
        self.activation = kwargs.get("activation", "relu")
        self.mlp_layers = []

        for size in hidden_sizes:
            self.mlp_layers.append(layers.Dense(size, activation=self.activation))
        # Two final layers for mean and log_sigma
        self.mu_layer = layers.Dense(self.num_latents)
        self.log_sigma_layer = layers.Dense(self.num_latents)

    def call(self, x, y):
        # Concatenate x and y along feature dimension
        encoder_input = ops.concatenate([x, y], axis=-1)

        # Pass through MLP
        hidden = encoder_input
        for layer in self.mlp_layers:
            hidden = layer(hidden)

        # Aggregate by taking the mean over all points
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
                "hidden_sizes": self.hidden_sizes,
                "num_latents": self.num_latents,
                "activation": self.activation,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class AttentiveEncoder(layers.Layer):
    def __init__(self, encoder_sizes, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.encoder_sizes = list(encoder_sizes)
        self.num_heads = num_heads
        self.rep_dim = self.encoder_sizes[-1]  # The unified representation dimension
        self.activation = kwargs.get("activation", "relu")

        # 1. MLP to process each (context_x, context_y) pair into a representation
        self.context_mlp_hidden_layers = []
        for size in self.encoder_sizes[:-1]:
            self.context_mlp_hidden_layers.append(
                layers.Dense(size, activation=self.activation)
            )
        # Final layer without activation
        self.context_mlp_final_layer = layers.Dense(self.rep_dim)

        # 2. Self-Attention over context points to model interactions
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.rep_dim
        )
        self.self_attention_norm = layers.LayerNormalization()

        # 3. MLP to project target_x into a query vector for cross-attention
        self.query_mlp = layers.Dense(self.rep_dim)

        # 4. Cross-Attention between target_x (query) and context (key/value)
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.rep_dim
        )
        self.cross_attention_norm = layers.LayerNormalization()

    def call(self, context_x, context_y, target_x):
        # Process context points individually
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)
        hidden = encoder_input
        for layer in self.context_mlp_hidden_layers:
            hidden = layer(hidden)
        context_representation = self.context_mlp_final_layer(hidden)

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
        return config


@keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, hidden_sizes, output_dims, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = list(hidden_sizes)
        self.output_dims = output_dims
        self.activation = kwargs.get("activation", "relu")
        self.mlp_layers = []

        for size in hidden_sizes[:-1]:
            self.mlp_layers.append(layers.Dense(size, activation=self.activation))

        # Final layer outputs mean and log_std
        if hidden_sizes[-1] != 2:
            # assert hidden_sizes % 2 == 0
            warnings.warn(
                "Warning: Ideally the last layer size should be 2. "
                "One neuron for mean, one neuron for std. "
                "Please keep this in mind and proceed only if ",
                "you know what you are doing.",
                UserWarning,
            )

        self.mlp_layers.append(layers.Dense(hidden_sizes[-1]))

    def call(self, representation, target_x):
        # Concatenate representation and target_x
        decoder_input = ops.concatenate([representation, target_x], axis=-1)

        # Pass through MLP
        hidden = decoder_input
        for layer in self.mlp_layers:
            hidden = layer(hidden)

        # Split into mean and log_std
        mean, log_std = ops.split(hidden, 2, axis=-1)

        # Bound the variance
        std = 0.1 + 0.9 * ops.softplus(log_std)

        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_sizes": self.hidden_sizes, "activation": self.activation}
        )
        return config


# =============================================================================
# 2. BASE AND MIXIN CLASSES (THE SKELETONS)
# =============================================================================


@keras.saving.register_keras_serializable()
class BaseNeuralProcess(keras.Model):
    """
    An abstract base class for all Neural Process models.
    It handles the high-level training loop, progress bars, and validation callbacks.

    Subclasses must implement:
    - `__init__(self, ...)`: To define their specific encoders and decoders.
    - `call(self, inputs, training=False)`: To define the forward pass.
    - `train_step(self, ...)`: To define the logic for a single gradient update.
    - `test_step(self, ...)`: To define the inference logic.
    """

    def train(
        self,
        X_train,
        y_train,
        epochs,
        optimizer,
        batch_size=64,
        num_context_range=[2, 10],
        num_context_mode="each",
        dataset_type="gp",
        X_val=None,
        y_val=None,
        plotcb=True,
        pbar=True,
        plot_every=1000,
        pred_points=None,
        seed_val=None,
    ):
        self.optimizer = optimizer

        if pbar:
            metric_names = ["loss"]
            if isinstance(self, LatentModelMixin):
                metric_names.extend(["recon_loss", "kl_div"])
            progbar = Progbar(epochs, stateful_metrics=metric_names)

        if plotcb:
            assert (
                X_val is not None and y_val is not None
            ), "Validation data must be provided if plotcb is True."
            if pred_points is None and dataset_type == "gp":
                pred_points = X_val.shape[1]

        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)
        callbacks.on_train_begin()

        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_begin(epoch)

            # --- Data Sampling for this Epoch ---
            X_train_batch, y_train_batch = utils.get_train_batch(
                X_train, y_train, batch_size
            )
            num_context = utils._sample_num_context(num_context_range)
            context_x_train, context_y_train = utils.get_context_set(
                X_train_batch, y_train_batch, num_context, num_context_mode
            )
            target_x_train, target_y_train = X_train_batch, y_train_batch

            # --- Perform a single training step ---
            # This will call the specific train_step of CNP, NP, or ANP
            logs = self.train_step(
                context_x_train, context_y_train, target_x_train, target_y_train
            )

            callbacks.on_epoch_end(epoch, logs)
            if pbar:
                progbar.update(epoch, values=[(k, float(v)) for k, v in logs.items()])

            # --- Validation and Plotting Callback ---
            if plotcb and (epoch % plot_every == 0):
                if dataset_type == "gp":
                    utils.gplike_val_step(
                        self,
                        X_val,
                        y_val,
                        pred_points,
                        num_context_range,
                        num_context_mode,
                        epoch,
                        seed_val,
                    )
                elif dataset_type == "mnist":
                    utils.mnist_val_step(
                        self, X_val, y_val, num_context_range, epoch, seed_val
                    )
        return history


class ConditionalModelMixin:
    """
    Docstring
    """

    # note: signature assumes long format
    # i.e., for x: (n_samples, n_points_total, 2) where 2 for [x_value, channel_id]
    # and for y: (n_samples, n_points_total, 1) where 1 for [y_value]
    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # context_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # context_y
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # target_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # target_y
        ],
    )
    def train_step(self, context_x, context_y, target_x, target_y):
        with tf.GradientTape() as tape:
            mean, std = self((context_x, context_y, target_x), training=True)
            # The loss is the negative log-likelihood of the targets
            # under the predicted distribution.
            dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            loss_value = -ops.mean(dist.log_prob(target_y))
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)
        return {"loss": loss_value}

    # note: signature assumes long format
    # i.e., for x: (n_samples, n_points_total, 2) where 2 for [x_value, channel_id]
    # and for y: (n_samples, n_points_total, 1) where 1 for [y_value]
    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # context_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # context_y
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # pred_x
        ],
    )
    def test_step(self, context_x, context_y, pred_x):
        mean, std = self((context_x, context_y, pred_x), training=False)
        return mean, std


class LatentModelMixin:
    """
    A mixin class providing the shared ELBO-based train_step and test_step
    for latent variable models like NP and ANP. A mixin class provides
    methods to other classes but is not meant to be instantiated on its own.
    """

    # note: signature assumes long format
    # i.e., for x: (n_samples, n_points_total, 2) where 2 for [x_value, channel_id]
    # and for y: (n_samples, n_points_total, 1) where 1 for [y_value]
    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # context_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # context_y
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # target_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # target_y
        ],
    )
    def train_step(self, context_x, context_y, target_x, target_y):
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
        self.optimizer.apply(grads, self.trainable_weights)

        return {"loss": loss_value, "recon_loss": reconstruction_loss, "kl_div": kl_div}

    # note: signature assumes long format
    # i.e., for x: (n_samples, n_points_total, 2) where 2 for [x_value, channel_id]
    # and for y: (n_samples, n_points_total, 1) where 1 for [y_value]
    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # context_x
            tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),  # context_y
            tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),  # pred_x
        ],
    )
    def test_step(self, context_x, context_y, pred_x):
        # At test time, we don't have target_y. The `call` method will
        # sample the latent z from the prior distribution p(z|context).
        pred_dist, _, _ = self((context_x, context_y, pred_x, None), training=False)
        return pred_dist.mean(), pred_dist.stddev()


# =============================================================================
# 3. FLAVOURED MODEL IMPLEMENTATIONS
# =============================================================================


@keras.saving.register_keras_serializable()
class CNP(BaseNeuralProcess):
    """
    Conditional Neural Process (CNP).
    A deterministic model that uses a mean-aggregated representation.
    """

    def __init__(self, encoder_sizes, decoder_sizes, output_dims, **kwargs):
        super().__init__(**kwargs)
        self.encoder_sizes = list(encoder_sizes)
        self.decoder_sizes = list(decoder_sizes)
        self.output_dims = output_dims

        self.encoder = MeanEncoder(self.encoder_sizes)
        self.decoder = Decoder(self.decoder_sizes, self.output_dims)

    def call(self, inputs, training=False):
        context_x, context_y, target_x = inputs

        # 1. Encode context to a single, global representation vector
        representation = self.encoder(context_x, context_y)

        # 2. Repeat the representation for each target point to match dimensions
        num_targets = ops.shape(target_x)[1]
        representation = ops.repeat(representation, num_targets, axis=1)

        # 3. Decode to get predictive distribution
        mean, std = self.decoder(representation, target_x)
        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "decoder_sizes": self.decoder_sizes,
                "output_dims": self.output_dims,
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
        det_encoder_sizes,
        latent_encoder_sizes,
        num_latents,
        decoder_sizes,
        output_dims,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.det_encoder_sizes = list(det_encoder_sizes)
        self.latent_encoder_sizes = list(latent_encoder_sizes)
        self.num_latents = num_latents
        self.decoder_sizes = list(decoder_sizes)
        self.output_dims = output_dims

        self.det_encoder = MeanEncoder(self.det_encoder_sizes)
        self.latent_encoder = LatentEncoder(self.latent_encoder_sizes, self.num_latents)
        self.decoder = Decoder(self.decoder_sizes, self.output_dims)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # Latent Path: determine prior and posterior distributions for z
        prior_dist = self.latent_encoder(context_x, context_y)
        if training:
            # During training, we use the full target set to form a richer posterior
            posterior_dist = self.latent_encoder(target_x, target_y)
            z = posterior_dist.sample()
        else:  # Inference time
            # At test time, we only have the context, so we sample from the prior
            posterior_dist = None
            z = prior_dist.sample()

        num_targets = ops.shape(target_x)[1]
        z_rep = ops.repeat(ops.expand_dims(z, axis=1), num_targets, axis=1)

        # Deterministic Path (using mean encoder)
        det_rep = self.det_encoder(context_x, context_y)
        det_rep = ops.repeat(det_rep, num_targets, axis=1)

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
                "output_dims": self.output_dims,
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
        att_encoder_sizes,
        num_heads,
        latent_encoder_sizes,
        num_latents,
        decoder_sizes,
        output_dims,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.att_encoder_sizes = list(att_encoder_sizes)
        self.num_heads = num_heads
        self.latent_encoder_sizes = list(latent_encoder_sizes)
        self.num_latents = num_latents
        self.decoder_sizes = list(decoder_sizes)
        self.output_dims = output_dims

        self.att_encoder = AttentiveEncoder(self.att_encoder_sizes, self.num_heads)
        self.latent_encoder = LatentEncoder(self.latent_encoder_sizes, self.num_latents)
        self.decoder = Decoder(self.decoder_sizes, self.output_dims)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # Latent Path (same as NP, provides global context)
        prior_dist = self.latent_encoder(context_x, context_y)
        if training:
            posterior_dist = self.latent_encoder(target_x, target_y)
            z = posterior_dist.sample()
        else:  # Inference time
            posterior_dist = None
            z = prior_dist.sample()

        num_targets = ops.shape(target_x)[1]
        z_rep = ops.repeat(ops.expand_dims(z, axis=1), num_targets, axis=1)

        # Deterministic Path (uses attention to get target-specific context)
        # Note: The output `det_rep` already has shape (batch, num_targets, features),
        # so it does NOT need to be repeated like in CNP/NP.
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
                "output_dims": self.output_dims,
            }
        )
        return config
