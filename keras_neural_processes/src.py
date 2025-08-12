import warnings

import keras
from keras import layers, ops
from keras.utils import Progbar

import tensorflow as tf

import tensorflow_probability as tfp

from . import utils


# Encoder network that processes context points
@keras.saving.register_keras_serializable()
class DeterministicEncoder(layers.Layer):
    def __init__(self, hidden_sizes=[128, 128, 128, 128], **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = []
        self.activation = kwargs.get("activation", "relu")

        for size in hidden_sizes[:-1]:
            self.hidden_layers.append(layers.Dense(size, activation=self.activation))
        # Final layer without activation
        self.hidden_layers.append(layers.Dense(hidden_sizes[-1]))

    def build(self, input_shape):
        context_x_shape, context_y_shape = input_shape

        # The input to the first Dense layer is the concatenation of x and y features
        current_shape_dim = context_x_shape[-1] + context_y_shape[-1]
        # Build each layer sequentially
        for layer in self.hidden_layers:
            layer.build(input_shape=(None, None, current_shape_dim))
            current_shape_dim = (
                layer.units
            )  # The output dim of this layer is the input dim for the next

        # Mark this layer as built
        super().build(input_shape)

    def call(self, context_x, context_y):
        # Concatenate x and y along feature dimension
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)

        # Pass through MLP
        hidden = encoder_input
        for layer in self.hidden_layers:
            hidden = layer(hidden)

        # Average over context points
        representation = ops.mean(hidden, axis=1, keepdims=True)
        return representation

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_sizes": self.hidden_sizes})
        return config


# Latent encoder network that processes context points for the latent path
@keras.saving.register_keras_serializable()
class LatentEncoder(layers.Layer):
    def __init__(self, hidden_sizes=[128, 128], num_latents=128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.num_latents = num_latents
        self.activation = kwargs.get("activation", "relu")

        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(layers.Dense(size, activation=self.activation))

        # Two final layers for mean and log_sigma
        self.mu_layer = layers.Dense(self.num_latents)
        self.log_sigma_layer = layers.Dense(self.num_latents)

    def build(self, input_shape):
        context_x_shape, context_y_shape = input_shape
        current_shape_dim = context_x_shape[-1] + context_y_shape[-1]

        for layer in self.hidden_layers:
            layer.build(input_shape=(None, None, current_shape_dim))
            current_shape_dim = layer.units

        # The penultimate layer's output is the input to mu and log_sigma layers
        penultimate_shape = (None, None, current_shape_dim)
        self.mu_layer.build(penultimate_shape)
        self.log_sigma_layer.build(penultimate_shape)

        super().build(input_shape)

    def call(self, x, y):
        # Concatenate x and y along feature dimension
        encoder_input = ops.concatenate([x, y], axis=-1)

        # Pass through MLP
        hidden = encoder_input
        for layer in self.hidden_layers:
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
            }
        )
        return config


# Attentive encoder network that uses self-attention and cross-attention.
@keras.saving.register_keras_serializable()
class AttentiveEncoder(layers.Layer):
    def __init__(self, encoder_sizes=[128, 128, 128, 128], num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.encoder_sizes = encoder_sizes
        self.num_heads = num_heads
        self.rep_dim = self.encoder_sizes[-1]  # The unified representation dimension
        self.activation = kwargs.get("activation", "relu")

        # --- Sub-layers ---
        # 1. MLP to process each context point (x, y) individually
        self.context_mlp_hidden_layers = []
        for size in self.encoder_sizes[:-1]:
            self.context_mlp_hidden_layers.append(
                layers.Dense(size, activation=self.activation)
            )
        self.context_mlp_final_layer = layers.Dense(self.rep_dim)

        # 2. Self-Attention over context points
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.rep_dim
        )
        self.self_attention_norm = layers.LayerNormalization()

        # 3. MLP to project target_x to be the cross-attention query
        self.query_mlp = layers.Dense(self.rep_dim)

        # 4. Cross-Attention between target_x and context
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.rep_dim
        )
        self.cross_attention_norm = layers.LayerNormalization()

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape = input_shape

        current_shape_dim = context_x_shape[-1] + context_y_shape[-1]
        for layer in self.context_mlp_hidden_layers:
            layer.build(input_shape=(None, None, current_shape_dim))
            current_shape_dim = layer.units
        self.context_mlp_final_layer.build(input_shape=(None, None, current_shape_dim))

        attention_input_shape = (None, None, self.rep_dim)
        self.self_attention.build(
            query_shape=attention_input_shape, value_shape=attention_input_shape
        )
        self.self_attention_norm.build(attention_input_shape)

        self.query_mlp.build(target_x_shape)

        query_shape = (None, None, self.rep_dim)
        self.cross_attention.build(
            query_shape=query_shape, value_shape=attention_input_shape
        )
        self.cross_attention_norm.build(query_shape)

        super().build(input_shape)

    def call(self, context_x, context_y, target_x):
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)
        hidden = encoder_input
        for layer in self.context_mlp_hidden_layers:
            hidden = layer(hidden)
        context_representation = self.context_mlp_final_layer(hidden)

        self_att_output = self.self_attention(
            query=context_representation,
            value=context_representation,
            key=context_representation,
        )
        context_representation = self.self_attention_norm(
            context_representation + self_att_output
        )

        query = self.query_mlp(target_x)

        cross_att_output = self.cross_attention(
            query=query, value=context_representation, key=context_representation
        )
        final_representation = self.cross_attention_norm(query + cross_att_output)

        return final_representation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "num_heads": self.num_heads,
            }
        )
        return config


# Decoder network that makes predictions at target points
@keras.saving.register_keras_serializable()
class DeterministicDecoder(layers.Layer):
    def __init__(self, hidden_sizes=[128, 128, 2], **kwargs):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = []
        self.activation = kwargs.get("activation", "relu")

        for size in hidden_sizes[:-1]:
            self.hidden_layers.append(layers.Dense(size, activation=self.activation))
        # Final layer outputs mean and log_std
        self.hidden_layers.append(layers.Dense(hidden_sizes[-1]))

    def build(self, input_shape):
        representation_shape, target_x_shape = input_shape

        # The input to the first Dense layer is
        # the concatenation of representation and target_x
        current_shape_dim = representation_shape[-1] + target_x_shape[-1]
        # Build each layer sequentially
        for layer in self.hidden_layers:
            layer.build(input_shape=(None, None, current_shape_dim))
            current_shape_dim = (
                layer.units
            )  # The output dim of this layer is the input dim for the next

        # Mark this layer as built
        super().build(input_shape)

    def call(self, representation, target_x):
        # Concatenate representation and target_x
        decoder_input = ops.concatenate([representation, target_x], axis=-1)

        # Pass through MLP
        hidden = decoder_input
        for layer in self.hidden_layers:
            hidden = layer(hidden)

        # Split into mean and log_std
        mean, log_std = ops.split(hidden, 2, axis=-1)

        # Bound the variance
        std = 0.1 + 0.9 * ops.softplus(log_std)

        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_sizes": self.hidden_sizes})
        return config


@keras.saving.register_keras_serializable()
class CNP(keras.Model):
    def __init__(
        self,
        encoder_sizes=[128, 128, 128, 128],
        decoder_sizes=[128, 128],
        output_dims=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set encoder and decoder sizes
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes_hidden = list(decoder_sizes)  # Create a mutable
        self.output_dims = output_dims

        full_decoder_sizes = self.decoder_sizes_hidden + [2 * self.output_dims]
        self.encoder = DeterministicEncoder(self.encoder_sizes)
        self.decoder = DeterministicDecoder(full_decoder_sizes)

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape = input_shape

        # 1. Build the encoder with its specific input shapes.
        #    The encoder's call method expects a tuple of (context_x, context_y).
        self.encoder.build((context_x_shape, context_y_shape))

        # 2. Determine the shape of the representation vector produced by the encoder
        #    to build the decoder. The shape is (batch, 1, last_encoder_size).
        representation_shape = (context_x_shape[0], 1, self.encoder_sizes[-1])

        # 3. Build the decoder with its specific input shapes.
        #    The decoder's call method expects (representation, target_x).
        self.decoder.build((representation_shape, target_x_shape))

        # 4. Run original validation logic from your build method
        xdims = context_x_shape[-1]  # No. of channels / dims in x
        ydims = context_y_shape[-1]  # No. of channels / dims in y
        if xdims > 1:
            warnings.warn(
                "Note: The encoder implementation doesn't implicitly "
                "depend on xdims. Ensure you use an encoder size that is "
                "appropriate for your data.",
                UserWarning,
            )
        assert ydims == self.output_dims

        # 5. Call the parent's build method at the end.
        super().build(input_shape)

    def call(self, inputs, training=False):
        context_x, context_y, target_x = inputs
        representation = self.encoder(
            context_x, context_y
        )  # Get representation from encoder

        # Expand representation to match the number of target points
        num_targets = ops.shape(target_x)[1]

        representation = ops.broadcast_to(
            representation,
            (ops.shape(representation)[0], num_targets, ops.shape(representation)[-1]),
        )  # Shape: (batch, num_targets, dim)

        mean, std = self.decoder(
            representation, target_x
        )  # Get predictions from decoder
        return mean, std

    @tf.function(jit_compile=True)
    def train_step(self, context_x, context_y, target_x, target_y):
        with tf.GradientTape() as tape:
            mean, std = self(
                (context_x, context_y, target_x), training=True
            )  # Forward pass
            dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            loss_value = -ops.mean(
                dist.log_prob(target_y)
            )  # negative log likelihood loss
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
        num_context_range=[2, 10],
        num_context_mode="each",
        dataset_type="gp",
        X_val=None,
        y_val=None,
        plotcb=True,
        pbar=True,
        pred_points=None,
        plot_every=1000,
        seed_val=None,
    ):
        self.optimizer = optimizer

        if pbar:
            progbar = Progbar(epochs, stateful_metrics=["loss"])

        #
        #        if X_val is None:
        #            assert pred_points is not None
        #        elif pred_points is None:
        #            pred_points = X_val.shape[1]
        #
        #        if plotcb:
        #            assert X_val is not None and y_val is not None
        #
        if plotcb:
            assert (
                X_val is not None and y_val is not None
            ), "Validation data must be provided if plotcb is True."
            if pred_points is None and dataset_type == "gp":
                pred_points = X_val.shape[1]

        # Initialize callbacks
        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)

        callbacks.on_train_begin()  # Called at the start of training

        # no batching yet - possibly implement later
        for epoch in range(1, epochs + 1):
            # Reset metrics at start of each epoch
            callbacks.on_epoch_begin(epoch)  # Called at start of epoch

            X_train_batch, y_train_batch = utils.get_train_batch(
                X_train, y_train, batch_size=batch_size
            )

            num_context = utils._sample_num_context(num_context_range)

            # sample context set from fixed target set
            context_x_train, context_y_train = utils.get_context_set(
                X_train_batch,
                y_train_batch,
                num_context=num_context,
                num_context_mode=num_context_mode,
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
                if dataset_type == "gp":
                    utils.gplike_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        pred_points=pred_points,
                        num_context_range=num_context_range,
                        num_context_mode=num_context_mode,
                        epoch=epoch,
                        seed=seed_val,
                    )
                elif dataset_type == "mnist":
                    utils.mnist_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        num_context_range=num_context_range,
                        epoch=epoch,
                        seed=seed_val,
                    )

        return history

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "decoder_sizes": self.decoder_sizes_hidden,
                "output_dims": self.output_dims,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class NP(keras.Model):
    def __init__(
        self,
        det_encoder_sizes=[128] * 4,
        latent_encoder_sizes=[128] * 2,
        num_latents=128,
        decoder_sizes=[128] * 2,
        output_dims=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_dims = output_dims
        self.num_latents = num_latents

        # Store architecture sizes for saving/loading
        self.det_encoder_sizes = det_encoder_sizes
        self.latent_encoder_sizes = latent_encoder_sizes
        self.decoder_sizes_hidden = list(decoder_sizes)

        # Instantiate the components
        self.det_encoder = DeterministicEncoder(self.det_encoder_sizes)
        self.latent_encoder = LatentEncoder(self.latent_encoder_sizes, self.num_latents)

        full_decoder_sizes = self.decoder_sizes_hidden + [2 * self.output_dims]
        # The decoder now needs to handle the concatenated
        # deterministic and latent representations
        self.decoder = DeterministicDecoder(full_decoder_sizes)

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape, target_y_shape = input_shape
        # 1. Build the deterministic and latent encoders with its specific input shapes.
        self.det_encoder.build((context_x_shape, context_y_shape))
        self.latent_encoder.build((context_x_shape, context_y_shape))

        # 2. Determine the shape of the representation vector produced by the encoder
        # Note that decoder input is concatenation of det_rep and latent_rep
        det_rep_dim = self.det_encoder_sizes[-1]
        latent_rep_dim = self.num_latents
        representation_shape = (None, 1, det_rep_dim + latent_rep_dim)

        # 3. Build the decoder with its specific input shapes.
        self.decoder.build((representation_shape, target_x_shape))

        # 4. Check for shapes
        xdims = context_x_shape[-1]  # No. of channels / dims in x
        ydims = context_y_shape[-1]  # No. of channels / dims in y
        if xdims > 1:
            warnings.warn(
                "Note: The encoder implementation doesn't implicitly "
                "depend on xdims. Ensure you use an encoder size that is "
                "appropriate for your data.",
                UserWarning,
            )
        assert ydims == self.output_dims

        # 5. Call the parent's build method at the end.
        super().build(input_shape)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # --- Latent Path ---
        # The prior is based on the context set C
        prior_dist = self.latent_encoder(context_x, context_y)

        if training:
            # The posterior is based on the full target set T (which includes C)
            posterior_dist = self.latent_encoder(target_x, target_y)
            # Sample z from the posterior for reconstruction during training
            z = posterior_dist.sample()
        else:
            # At test time, sample from prior as posterior is unavailable
            posterior_dist = None  # Not available at test time
            z = prior_dist.sample()

        # Expand z to match the number of target points
        num_targets = ops.shape(target_x)[1]
        z_rep = ops.broadcast_to(
            ops.expand_dims(z, axis=1),
            (ops.shape(z)[0], num_targets, ops.shape(z)[-1]),
        )

        # --- Deterministic Path ---
        det_rep = self.det_encoder(context_x, context_y)  # Shape: (batch, 1, dim)
        det_rep = ops.broadcast_to(
            det_rep,
            (ops.shape(det_rep)[0], num_targets, ops.shape(det_rep)[-1]),
        )  # Shape: (batch, num_targets, dim)

        # --- Combine and Decode ---
        representation = ops.concatenate([det_rep, z_rep], axis=-1)
        mean, std = self.decoder(representation, target_x)

        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    @tf.function(jit_compile=True)
    def train_step(self, context_x, context_y, target_x, target_y):
        with tf.GradientTape() as tape:
            # Forward pass - `training=True` to get posterior
            pred_dist, prior, posterior = self(
                (context_x, context_y, target_x, target_y), training=True
            )

            # 1. Reconstruction Loss
            log_likelihood = pred_dist.log_prob(target_y)

            # 2. KL Divergence (Regularization)
            kl_div = tfp.distributions.kl_divergence(posterior, prior)
            # Sum KL over latent dims, but average over batch
            kl_div = ops.mean(ops.sum(kl_div, axis=-1))

            # 3. ELBO Loss
            # We want to maximize ELBO = log_likelihood - kl_div
            # So we minimize -ELBO = -log_likelihood + kl_div
            num_targets_float = ops.cast(ops.shape(target_x)[1], "float32")
            # The KL is weighted, a common practice to balance the terms
            reconstruction_loss = -ops.mean(log_likelihood)
            loss_value = reconstruction_loss + kl_div / num_targets_float

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)

        return loss_value, reconstruction_loss, kl_div

    @tf.function(jit_compile=True)
    def test_step(self, context_x, context_y, pred_x):
        # We don't have target_y, so it's None. `call` will sample from the prior.
        target_y = None
        pred_dist, _, _ = self((context_x, context_y, pred_x, target_y), training=False)
        return pred_dist.mean(), pred_dist.stddev()

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
        pred_points=None,
        plot_every=1000,
        seed_val=None,
    ):
        self.optimizer = optimizer

        if pbar:
            progbar = Progbar(epochs, stateful_metrics=["loss", "recon_loss", "kl_div"])

        if plotcb:
            assert (
                X_val is not None and y_val is not None
            ), "Validation data must be provided if plotcb is True."
            if pred_points is None and dataset_type == "gp":
                pred_points = X_val.shape[1]

        # Initialize callbacks
        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)

        callbacks.on_train_begin()  # Called at the start of training

        # no batching yet - possibly implement later
        for epoch in range(1, epochs + 1):
            # Reset metrics at start of each epoch
            callbacks.on_epoch_begin(epoch)  # Called at start of epoch

            X_train_batch, y_train_batch = utils.get_train_batch(
                X_train, y_train, batch_size=batch_size
            )

            num_context = utils._sample_num_context(num_context_range)

            # sample context set from fixed target set
            context_x_train, context_y_train = utils.get_context_set(
                X_train_batch,
                y_train_batch,
                num_context=num_context,
                num_context_mode=num_context_mode,
            )

            target_x_train = X_train_batch
            target_y_train = y_train_batch

            loss, reconstruction_loss, kl_div = self.train_step(
                context_x_train, context_y_train, target_x_train, target_y_train
            )

            # Update metrics with both loss and mae if reqd.
            logs = {
                "loss": float(loss),
                "reconstruction_loss": float(reconstruction_loss),
                "kl_div": float(kl_div),
            }
            callbacks.on_epoch_end(epoch, logs)  # Called at end of epoch

            if pbar:
                progbar.update(epoch, values=[(k, float(v)) for k, v in logs.items()])

            if plotcb and (epoch % plot_every == 0):
                if dataset_type == "gp":
                    utils.gplike_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        pred_points=pred_points,
                        num_context_range=num_context_range,
                        num_context_mode=num_context_mode,
                        epoch=epoch,
                        seed=seed_val,
                    )
                elif dataset_type == "mnist":
                    utils.mnist_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        num_context_range=num_context_range,
                        epoch=epoch,
                        seed=seed_val,
                    )

        return history

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "det_encoder_sizes": self.det_encoder_sizes,
                "latent_encoder_sizes": self.latent_encoder_sizes,
                "num_latents": self.num_latents,
                "decoder_sizes": self.decoder_sizes_hidden,
                "output_dims": self.output_dims,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class ANP(keras.Model):
    def __init__(
        self,
        encoder_sizes=[128] * 4,  # Simplified from att_encoder_sizes
        num_heads=8,
        latent_encoder_sizes=[128] * 2,
        num_latents=128,
        decoder_sizes=[128] * 2,
        output_dims=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_dims = output_dims
        self.num_latents = num_latents
        self.num_heads = num_heads

        self.encoder_sizes = encoder_sizes
        self.latent_encoder_sizes = latent_encoder_sizes
        self.decoder_sizes_hidden = list(decoder_sizes)

        self.att_encoder = AttentiveEncoder(self.encoder_sizes, self.num_heads)
        self.latent_encoder = LatentEncoder(self.latent_encoder_sizes, self.num_latents)

        full_decoder_sizes = self.decoder_sizes_hidden + [2 * self.output_dims]
        self.decoder = DeterministicDecoder(full_decoder_sizes)

    def build(self, input_shape):
        (context_x_shape, context_y_shape, target_x_shape, target_y_shape) = input_shape

        self.att_encoder.build((context_x_shape, context_y_shape, target_x_shape))
        self.latent_encoder.build((context_x_shape, context_y_shape))

        det_rep_dim = self.encoder_sizes[-1]
        latent_rep_dim = self.num_latents
        representation_shape = (None, None, det_rep_dim + latent_rep_dim)

        self.decoder.build((representation_shape, target_x_shape))

        super().build(input_shape)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, target_y = inputs

        # --- Latent Path ---
        # The prior is based on the context set C
        prior_dist = self.latent_encoder(context_x, context_y)

        if training:
            # The posterior is based on the full target set T (which includes C)
            posterior_dist = self.latent_encoder(target_x, target_y)
            # Sample z from the posterior for reconstruction during training
            z = posterior_dist.sample()
        else:
            # At test time, sample from prior as posterior is unavailable
            posterior_dist = None  # Not available at test time
            z = prior_dist.sample()

        # Expand z to match the number of target points
        num_targets = ops.shape(target_x)[1]
        z_rep = ops.repeat(ops.expand_dims(z, axis=1), num_targets, axis=1)

        # --- Deterministic Path ---
        det_rep = self.att_encoder(context_x, context_y, target_x)

        # --- Combine and Decode ---
        representation = ops.concatenate([det_rep, z_rep], axis=-1)
        mean, std = self.decoder(representation, target_x)

        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    @tf.function(jit_compile=True)
    def train_step(self, context_x, context_y, target_x, target_y):
        with tf.GradientTape() as tape:
            # Forward pass - `training=True` to get posterior
            pred_dist, prior, posterior = self(
                (context_x, context_y, target_x, target_y), training=True
            )

            # 1. Reconstruction Loss
            log_likelihood = pred_dist.log_prob(target_y)

            # 2. KL Divergence (Regularization)
            kl_div = tfp.distributions.kl_divergence(posterior, prior)
            # Sum KL over latent dims, but average over batch
            kl_div = ops.mean(ops.sum(kl_div, axis=-1))

            # 3. ELBO Loss
            # We want to maximize ELBO = log_likelihood - kl_div
            # So we minimize -ELBO = -log_likelihood + kl_div
            num_targets_float = ops.cast(ops.shape(target_x)[1], "float32")
            # The KL is weighted, a common practice to balance the terms
            reconstruction_loss = -ops.mean(log_likelihood)
            loss_value = reconstruction_loss + kl_div / num_targets_float

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)

        return loss_value, reconstruction_loss, kl_div

    @tf.function(jit_compile=True)
    def test_step(self, context_x, context_y, pred_x):
        # We don't have target_y, so it's None. `call` will sample from the prior.
        target_y = None
        pred_dist, _, _ = self((context_x, context_y, pred_x, target_y), training=False)
        return pred_dist.mean(), pred_dist.stddev()

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
        pred_points=None,
        plot_every=1000,
        seed_val=None,
    ):
        self.optimizer = optimizer

        if pbar:
            progbar = Progbar(epochs, stateful_metrics=["loss", "recon_loss", "kl_div"])

        if plotcb:
            assert (
                X_val is not None and y_val is not None
            ), "Validation data must be provided if plotcb is True."
            if pred_points is None and dataset_type == "gp":
                pred_points = X_val.shape[1]

        # Initialize callbacks
        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)

        callbacks.on_train_begin()  # Called at the start of training

        # no batching yet - possibly implement later
        for epoch in range(1, epochs + 1):
            # Reset metrics at start of each epoch
            callbacks.on_epoch_begin(epoch)  # Called at start of epoch

            X_train_batch, y_train_batch = utils.get_train_batch(
                X_train, y_train, batch_size=batch_size
            )

            num_context = utils._sample_num_context(num_context_range)

            # sample context set from fixed target set
            context_x_train, context_y_train = utils.get_context_set(
                X_train_batch,
                y_train_batch,
                num_context=num_context,
                num_context_mode=num_context_mode,
            )

            target_x_train = X_train_batch
            target_y_train = y_train_batch

            loss, reconstruction_loss, kl_div = self.train_step(
                context_x_train, context_y_train, target_x_train, target_y_train
            )

            # Update metrics with both loss and mae if reqd.
            logs = {
                "loss": float(loss),
                "reconstruction_loss": float(reconstruction_loss),
                "kl_div": float(kl_div),
            }
            callbacks.on_epoch_end(epoch, logs)  # Called at end of epoch

            if pbar:
                progbar.update(epoch, values=[(k, float(v)) for k, v in logs.items()])

            if plotcb and (epoch % plot_every == 0):
                if dataset_type == "gp":
                    utils.gplike_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        pred_points=pred_points,
                        num_context_range=num_context_range,
                        num_context_mode=num_context_mode,
                        epoch=epoch,
                        seed=seed_val,
                    )
                elif dataset_type == "mnist":
                    utils.mnist_val_step(
                        model=self,
                        X_val=X_val,
                        y_val=y_val,
                        num_context_range=num_context_range,
                        epoch=epoch,
                        seed=seed_val,
                    )
        return history

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sizes": self.encoder_sizes,
                "num_heads": self.num_heads,
                "latent_encoder_sizes": self.latent_encoder_sizes,
                "num_latents": self.num_latents,
                "decoder_sizes": self.decoder_sizes_hidden,
                "output_dims": self.output_dims,
            }
        )
        return config
