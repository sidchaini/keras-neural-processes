import sys
import warnings

import numpy as np
import keras
from keras import ops, layers
from keras.utils import Progbar
import tensorflow as tf
import tensorflow_probability as tfp

from .data import create_stratified_np_dataset, get_context_set_dense
from .validation import val_step_physical
from .CONSTANTS import get_ADDFLUX_FOR_MAG_CONST

# =============================================================================
# 1. COMPOSITIONAL AGGREGATORS
# =============================================================================


@keras.saving.register_keras_serializable()
class SimpleAggregator(layers.Layer):
    """Aggregates point representations using a simple permutation-invariant function."""

    def __init__(self, aggregate_fn="mean", **kwargs):
        if isinstance(aggregate_fn, str):
            self.aggregate_str = aggregate_fn  # Store the string name
            self.aggregate_fn = getattr(ops, aggregate_fn)
        else:
            if not callable(aggregate_fn):
                raise ValueError("`aggregate_fn` must be a callable or a valid string.")
            self.aggregate_fn = aggregate_fn
            self.aggregate_str = "custom_function"
            print(
                "[WARNING] Saving models with custom aggregate function is not yet supported.",
                file=sys.stderr,
            )
        super().__init__(**kwargs)

    def build(
        self, context_representations_shape, target_x_shape=None, context_x_shape=None
    ):
        # Keras parent build for context_representations_shape
        # target_x_shape and context_x_shape are ignored, only there
        # for API compatibility
        super().build(context_representations_shape)

    def call(self, context_representations, target_x=None, context_x=None):
        return self.aggregate_fn(context_representations, axis=1, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({"aggregate_fn": self.aggregate_str})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class AttentionAggregator(layers.Layer):
    """Aggregates context using self- and cross-attention."""

    def __init__(
        self,
        num_heads,
        use_self_attention=True,
        cross_attn_key="context_representation",
        *,
        self_attention=None,
        self_attention_norm=None,
        query_mlp=None,
        key_mlp=None,
        cross_attention=None,
        cross_attention_norm=None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.use_self_attention = use_self_attention
        if cross_attn_key not in ["context_x", "context_representation"]:
            raise ValueError(
                "`cross_attn_key` can only be `context_x` or `context_representation`."
            )

        self.cross_attn_key = cross_attn_key

        # Will create layers below in build
        self._self_attention = self_attention
        self._self_attention_norm = self_attention_norm
        self._query_mlp = query_mlp
        self._key_mlp = key_mlp
        self._cross_attention = cross_attention
        self._cross_attention_norm = cross_attention_norm

        super().__init__(**kwargs)

    def build(self, context_representations_shape, target_x_shape, context_x_shape):
        rep_dim = context_representations_shape[-1]
        head_dim = max(1, rep_dim // self.num_heads)

        if self.use_self_attention:
            if self._self_attention is None:
                self._self_attention = layers.MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=head_dim
                )
            if self._self_attention_norm is None:
                self._self_attention_norm = layers.LayerNormalization()
        if self._query_mlp is None:
            self._query_mlp = layers.Dense(rep_dim, name="query_mlp")
        if self.cross_attn_key == "context_x":
            if self._key_mlp is None:
                self._key_mlp = layers.Dense(rep_dim, name="key_mlp")
        if self._cross_attention is None:
            self._cross_attention = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=head_dim, name="cross_attention"
            )
        if self._cross_attention_norm is None:
            self._cross_attention_norm = layers.LayerNormalization(
                name="cross_attention_norm"
            )

        # 3. Keras parent build for everything else
        super().build((context_representations_shape, target_x_shape, context_x_shape))

    def call(self, context_representations, target_x, context_x=None):

        # 1. Self Attention
        if self.use_self_attention:
            self_att_output = self._self_attention(
                query=context_representations,
                value=context_representations,
                key=context_representations,
            )
            context_representations = self._self_attention_norm(
                context_representations + self_att_output
            )

        # 2. Cross Attention
        queries = self._query_mlp(target_x)
        values = context_representations

        if self.cross_attn_key == "context_x":
            # Key is either context x-locations only (like the DeepMind notebook)
            if context_x is None:
                raise ValueError("`context_x` can not be `None`.")

            keys = self._key_mlp(context_x)
        else:
            # Or key is full context representation (like full ANP paper)
            keys = context_representations

        cross_att_output = self._cross_attention(query=queries, value=values, key=keys)
        final_representations = self._cross_attention_norm(queries + cross_att_output)

        return final_representations

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "use_self_attention": self.use_self_attention,
                "cross_attn_key": self.cross_attn_key,
                "self_attention": keras.saving.serialize_keras_object(
                    self._self_attention
                ),
                "self_attention_norm": keras.saving.serialize_keras_object(
                    self._self_attention_norm
                ),
                "query_mlp": keras.saving.serialize_keras_object(self._query_mlp),
                "key_mlp": keras.saving.serialize_keras_object(self._key_mlp),
                "cross_attention": keras.saving.serialize_keras_object(
                    self._cross_attention
                ),
                "cross_attention_norm": keras.saving.serialize_keras_object(
                    self._cross_attention_norm
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["self_attention"] = keras.saving.deserialize_keras_object(
            config.get("self_attention")
        )
        config["self_attention_norm"] = keras.saving.deserialize_keras_object(
            config.get("self_attention_norm")
        )
        config["query_mlp"] = keras.saving.deserialize_keras_object(
            config.get("query_mlp")
        )
        config["key_mlp"] = keras.saving.deserialize_keras_object(config.get("key_mlp"))
        config["cross_attention"] = keras.saving.deserialize_keras_object(
            config.get("cross_attention")
        )
        config["cross_attention_norm"] = keras.saving.deserialize_keras_object(
            config.get("cross_attention_norm")
        )
        return cls(**config)


# =============================================================================
# 2. COMPOSITIONAL ENCODERS AND DECODER
# =============================================================================


@keras.saving.register_keras_serializable()
class DeterministicEncoder(layers.Layer):
    """Compositional encoder: applies a encoder_net MLP then an aggregator."""

    def __init__(self, encoder_net, aggregator, **kwargs):
        self.encoder_net = encoder_net
        self.aggregator = aggregator
        super().__init__(**kwargs)

    def build(self, context_x_shape, context_y_shape, target_x_shape=None):
        # Note how target_x is always passed, but
        # not used unless aggregator is attention based

        # 1. Build encoder NN first
        encoder_net_input_shape = list(context_x_shape)
        encoder_net_input_shape[-1] += context_y_shape[-1]
        if hasattr(self.encoder_net, "build") and not self.encoder_net.built:
            self.encoder_net.build(tuple(encoder_net_input_shape))

        # 2. Build aggregator next
        context_rep_shape = self.encoder_net.output_shape
        if hasattr(self.aggregator, "build") and not self.aggregator.built:
            # Passing all shapes here: aggregator will use what it needs
            # based on if it's simple or attention based
            self.aggregator.build(context_rep_shape, target_x_shape, context_x_shape)

        # 3. Keras parent build for everything else
        super().build((context_x_shape, context_y_shape, target_x_shape))

    def call(self, context_x, context_y, target_x=None):
        encoder_input = ops.concatenate([context_x, context_y], axis=-1)
        context_representations = self.encoder_net(encoder_input)

        # target_x and context_x is ignored if it's a simple aggregator
        # and only used for attention aggregator
        return self.aggregator(context_representations, target_x, context_x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_net": keras.saving.serialize_keras_object(self.encoder_net),
                "aggregator": keras.saving.serialize_keras_object(self.aggregator),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["encoder_net"] = keras.saving.deserialize_keras_object(
            config["encoder_net"]
        )
        config["aggregator"] = keras.saving.deserialize_keras_object(
            config["aggregator"]
        )
        return cls(**config)


@keras.saving.register_keras_serializable()
class LatentEncoder(layers.Layer):
    """Encodes a set of points into the parameters of a latent distribution."""

    def __init__(
        self,
        encoder_net,
        num_latents,
        aggregator=SimpleAggregator(aggregate_fn="mean"),
        *,
        apply_penultimate_layer=False,
        mu_layer=None,
        log_sigma_layer=None,
        penultimate_layer=None,
        **kwargs,
    ):
        self.encoder_net = encoder_net
        self.num_latents = num_latents
        self.aggregator = aggregator
        self.apply_penultimate_layer = apply_penultimate_layer

        self._mu_layer = mu_layer
        self._log_sigma_layer = log_sigma_layer

        if self.apply_penultimate_layer:
            self._penultimate_layer = penultimate_layer
        super().__init__(**kwargs)

    # target_x_shape=None
    def build(self, context_x_shape, context_y_shape):
        # 1. first build encoder net so we know the output shape
        encoder_net_input_shape = list(context_x_shape)
        encoder_net_input_shape[-1] += context_y_shape[-1]
        if hasattr(self.encoder_net, "build") and not self.encoder_net.built:
            self.encoder_net.build(tuple(encoder_net_input_shape))
        encoder_output_dim = self.encoder_net.output_shape[-1]

        # 2. use encoder output shape to build penultimate layer
        if self.apply_penultimate_layer:
            if self._penultimate_layer is None:
                penultimate_units = (self.num_latents + encoder_output_dim) // 2
                self._penultimate_layer = layers.Dense(
                    penultimate_units, activation=None, name="penultimate_layer"
                )

        if self._mu_layer is None:
            self._mu_layer = layers.Dense(self.num_latents, name="mu_layer")
        if self._log_sigma_layer is None:
            self._log_sigma_layer = layers.Dense(
                self.num_latents, name="log_sigma_layer"
            )

        # 3. Keras parent build for everything else
        super().build((context_x_shape, context_y_shape))

    def call(self, x, y):
        encoder_input = ops.concatenate([x, y], axis=-1)
        context_representations = self.encoder_net(encoder_input)
        hidden = self.aggregator(context_representations)
        hidden = ops.squeeze(hidden, axis=1)

        if self.apply_penultimate_layer:
            hidden = self._penultimate_layer(hidden)

        mu = self._mu_layer(hidden)
        log_sigma = self._log_sigma_layer(hidden)

        min_latent_std = 1e-3
        sigma = ops.maximum(min_latent_std, ops.softplus(log_sigma))
        return tfp.distributions.Normal(loc=mu, scale=sigma)

    def get_config(self):
        config = super().get_config()
        penultimatelayer = (
            keras.saving.serialize_keras_object(self._penultimate_layer)
            if self.apply_penultimate_layer
            else None
        )
        config.update(
            {
                "encoder_net": keras.saving.serialize_keras_object(self.encoder_net),
                "num_latents": self.num_latents,
                "aggregator": keras.saving.serialize_keras_object(self.aggregator),
                "apply_penultimate_layer": self.apply_penultimate_layer,
                "mu_layer": keras.saving.serialize_keras_object(self._mu_layer),
                "log_sigma_layer": keras.saving.serialize_keras_object(
                    self._log_sigma_layer
                ),
                "penultimate_layer": penultimatelayer,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config["encoder_net"] = keras.saving.deserialize_keras_object(
            config["encoder_net"]
        )
        config["aggregator"] = keras.saving.deserialize_keras_object(
            config["aggregator"]
        )
        config["mu_layer"] = keras.saving.deserialize_keras_object(config["mu_layer"])
        config["log_sigma_layer"] = keras.saving.deserialize_keras_object(
            config["log_sigma_layer"]
        )
        config["penultimate_layer"] = keras.saving.deserialize_keras_object(
            config.get("penultimate_layer")
        )
        return cls(**config)


@keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    """The decoder neural network."""

    def __init__(self, full_decoder_net, **kwargs):
        self.full_decoder_net = full_decoder_net
        super().__init__(**kwargs)

    def build(self, representation_shape, target_x_shape):
        # 1. Get proper shapes of inputs
        rep_dim = representation_shape[-1]
        target_x_dim = target_x_shape[-1]
        # Also Allow for dynamic dimensions (None) during tracing
        decoder_input_dim = (
            None
            if (rep_dim is None or target_x_dim is None)
            else rep_dim + target_x_dim
        )
        decoder_net_input_shape = list(target_x_shape[:-1]) + [decoder_input_dim]

        # 1. Build decoder neural net
        if hasattr(self.full_decoder_net, "build") and not self.full_decoder_net.built:
            self.full_decoder_net.build(tuple(decoder_net_input_shape))

        # 2. Keras parent build for everything else
        super().build((representation_shape, target_x_shape))

    def call(self, representation, target_x):
        decoder_input = ops.concatenate([representation, target_x], axis=-1)
        decodings = self.full_decoder_net(decoder_input)
        mean, log_std = ops.split(decodings, 2, axis=-1)

        mean = ops.softplus(mean)  # positive mean

        min_obs_std = 1e-3
        std = ops.maximum(min_obs_std, ops.softplus(log_std))

        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "full_decoder_net": keras.saving.serialize_keras_object(
                    self.full_decoder_net
                )
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["full_decoder_net"] = keras.saving.deserialize_keras_object(
            config["full_decoder_net"]
        )
        return cls(**config)


# =============================================================================
# 3. BASE, MIXIN, AND FINAL MODEL CLASSES
# =============================================================================


def _make_mlp(
    network_or_list,
    default_mlp_list,
    # output_size=None,
    activation="relu",
    **kwargs,
):
    """
    Builds a Keras Sequential MLP from a flexible specification.
    """
    if isinstance(network_or_list, keras.Model):
        return network_or_list

    if network_or_list is None:
        layer_sizes = default_mlp_list
    elif isinstance(network_or_list, list):
        layer_sizes = network_or_list
    else:
        warnings.warn(
            "Warning: Please provide a keras.Model or a list of "
            "integers to make an MLP model. Creating a default model instead.",
            UserWarning,
        )
        layer_sizes = default_mlp_list

    hidden_layers = []
    # All but the last layer have the specified activation
    for size in layer_sizes[:-1]:
        hidden_layers.append(layers.Dense(size, activation=activation))
    # Final layer is linear
    hidden_layers.append(layers.Dense(layer_sizes[-1], activation=None))

    return keras.Sequential(hidden_layers, **kwargs)


class BaseNeuralProcess(keras.Model):
    def __init__(self, **kwargs):
        self._compiled_train_step = None
        self._compiled_test_step = None
        super().__init__(**kwargs)

    def _prepare_data(self, inputs):
        if isinstance(inputs, dict):
            return (
                inputs["context_x"],
                inputs["context_y"],
                inputs["target_x"],
                inputs.get("target_y", None),
            )
        return inputs

    def train_step(self, target_x, target_y, num_context):
        if self._compiled_train_step is None:
            print(
                "[INFO] JIT Compiling train_step... (This happens only once)",
                file=sys.stderr,
            )
            self._compiled_train_step = tf.function(
                self._train_step_logic,
                input_signature=[
                    tf.TensorSpec(
                        shape=[None, None, target_x.shape[-1]], dtype=target_x.dtype
                    ),
                    tf.TensorSpec(
                        shape=[None, None, target_y.shape[-1]], dtype=target_y.dtype
                    ),
                    tf.TensorSpec(
                        shape=(), dtype=tf.int32
                    ),  # New argument for num_context
                ],
                jit_compile=True,
            )
        return self._compiled_train_step(target_x, target_y, num_context)

    def test_step(self, context_x, context_y, pred_x):
        if self._compiled_test_step is None:
            print(
                "[INFO] JIT Compiling test_step... (This happens only once)",
                file=sys.stderr,
            )
            self._compiled_test_step = tf.function(
                self._test_step_logic,
                input_signature=[
                    tf.TensorSpec(
                        shape=[None, None, context_x.shape[-1]], dtype=context_x.dtype
                    ),
                    tf.TensorSpec(
                        shape=[None, None, context_y.shape[-1]], dtype=context_y.dtype
                    ),
                    tf.TensorSpec(
                        shape=[None, None, pred_x.shape[-1]], dtype=pred_x.dtype
                    ),
                ],
                jit_compile=True,
            )
        return self._compiled_test_step(context_x, context_y, pred_x)

    def predict(self, context_x, context_y, pred_x):
        return self.test_step(context_x, context_y, pred_x)

    def train(
        self,
        X_train,
        y_train,
        epochs,
        optimizer,
        batch_size=64,
        num_context_choices=[50],
        X_val=None,
        y_val=None,
        plotcb=True,
        pbar=True,
        plot_every=1000,
        seed=None,
        stratify_labels=None,
        num_target_points=100,
        # TODO FIX below all_scenarios, time_scaler and flux_scaler
        all_scenarios=None,
        time_scaler=None,
        flux_scaler=None,
    ):
        # FOR MIXED PRECISION
        # self.optimizer = keras.optimizers.LossScaleOptimizer(optimizer)

        self.optimizer = optimizer
        # choices_tensor = ops.convert_to_tensor(num_context_choices)
        num_choices = len(num_context_choices)  # noqa

        train_dataset = create_stratified_np_dataset(
            X_train,
            y_train,
            stratify_labels,
            batch_size=batch_size,
            num_target_points=num_target_points,
            slice_width=num_target_points,
        )
        train_iterator = iter(train_dataset)

        if pbar:
            metric_names = ["loss"] + getattr(self, "extra_metrics", [])
            progbar = Progbar(epochs, stateful_metrics=metric_names)

        history = keras.callbacks.History()
        callbacks = keras.callbacks.CallbackList([history], model=self)
        callbacks.on_train_begin()

        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_begin(epoch)

            # 1. Get the full (resampled) target set
            target_x, target_y, _ = next(train_iterator)

            # 2. Decide how many context points to sample (this is a simple Python op)
            num_context = np.random.choice(num_context_choices)

            # 3. Pass the full target set AND the number of context points to train_step
            #    The actual sampling will now happen inside the compiled function.
            logs = self.train_step(target_x, target_y, num_context)

            callbacks.on_epoch_end(epoch, logs)
            if pbar:
                progbar.update(epoch, values=[(k, float(v)) for k, v in logs.items()])

            if plotcb and (epoch % plot_every == 0):
                self.save(f"model_{int(epoch)}.keras")
                print("*" * 50)
                print(f"Iteration {epoch}")
                print("*" * 50)
                print(f"Train logs: {logs}")

                # run_physics_validation(...)
                resu = val_step_physical(
                    self,
                    all_scenarios,
                    time_scaler,
                    flux_scaler,
                    X_val,
                    y_val,
                    TEST_OBJ_CHOOSE=1,
                    SCENARIO_CHOOSE=0,
                    ADDFLUX_FOR_MAG_CONST=get_ADDFLUX_FOR_MAG_CONST(flux_scaler),
                    epochnum=epoch,
                )
                resu.to_parquet(f"results_{epoch}.parquet")

        callbacks.on_train_end()
        return history

    def _prepare_x(self, X, name="X"):
        """
        Validate and prepare an X tensor.
        Handles optional 1D to 2D conversion.
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        if len(X.shape) != 3:
            raise ValueError(
                f"{name} must be a 3D tensor `(num_samples, num_points, num_xchannels)`, but got {len(X.shape)} dims."
            )
        return X

    def _prepare_y(self, y, name="y"):
        """Validate and prepare a y tensor."""
        y = ops.convert_to_tensor(y, dtype="float32")
        if len(y.shape) != 3 or y.shape[-1] != self.output_dims:
            raise ValueError(
                f"{name} must have shape (num_samples, num_points, num_ychannels={self.output_dims}), "
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


class ConditionalModelMixin:
    """Provides the NLL loss and JIT-compiled logic for deterministic models (CNP)."""

    extra_metrics = []

    def _train_step_logic(self, target_x, target_y, num_context):
        context_x, context_y = get_context_set_dense(target_x, target_y, num_context)

        with tf.GradientTape() as tape:
            mean, std = self((context_x, context_y, target_x, target_y), training=True)

            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            loss_value = -ops.mean(dist.log_prob(target_y))  # NLL

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)
        return {"loss": loss_value}

    def _test_step_logic(self, context_x, context_y, pred_x):
        mean, std = self((context_x, context_y, pred_x, None), training=False)
        return mean, std


class LatentModelMixin:
    """Provides ELBO loss and JIT-compiled training/testing logic for NP and ANP."""

    extra_metrics = ["reconstruction_loss", "kl_div"]

    def _train_step_logic(self, target_x, target_y, num_context):
        context_x, context_y = get_context_set_dense(target_x, target_y, num_context)

        with tf.GradientTape() as tape:
            pred_dist, prior, posterior = self(
                (context_x, context_y, target_x, target_y), training=True
            )
            log_likelihood = pred_dist.log_prob(target_y)
            kl_div = tfp.distributions.kl_divergence(posterior, prior)
            reconstruction_loss = -ops.mean(log_likelihood)
            kl_div_loss = ops.mean(ops.sum(kl_div, axis=-1))
            num_targets_float = ops.cast(ops.shape(target_x)[1], "float32")
            kl_div_scaled = kl_div_loss / num_targets_float
            loss_value = reconstruction_loss + kl_div_scaled
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply(grads, self.trainable_weights)
        return {
            "loss": loss_value,
            "reconstruction_loss": reconstruction_loss,
            "kl_div_raw": kl_div_loss,
            "kl_div": kl_div_scaled,
        }

    def _test_step_logic(self, context_x, context_y, pred_x):
        target_y = None
        pred_dist, _, _ = self(
            (context_x, context_y, pred_x, target_y), training=False, sample_z=False
        )
        return pred_dist.mean(), pred_dist.stddev()


@keras.saving.register_keras_serializable()
class CNP(ConditionalModelMixin, BaseNeuralProcess):
    """
    Conditional Neural Process (CNP).
    A deterministic model that uses a simple, permutation-invariant aggregator.
    """

    def __init__(
        self,
        output_dims,
        det_encoder_net=None,
        decoder_net=None,
        *,  # ones below are for internal use only (e.g. for loading models)
        det_encoder=None,
        decoder=None,
        **kwargs,
    ):
        self.output_dims = output_dims

        # 1. Deterministic encoder
        # If a complete det_encoder is provided, use it directly.
        if det_encoder:  # Deserialization Path
            self._det_encoder = det_encoder
        # Otherwise, build a new one from its components.
        else:  # Initialization Path
            det_encoder_net = _make_mlp(
                network_or_list=det_encoder_net,
                default_mlp_list=[128, 128, 128, 128],
                name="det_encoder_net",
            )
            self._det_encoder = DeterministicEncoder(
                encoder_net=det_encoder_net,
                aggregator=SimpleAggregator("mean"),
                name="deterministic_encoder",
            )

        # 2. Decoder
        if decoder:  # Deserialization Path
            self._decoder = decoder
        else:  # Initialization Path
            decoder_net = _make_mlp(
                network_or_list=decoder_net,
                default_mlp_list=[128, 128],
                name="decoder_net",
            )
            full_decoder_net = keras.Sequential(
                [decoder_net, layers.Dense(2 * self.output_dims)],
                name="full_decoder_net",
            )
            self._decoder = Decoder(
                full_decoder_net=full_decoder_net,
                name="decoder",
            )

        super().__init__(**kwargs)

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape, _ = input_shape

        # 1. Build det. encoder w/ shape: context_x, context_y
        self._det_encoder.build(context_x_shape, context_y_shape)

        # 2. Build decoder with representation shape
        encoder_output_dim = self._det_encoder.encoder_net.output_shape[-1]
        decoder_rep_shape = (None, None, encoder_output_dim)
        self._decoder.build(decoder_rep_shape, target_x_shape)

        # 3. Keras parent build for everything else
        super().build(input_shape)

    def call(self, inputs, training=False):
        context_x, context_y, target_x, _ = self._prepare_data(inputs)

        # 1. Encode context to a single, global representation vector
        # Shape: (batch, 1, features)
        representation = self._det_encoder(context_x, context_y)

        # 2. Repeat the representation for each target point to match dimensions
        num_targets = ops.shape(target_x)[1]
        rep_shape = ops.shape(representation)
        representation = ops.broadcast_to(
            representation,
            (rep_shape[0], num_targets, rep_shape[2]),
        )  # Shape: (batch, num_targets, dim)

        # 3. Decode to get predicted distribution parameters
        mean, std = self._decoder(representation, target_x)
        return mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
                "det_encoder": keras.saving.serialize_keras_object(self._det_encoder),
                "decoder": keras.saving.serialize_keras_object(self._decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["det_encoder"] = keras.saving.deserialize_keras_object(
            config["det_encoder"]
        )
        config["decoder"] = keras.saving.deserialize_keras_object(config["decoder"])
        return cls(**config)


@keras.saving.register_keras_serializable()
class NP(LatentModelMixin, BaseNeuralProcess):
    """
    Neural Process (NP).
    A probabilistic model with a deterministic path (using mean-aggregation)
    and a latent path to model global uncertainty.
    """

    def __init__(
        self,
        output_dims,
        num_latents=128,
        latent_encoder_net=None,
        det_encoder_net=None,
        decoder_net=None,
        *,  # ones below are for internal use only (e.g. for loading models)
        latent_encoder=None,
        det_encoder=None,
        decoder=None,
        **kwargs,
    ):
        self.output_dims = output_dims
        self.num_latents = num_latents

        # 1. Latent encoder
        if latent_encoder:  # Deserialization Path
            self._latent_encoder = latent_encoder
            self.num_latents = self._latent_encoder.num_latents
        else:
            apply_penultimate_layer = kwargs.pop("apply_penultimate_layer", False)
            latent_encoder_net = _make_mlp(
                network_or_list=latent_encoder_net,
                default_mlp_list=[128, 128],
                name="latent_encoder_net",
            )
            self._latent_encoder = LatentEncoder(
                encoder_net=latent_encoder_net,
                num_latents=self.num_latents,
                apply_penultimate_layer=apply_penultimate_layer,
                name="latent_encoder",
            )

        # 2. Deterministic encoder
        if det_encoder:  # Deserialization Path
            self._det_encoder = det_encoder
        else:  # Initialization Path
            det_encoder_net = _make_mlp(
                network_or_list=det_encoder_net,
                default_mlp_list=[128, 128, 128, 128],
                name="det_encoder_net",
            )
            self._det_encoder = DeterministicEncoder(
                encoder_net=det_encoder_net,
                aggregator=SimpleAggregator("mean"),
                name="deterministic_encoder",
            )

        # 3. Decoder
        if decoder:  # Deserialization Path
            self._decoder = decoder
        else:  # Initialization Path
            decoder_net = _make_mlp(
                network_or_list=decoder_net,
                default_mlp_list=[128, 128],
                name="decoder_net",
            )
            full_decoder_net = keras.Sequential(
                [decoder_net, layers.Dense(2 * self.output_dims)],
                name="full_decoder_net",
            )
            self._decoder = Decoder(full_decoder_net=full_decoder_net, name="decoder")

        super().__init__(**kwargs)

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape, _ = input_shape

        # 1. Build latent encoder w/ shape: context_x, context_y
        self._latent_encoder.build(context_x_shape, context_y_shape)

        # 2. Build det. encoder w/ shape: context_x, context_y, target_x
        # NOTE: Target x because it's an Attentive Det. Encoder
        self._det_encoder.build(context_x_shape, context_y_shape, target_x_shape)

        # 3. Build decoder with shape of
        # representation - that'll be fed to decoder
        det_rep_dim = self._det_encoder.encoder_net.output_shape[-1]
        combined_rep_dim = det_rep_dim + self.num_latents
        decoder_rep_shape = (None, None, combined_rep_dim)
        self._decoder.build(decoder_rep_shape, target_x_shape)

        # 4. Keras parent build for everything else
        super().build(input_shape)

    def call(self, inputs, training=False, sample_z=True):
        context_x, context_y, target_x, target_y = self._prepare_data(inputs)

        # 1A. Latent Path: determine prior and posterior distributions for z
        prior_dist = self._latent_encoder(context_x, context_y)
        if training:
            # During training use full target set to form posterior
            posterior_dist = self._latent_encoder(target_x, target_y)
            z = posterior_dist.sample()  # CONSIDER TAKING MEAN FOR TRAINING
        else:
            # During inference sample from the prior
            posterior_dist = None
            if sample_z:
                z = prior_dist.sample()
            else:
                z = prior_dist.mean()

        num_targets = ops.shape(target_x)[1]
        z_expanded = ops.expand_dims(z, axis=1)
        z_shape = ops.shape(z_expanded)
        z_rep = ops.broadcast_to(
            z_expanded,
            (z_shape[0], num_targets, z_shape[2]),
        )

        # 1B. Deterministic Path (using mean encoder)
        # But broadcast to match number of targets
        det_rep = self._det_encoder(context_x, context_y)
        det_rep_shape = ops.shape(det_rep)  # Shape: (batch, 1, features)
        det_rep = ops.broadcast_to(
            det_rep,
            (det_rep_shape[0], num_targets, det_rep_shape[2]),
        )  # Shape: (batch, num_targets, dim)

        # 2. Combine both encoded representations
        representation = ops.concatenate([det_rep, z_rep], axis=-1)

        # 3. Apply decoder on combined representation
        mean, std = self._decoder(representation, target_x)
        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
                "latent_encoder": keras.saving.serialize_keras_object(
                    self._latent_encoder
                ),
                "det_encoder": keras.saving.serialize_keras_object(self._det_encoder),
                "decoder": keras.saving.serialize_keras_object(self._decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["latent_encoder"] = keras.saving.deserialize_keras_object(
            config["latent_encoder"]
        )
        config["det_encoder"] = keras.saving.deserialize_keras_object(
            config["det_encoder"]
        )
        config["decoder"] = keras.saving.deserialize_keras_object(config["decoder"])
        return cls(**config)


@keras.saving.register_keras_serializable()
class ANP(LatentModelMixin, BaseNeuralProcess):
    """
    Attentive Neural Process (ANP).
    A probabilistic model that uses an attention mechanism in its deterministic
    path to avoid underfitting, combined with a latent path for global uncertainty.
    """

    def __init__(
        self,
        output_dims,
        latent_encoder_net=None,
        num_latents=128,
        det_encoder_net=None,
        num_heads=8,
        decoder_net=None,
        *,  # ones below are for internal use only (e.g. for loading models)
        latent_encoder=None,
        det_encoder=None,
        decoder=None,
        **kwargs,
    ):
        self.output_dims = output_dims
        self.num_latents = num_latents

        # 1. Latent encoder
        if latent_encoder:  # Deserialization Path
            self._latent_encoder = latent_encoder
            self.num_latents = self._latent_encoder.num_latents
        else:  # Initialization Path
            apply_penultimate_layer = kwargs.pop("apply_penultimate_layer", False)
            latent_encoder_net = _make_mlp(
                network_or_list=latent_encoder_net,
                default_mlp_list=[128, 128],
                name="latent_encoder_net",
            )
            self._latent_encoder = LatentEncoder(
                encoder_net=latent_encoder_net,
                num_latents=self.num_latents,
                apply_penultimate_layer=apply_penultimate_layer,
                name="latent_encoder",
            )

        # 2. Deterministic encoder
        if det_encoder:  # Deserialization Path
            self._det_encoder = det_encoder
        else:  # Initialization Path
            use_self_attention = kwargs.pop("use_self_attention", True)
            cross_attn_key = kwargs.pop("cross_attn_key", "context_representation")
            det_encoder_net = _make_mlp(
                network_or_list=det_encoder_net,
                default_mlp_list=[128, 128, 128, 128],
                name="det_encoder_net",
            )
            self._det_encoder = DeterministicEncoder(
                encoder_net=det_encoder_net,
                aggregator=AttentionAggregator(
                    num_heads=num_heads,
                    use_self_attention=use_self_attention,
                    cross_attn_key=cross_attn_key,
                ),
                name="deterministic_encoder",
            )

        # 3. Decoder
        if decoder:  # Deserialization Path
            self._decoder = decoder
        else:  # Initialization Path
            decoder_net = _make_mlp(
                network_or_list=decoder_net,
                default_mlp_list=[128, 128],
                name="decoder_net",
            )
            full_decoder_net = keras.Sequential(
                [decoder_net, layers.Dense(2 * self.output_dims)],
                name="full_decoder_net",
            )
            self._decoder = Decoder(full_decoder_net=full_decoder_net, name="decoder")
        super().__init__(**kwargs)

    def build(self, input_shape):
        context_x_shape, context_y_shape, target_x_shape, _ = input_shape

        # 1. Build latent encoder w/ shape: context_x, context_y
        self._latent_encoder.build(context_x_shape, context_y_shape)

        # 2. Build det. encoder w/ shape: context_x, context_y, target_x
        # NOTE: Target x because it's an Attentive Det. Encoder
        self._det_encoder.build(context_x_shape, context_y_shape, target_x_shape)

        # 3. Build decoder with shape of
        # representation - that'll be fed to decoder
        det_rep_dim = self._det_encoder.encoder_net.output_shape[-1]
        combined_rep_dim = det_rep_dim + self.num_latents
        decoder_rep_shape = (None, None, combined_rep_dim)
        self._decoder.build(decoder_rep_shape, target_x_shape)

        # 4. Keras parent build for everything else
        super().build(input_shape)

    def call(self, inputs, training=False, sample_z=True):
        context_x, context_y, target_x, target_y = self._prepare_data(inputs)

        # 1A. Latent Path (like NP): determine prior and posterior distributions for z
        prior_dist = self._latent_encoder(context_x, context_y)
        if training:
            # During training use full target set to form posterior
            posterior_dist = self._latent_encoder(target_x, target_y)
            z = posterior_dist.sample()
        else:
            # During inference sample from the prior
            posterior_dist = None
            if sample_z:
                z = prior_dist.sample()
            else:
                z = prior_dist.mean()

        num_targets = ops.shape(target_x)[1]
        z_expanded = ops.expand_dims(z, axis=1)
        z_shape = ops.shape(z_expanded)
        z_rep = ops.broadcast_to(
            z_expanded,
            (z_shape[0], num_targets, z_shape[2]),
        )

        # 1B. Deterministic Path
        # note: unlike NP, no broadcasting reqd. as the attention
        # took care of it, which is built into the det. encoder.
        det_rep = self._det_encoder(context_x, context_y, target_x)

        # 2. Combine both encoded representations
        representation = ops.concatenate([det_rep, z_rep], axis=-1)

        # 3. Apply decoder on combined representation
        mean, std = self._decoder(representation, target_x)
        pred_dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return pred_dist, prior_dist, posterior_dist

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
                "latent_encoder": keras.saving.serialize_keras_object(
                    self._latent_encoder
                ),
                "det_encoder": keras.saving.serialize_keras_object(self._det_encoder),
                "decoder": keras.saving.serialize_keras_object(self._decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["latent_encoder"] = keras.saving.deserialize_keras_object(
            config["latent_encoder"]
        )
        config["det_encoder"] = keras.saving.deserialize_keras_object(
            config["det_encoder"]
        )
        config["decoder"] = keras.saving.deserialize_keras_object(config["decoder"])
        return cls(**config)
