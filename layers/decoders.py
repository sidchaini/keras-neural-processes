from keras import layers, ops


# Decoder network that makes predictions at target points
class DeterministicDecoder(layers.Layer):
    def __init__(self, hidden_sizes=[128, 128, 2]):
        super().__init__()
        self.hidden_layers = []
        for size in hidden_sizes[:-1]:
            self.hidden_layers.append(layers.Dense(size, activation="relu"))
        # Final layer outputs mean and log_std
        self.hidden_layers.append(layers.Dense(hidden_sizes[-1]))

    def call(self, representation, target_x):
        # Expand representation to match target batch size
        num_targets = ops.shape(target_x)[1]
        representation = ops.repeat(representation, num_targets, axis=1)

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
