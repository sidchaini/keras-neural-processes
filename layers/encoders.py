from keras import layers, ops


# Encoder network that processes context points
class DeterministicEncoder(layers.Layer):
    def __init__(self, hidden_sizes=[128, 128, 128, 128]):
        super().__init__()
        self.hidden_layers = []
        for size in hidden_sizes[:-1]:
            self.hidden_layers.append(layers.Dense(size, activation="relu"))
        # Final layer without activation
        self.hidden_layers.append(layers.Dense(hidden_sizes[-1]))

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
