# keras-neuralprocesses
 Neural Processes written in Keras.

Based on Keras 3 (using TF backend right now - plans to support PyTorch/Jax in the future).


## Installation

To install from PyPi:
```sh
pip install keras_neural_processes
```

To install from GitHub:
```sh
pip install git+https://github.com/sidchaini/keras-neural-processes
```

## Quickstart

```py
import keras
import keras_neural_processes as knp

model = knp.CNP()
history = model.train(
    X_train=X_train,
    y_train=y_train,
    epochs=epochs,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    batch_size=64,
    num_context_range=[10,20],
    num_context_mode='each',
    dataset_type="gp",
    X_val=X_test, y_val=y_test,
    plotcb=True,
    pbar=True,
    plot_every=plot_every,
)```