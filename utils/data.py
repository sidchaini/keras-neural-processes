### Data copypaste
# ---
### GP CODE from [CNP repo](https://github.com/google-deepmind/neural-processes/blob/master/conditional_neural_process.ipynb)

import tensorflow as tf


# shape is (samples, points, channels) for x and y both
def get_context_set(
    target_x,
    target_y,
    num_context=7,  # ADD SUPPORT FOR RANGE LATER - then sample different num
    # context_frac_range=[0.01, 0.05]
):
    """
    Remember that context sets are a subset of the target set.
    context_frac is fraction of target set to be made context set.
    """
    assert (
        target_x.shape[:-1] == target_y.shape[:-1]
    )  # the inputs should have same no. of samples and points, channels may differ
    n_points = target_x.shape[1]

    # n_samples = target_x.shape[0]
    # nx_channels = target_x.shape[2]
    # ny_channels = target_y.shape[2]

    # num_context = np.random.randint(low=context_frac_range[0]*n_samples, high=context_frac_range[1]*n_samples, dtype="int")
    # context_points = int(context_frac*n_points)

    indices = np.random.choice(
        n_points, num_context, replace=False
    )  # choose num_context points from n_points e.g. 6 from 400 randomly

    if isinstance(target_x, np.ndarray):
        context_x = target_x[:, indices, :]
        context_y = target_y[:, indices, :]
    elif tf.is_tensor(target_x):
        context_x = tf.gather(target_x, indices, axis=1)
        context_y = tf.gather(target_y, indices, axis=1)
    else:
        print("target_x is neither a NumPy array nor a TensorFlow tensor")

    return context_x, context_y


### Data copypaste
# ---
### GP CODE from [CNP repo](https://github.com/google-deepmind/neural-processes/blob/master/conditional_neural_process.ipynb)


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        num_target=400,
        x_size=1,
        y_size=1,
        l1_scale=0.4,
        sigma_scale=1.0,
    ):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
          batch_size: An integer.
          max_num_context: The max number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_size: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._num_target = num_target

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
          xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
              the values of the x-axis data.
          l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
              parameter of the Gaussian kernel.
          sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
              of the std.
          sigma_noise: Float, std of the noise that we add for stability.

        Returns:
          The kernel, a float tensor with shape
          `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        num_total_points = tf.shape(xdata)[1]

        # Expand and take the difference
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = tf.reduce_sum(
            norm, -1
        )  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * tf.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
          context_x, context_x, target_x, target_y
        """
        num_context = tf.random.uniform(
            shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32
        )

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.

        # #SC THIS PART JUST CREATES IDENTICAL X_VALS 64 times
        # EQUIVALENT NUMPY:
        # # Parameters
        # batch_size = 64  # Example batch size
        # start = -2.0
        # stop = 2.0
        # step = 1.0 / 100

        # # Create a 1D array of evenly spaced values
        # x_values_1d = np.arange(start, stop, step, dtype=np.float32)

        # # Expand dimensions to make it 2D
        # x_values_2d = np.expand_dims(x_values_1d, axis=0)

        # # Tile the array to create a batch
        # x_values_batch = np.tile(x_values_2d, (batch_size, 1))

        num_total_points = self._num_target
        x_values = tf.tile(
            tf.expand_dims(tf.range(-2.0, 2.0, 1.0 / 100, dtype=tf.float32), axis=0),
            [self._batch_size, 1],
        )
        # #SC NOTE: Above 1 should be no. of xchannels I think
        x_values = tf.expand_dims(x_values, axis=-1)

        # # #SC
        # # WHEN TRAINING
        # # the number of target points and their x-positions are
        # # selected at random
        # # i.e b/w 3 and 10 points are chosen as no. of context.
        # # and b/w 2 and 10 as no. of target
        # # Note that now, all the points are different (note lack of tiling)

        #   num_target = tf.random.uniform(
        #       shape=(), minval=2, maxval=self._max_num_context, dtype=tf.int32)
        #   num_total_points = num_context + num_target
        #   x_values = tf.random.uniform(
        #       [self._batch_size, num_total_points, self._x_size], -2, 2)

        # Set kernel parameters
        l1 = (
            tf.ones(shape=[self._batch_size, self._y_size, self._x_size])
            * self._l1_scale
        )
        sigma_f = tf.ones(shape=[self._batch_size, self._y_size]) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = tf.cast(tf.linalg.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = tf.matmul(
            cholesky,
            tf.random.normal([self._batch_size, self._y_size, num_total_points, 1]),
        )

        # [batch_size, num_total_points, y_size]
        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        # Select the targets
        # #SC: WHEN TESTING everything is a target
        target_x = x_values
        target_y = y_values

        # #SC: WHEN TESTING context is still a random sample of 10 points
        # # Shuffling here b/c TEST XVALS were sequential
        # Select the observations
        idx = tf.random.shuffle(tf.range(self._num_target))
        context_x = tf.gather(x_values, idx[:num_context], axis=1)
        context_y = tf.gather(y_values, idx[:num_context], axis=1)

        # else:
        #     # #SC: WHEN TRAINING, target is first n points.
        #     # No shuffling here b/c TRAIN XVALS were random and not sequential
        #     # Select the targets which will consist of the context points as well as
        #     # some new target points
        #     target_x = x_values[:, : num_target + num_context, :]
        #     target_y = y_values[:, : num_target + num_context, :]

        #     # #SC: WHEN TRAINING, context is first m (<n) points
        #     # Select the observations
        #     context_x = x_values[:, :num_context, :]
        #     context_y = y_values[:, :num_context, :]

        return context_x, context_y, target_x, target_y
        # num_total_points = target_x.shape[1]
        # num_context_points = num_context


# So generate 65, first 64 become train, then get sampled. 1 remains test.
# So there probably needs to be some way to pass the sampling strategy to the model


def generate_target_curves(train_samples=64, test_samples=1, max_num_context=10):
    tot_curves = train_samples + test_samples
    obj = GPCurvesReader(batch_size=tot_curves, max_num_context=max_num_context)
    _, __, target_x, target_y = obj.generate_curves()
    target_x = target_x.numpy()
    target_y = target_y.numpy()

    return target_x, target_y


def generate_deepmind_curves(train_samples=64, test_samples=1, max_num_context=10):
    tot_curves = train_samples + test_samples
    obj = GPCurvesReader(batch_size=tot_curves, max_num_context=max_num_context)
    _, __, target_x, target_y = obj.generate_curves()
    target_x = target_x.numpy()
    target_y = target_y.numpy()

    target_x_train = target_x[:train_samples, :, :]
    target_y_train = target_y[:train_samples, :, :]
    target_x_test = target_x[train_samples:, :, :]
    target_y_test = target_y[train_samples:, :, :]

    return target_x_train, target_y_train, target_x_test, target_y_test
