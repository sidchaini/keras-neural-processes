import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops


def get_ragged_tensor(mydata, lengths=None):
    if lengths is None:
        lengths = [arr.shape[0] for arr in mydata]
        values = np.concatenate(mydata, axis=0)
        arr_ragged = tf.RaggedTensor.from_row_lengths(
            values=values, row_lengths=lengths
        )
    else:
        arr_ragged = tf.RaggedTensor.from_row_lengths(
            values=mydata, row_lengths=lengths
        )
    return arr_ragged


def compute_peak_indices(y_ragged):
    splits = y_ragged.row_splits.numpy()
    flat = y_ragged.values.numpy().reshape(-1)
    return np.array(
        [
            int(np.argmax(flat[splits[i] : splits[i + 1]]))
            for i in range(len(splits) - 1)
        ],
        dtype=np.int32,
    )


@tf.function
def _slice_and_normalize_target_set(x, y, num_target_points, slice_width, peak_idx):
    current_len = tf.shape(x)[0]

    effective_slice_width = tf.minimum(current_len, slice_width)
    max_start_index = current_len - effective_slice_width
    start_index = tf.cond(
        max_start_index > 0,
        lambda: tf.random.uniform(shape=(), maxval=max_start_index + 1, dtype=tf.int32),
        lambda: 0,
    )
    x_slice = x[start_index : start_index + effective_slice_width]
    y_slice = y[start_index : start_index + effective_slice_width]

    slice_len = tf.shape(x_slice)[0]

    def subsample_fn():
        indices = tf.random.shuffle(tf.range(slice_len))[:num_target_points]
        return tf.gather(x_slice, indices), tf.gather(y_slice, indices)

    def oversample_fn():
        indices = tf.random.uniform(
            shape=[num_target_points], maxval=slice_len, dtype=tf.int32
        )
        return tf.gather(x_slice, indices), tf.gather(y_slice, indices)

    resampled_x, resampled_y = tf.cond(
        slice_len >= num_target_points, true_fn=subsample_fn, false_fn=oversample_fn
    )

    time_col = resampled_x[..., 0:1]
    filter_col = resampled_x[..., 1:]

    slice_min_time = tf.reduce_min(time_col, axis=0, keepdims=True)
    centered_time = time_col - slice_min_time

    final_target_x = tf.concat([centered_time, filter_col], axis=-1)

    peak_t_rezeroed = tf.cond(
        peak_idx >= 0,
        lambda: x[peak_idx, 0] - slice_min_time[0, 0],
        lambda: tf.constant(float("inf"), dtype=x.dtype),
    )

    return final_target_x, resampled_y, peak_t_rezeroed


def create_stratified_np_dataset(
    X_ragged,
    y_ragged,
    labels,
    batch_size,
    num_target_points,
    slice_width,
    peak_idx_arr=None,
):
    grouped_indices = (
        pd.DataFrame({"label": ops.convert_to_numpy(labels)}).groupby("label").groups
    )
    unique_classes = list(grouped_indices.keys())
    num_classes = len(unique_classes)
    samples_per_class = batch_size // num_classes
    remainder = batch_size % num_classes

    n_objects = int(X_ragged.nrows().numpy())
    if peak_idx_arr is None:
        peak_idx_tensor = tf.constant(np.full(n_objects, -1, dtype=np.int32))
    else:
        peak_idx_tensor = tf.constant(np.asarray(peak_idx_arr, dtype=np.int32))

    datasets_by_class = []
    for i, class_id in enumerate(unique_classes):
        n_to_sample = samples_per_class + 1 if i < remainder else samples_per_class
        if n_to_sample == 0:
            continue

        class_indices = ops.convert_to_tensor(
            grouped_indices[class_id].to_numpy(), dtype="int32"
        )
        class_index_ds = tf.data.Dataset.from_tensor_slices(class_indices)

        @tf.function
        def gather_and_process(index):
            x = X_ragged[index]
            y = y_ragged[index]
            peak_idx = peak_idx_tensor[index]
            x_s, y_s, peak_t = _slice_and_normalize_target_set(
                x, y, num_target_points, slice_width, peak_idx
            )
            x_s.set_shape([num_target_points, x.shape[-1]])
            y_s.set_shape([num_target_points, y.shape[-1]])
            peak_t.set_shape([])
            return x_s, y_s, index, peak_t

        processed_class_ds = (
            class_index_ds.shuffle(buffer_size=len(class_indices))
            .repeat()
            .map(gather_and_process, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(n_to_sample)
        )
        datasets_by_class.append(processed_class_ds)

    if not datasets_by_class:
        raise ValueError("Batch size is smaller than the number of classes.")

    stratified_dataset = tf.data.Dataset.zip(tuple(datasets_by_class))

    @tf.function
    def concatenate_batches(*batches):
        all_x = tf.concat([b[0] for b in batches], axis=0)
        all_y = tf.concat([b[1] for b in batches], axis=0)
        all_indices = tf.concat([b[2] for b in batches], axis=0)
        all_peak_t = tf.concat([b[3] for b in batches], axis=0)
        return all_x, all_y, all_indices, all_peak_t

    final_dataset = stratified_dataset.map(
        concatenate_batches, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)
    return final_dataset


@tf.function(jit_compile=True)
def get_context_set_dense(target_x, target_y, num_context, seed=None):
    batch_size = ops.shape(target_x)[0]
    num_points = ops.shape(target_x)[1]

    indices = tf.random.uniform(
        shape=(batch_size, num_context), maxval=num_points, dtype=tf.int32, seed=seed
    )

    # keras.ops doesn't seem to have batch_dims ??
    context_x = tf.gather(target_x, indices, batch_dims=1)
    context_y = tf.gather(target_y, indices, batch_dims=1)

    return context_x, context_y


@tf.function(jit_compile=True)
def get_context_set_dense_forecast(
    target_x, target_y, num_context, peak_t, forecast_prob, seed=None
):
    batch_size = tf.shape(target_x)[0]
    num_points = tf.shape(target_x)[1]

    times = target_x[..., 0]
    sorted_idx = tf.argsort(times, axis=1)
    sorted_times = tf.gather(times, sorted_idx, batch_dims=1)

    pre_peak_count = tf.reduce_sum(
        tf.cast(sorted_times <= peak_t[:, None], tf.int32), axis=1
    )
    pool_size = tf.where(
        pre_peak_count > 0,
        pre_peak_count,
        tf.fill([batch_size], num_context),
    )
    pool_size = tf.minimum(pool_size, num_points)

    init_indices = tf.random.uniform(
        shape=(batch_size, num_context),
        maxval=num_points,
        dtype=tf.int32,
        seed=seed,
    )
    init_times = tf.gather(times, init_indices, batch_dims=1)
    is_post_peak = init_times > peak_t[:, None]

    refill_rand = tf.random.uniform(
        shape=(batch_size, num_context), seed=seed, dtype=tf.float32
    )
    pool_pos = tf.cast(
        refill_rand * tf.cast(pool_size, tf.float32)[:, None], tf.int32
    )
    refill_idx = tf.gather(sorted_idx, pool_pos, batch_dims=1)
    final_idx_forecast = tf.where(is_post_peak, refill_idx, init_indices)

    is_forecast = tf.random.uniform(shape=(batch_size,), seed=seed) < forecast_prob
    final_idx = tf.where(is_forecast[:, None], final_idx_forecast, init_indices)

    context_x = tf.gather(target_x, final_idx, batch_dims=1)
    context_y = tf.gather(target_y, final_idx, batch_dims=1)
    return context_x, context_y


def get_gplike_valset(
    X_val,
    y_val,
    num_context,
    objnum=0,
    seed=None,
):
    X_val_obj = ops.expand_dims(X_val[objnum], 0)
    y_val_obj = ops.expand_dims(y_val[objnum], 0)

    target_x_val, target_y_val = X_val_obj, y_val_obj

    context_x_val, context_y_val = get_context_set_dense(
        target_x_val, target_y_val, num_context, seed=seed
    )

    return context_x_val, context_y_val, target_x_val, target_y_val


def unscale_values(values_norm, scaler):
    if hasattr(values_norm, "numpy"):
        values_norm = values_norm.numpy()
    values_norm = np.asarray(values_norm)
    original_shape = values_norm.shape
    flattened = values_norm.flatten().reshape(-1, 1)
    unscaled = scaler.inverse_transform(flattened)
    return unscaled.reshape(original_shape)
