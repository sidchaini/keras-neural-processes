import sys
import numpy as np
import numpy.typing as npt
import joblib
import pandas as pd
import gc
import polars as pl
from tqdm.auto import tqdm
import tensorflow as tf
from keras import ops
from sklearn.preprocessing import MinMaxScaler
from .CONSTANTS import seed_val


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
        # here values are flat, and lengths are provided
    return arr_ragged


class LightCurvePreprocessor:
    def __init__(
        self,
        time_scaler=None,
        flux_scaler=None,
        fltnum_scaler=None,
        FLUX_MAX=1000,
        FLTNUM_MAX=5,
    ):
        # use sklearn scalers for management but polars expr. for speed
        self.time_scaler = time_scaler if time_scaler else MinMaxScaler()
        self.fltnum_scaler = (
            fltnum_scaler
            if fltnum_scaler
            else MinMaxScaler(feature_range=(0, FLTNUM_MAX))
        )
        self.flux_scaler = (
            flux_scaler if flux_scaler else MinMaxScaler(feature_range=(0, FLUX_MAX))
        )
        self.is_fitted = False

    def fit(self, lc_lazy):
        if self.is_fitted:
            print("WARNING: Overwriting previous scaler fits", file=sys.stderr)

        if isinstance(lc_lazy, pl.DataFrame):
            lc_lazy = lc_lazy.lazy()

        stats = lc_lazy.select(
            [
                pl.col("mjd").list.min().min().alias("mjd_min"),
                pl.col("mjd").list.max().max().alias("mjd_max"),
                pl.col("fltnum").list.min().min().alias("fltnum_min"),
                pl.col("fltnum").list.max().max().alias("fltnum_max"),
                pl.col("flux").list.min().min().alias("flux_min"),
                pl.col("flux").list.max().max().alias("flux_max"),
            ]
        ).collect()

        self.time_scaler.fit([[stats["mjd_min"][0]], [stats["mjd_max"][0]]])
        self.fltnum_scaler.fit([[stats["fltnum_min"][0]], [stats["fltnum_max"][0]]])
        self.flux_scaler.fit([[stats["flux_min"][0]], [stats["flux_max"][0]]])
        self.is_fitted = True

    def transform(self, phot_df):
        def scale_polars(col_name, scaler):
            if isinstance(scaler, MinMaxScaler):
                data_min = scaler.data_min_[0]
                data_max = scaler.data_max_[0]
                feature_min, feature_max = scaler.feature_range

                scale = (feature_max - feature_min) / (data_max - data_min)
                min_val = feature_min - data_min * scale

                return (
                    pl.col(col_name).cast(pl.List(pl.Float32)) * scale + min_val
                ).cast(pl.List(pl.Float32))

            else:
                raise NotImplementedError("Only MinMaxScaler supported currently")

        return phot_df.with_columns(
            [
                scale_polars("mjd", self.time_scaler),
                scale_polars("fltnum", self.fltnum_scaler).cast(pl.List(pl.UInt8)),
                scale_polars("flux", self.flux_scaler),
            ]
        )

    def inverse_transform(self, phot_df):
        # to invert transform, take flux and flux err, calculate upper and lower bounds,
        # then invert transform these bounds rather than the error
        # then, take bounds, and get error as half of distance between them.
        # doing this b/c not obvious to transform error directly
        def inverse_scale_polars(col_name, scaler):
            if isinstance(scaler, MinMaxScaler):
                data_min = scaler.data_min_[0]
                data_max = scaler.data_max_[0]
                feature_min, feature_max = scaler.feature_range

                scale = (feature_max - feature_min) / (data_max - data_min)
                min_val = feature_min - data_min * scale

                return (
                    (pl.col(col_name).cast(pl.List(pl.Float32)) - min_val) / scale
                ).cast(pl.List(pl.Float32))

            else:
                raise NotImplementedError("Only MinMaxScaler supported currently")

        # make pl expression for inverse transform
        inverted_expr = [
            inverse_scale_polars("mjd", self.time_scaler),
            inverse_scale_polars("fltnum", self.fltnum_scaler).cast(pl.List(pl.UInt8)),
            inverse_scale_polars("flux", self.flux_scaler),
        ]

        if "flux_err" in phot_df.columns:
            # get upper and lower bounds
            phot_df = phot_df.select(
                pl.all(),
                (pl.col("flux") + pl.col("flux_err")).alias("flux_upper"),
                (pl.col("flux") - pl.col("flux_err")).alias("flux_lower"),
            ).drop("flux_err")

            # add pl expression for inverse-transforming the bounds
            inverted_expr = inverted_expr + [
                inverse_scale_polars("flux_upper", self.flux_scaler).cast(
                    pl.List(pl.Float32)
                ),
                inverse_scale_polars("flux_lower", self.flux_scaler).cast(
                    pl.List(pl.Float32)
                ),
            ]

        # apply pl expressions to light curves
        phot_df = phot_df.with_columns(inverted_expr)

        # calculate standalone error from bounds after transformation
        if "flux_upper" in phot_df.columns and "flux_lower" in phot_df.columns:
            phot_df = phot_df.select(
                pl.all().exclude(["flux_upper", "flux_lower"]),
                (
                    (pl.col("flux_upper") - pl.col("flux_lower"))
                    / pl.lit(2, dtype=pl.Float32)
                ).alias("flux_err"),
                # removed .cast(pl.List(pl.Float32)) at the end
            )
        return phot_df

    def process(self, lc_lazy):
        # def process(self, lc_lazy, apply_transform=True):
        #     if apply_transform:
        #         # default is true
        #         # b/c double transformation changes nothing
        #         lc_lazy = self.transform(
        #             lc_lazy
        #         )  # .sort("objid") # should already maintain order

        df = lc_lazy.select(
            [pl.col("mjd"), pl.col("fltnum").cast(pl.List(pl.Float32)), pl.col("flux")]
        ).collect()

        mjd_arrow = df.drop_in_place("mjd").to_arrow()
        flt_arrow = df.drop_in_place("fltnum").to_arrow()
        splits = mjd_arrow.offsets

        X_ragged = np.stack(
            [
                mjd_arrow.values.to_numpy(),
                flt_arrow.values.to_numpy(),
            ],
            axis=1,
        )
        del (mjd_arrow, flt_arrow)

        X_ragged = tf.RaggedTensor.from_row_splits(X_ragged, splits)

        y_ragged = df.drop_in_place("flux").to_arrow().values.to_numpy().reshape(-1, 1)
        del df
        y_ragged = tf.RaggedTensor.from_row_splits(y_ragged, splits)

        return X_ragged, y_ragged

    def get_lightcurves(self, X, y, objids, yerr=None, inverse_transform=True):
        if not self.is_fitted:
            raise ValueError("Processor not fitted! Cannot unscale.")

        X = get_ragged_tensor(X) if not isinstance(X, tf.RaggedTensor) else X
        y = get_ragged_tensor(y) if not isinstance(y, tf.RaggedTensor) else y
        if yerr is not None and not isinstance(yerr, tf.RaggedTensor):
            yerr = get_ragged_tensor(yerr)

        row_lengths = X.row_lengths().numpy()
        if len(objids) != len(row_lengths):
            raise ValueError(
                f"Shape mismatch: {len(objids)} objids vs {len(row_lengths)} rows"
            )

        mjd = X[..., 0].flat_values.numpy()
        fltnum = X[..., 1].flat_values.numpy()
        del X
        flux = y[..., 0].flat_values.numpy()
        del y
        flux_err = yerr[..., 0].flat_values.numpy() if yerr is not None else None
        del yerr
        gc.collect()

        phot_df = {
            "objid": pl.Series(np.repeat(objids, row_lengths)).cast(pl.UInt32),
            "mjd": pl.Series(mjd).cast(pl.Float32),
            "fltnum": pl.Series(fltnum).cast(pl.UInt8),
            "flux": pl.Series(flux).cast(pl.Float32),
        }
        if flux_err is not None:
            phot_df["flux_err"] = pl.Series(flux_err).cast(pl.Float32)

        phot_df = (
            pl.DataFrame(phot_df)
            .group_by("objid", maintain_order=True)
            .agg(pl.exclude("objid"))
        )

        if inverse_transform:
            phot_df = self.inverse_transform(phot_df)

        return phot_df

    def save(self, path):
        joblib.dump(
            {
                "time": self.time_scaler,
                "fltnum": self.fltnum_scaler,
                "flux": self.flux_scaler,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        self.time_scaler = data["time"]
        self.fltnum_scaler = data["fltnum"]
        self.flux_scaler = data["flux"]
        self.is_fitted = data["is_fitted"]


@tf.function
def _slice_and_normalize_target_set(x, y, num_target_points, slice_width):
    current_len = tf.shape(x)[0]

    # 1. Take a slice of the light curve
    effective_slice_width = tf.minimum(current_len, slice_width)
    max_start_index = current_len - effective_slice_width
    start_index = tf.cond(
        max_start_index > 0,
        lambda: tf.random.uniform(shape=(), maxval=max_start_index + 1, dtype=tf.int32),
        lambda: 0,
    )
    x_slice = x[start_index : start_index + effective_slice_width]
    y_slice = y[start_index : start_index + effective_slice_width]

    # 2. Resample slice to some fixed size (fixed is faster for training)
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

    # 3. Starting time should be zero for the slice
    time_col = resampled_x[..., 0:1]
    filter_col = resampled_x[..., 1:]

    centered_time = time_col - tf.reduce_min(time_col, axis=0, keepdims=True)

    final_target_x = tf.concat([centered_time, filter_col], axis=-1)

    return final_target_x, resampled_y


def create_stratified_np_dataset(
    X_ragged,
    y_ragged,
    labels,
    batch_size,
    num_target_points,
    slice_width,
):
    df = pd.DataFrame({"label": ops.convert_to_numpy(labels)})
    grouped_indices = df.groupby("label").groups
    unique_classes = list(grouped_indices.keys())
    num_classes = len(unique_classes)
    samples_per_class = batch_size // num_classes
    remainder = batch_size % num_classes

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
            x_s, y_s = _slice_and_normalize_target_set(
                x, y, num_target_points, slice_width
            )
            x_s.set_shape([num_target_points, x.shape[-1]])
            y_s.set_shape([num_target_points, y.shape[-1]])
            return x_s, y_s, index

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
        return all_x, all_y, all_indices

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


def get_gplike_valset(
    X_val,
    y_val,
    num_context,
    objnum=0,
    seed=None,
):
    # plot one so just convert it to a dense tensor
    X_val_obj = ops.expand_dims(X_val[objnum], 0)
    y_val_obj = ops.expand_dims(y_val[objnum], 0)

    target_x_val, target_y_val = X_val_obj, y_val_obj

    context_x_val, context_y_val = get_context_set_dense(
        target_x_val, target_y_val, num_context, seed=seed
    )

    return context_x_val, context_y_val, target_x_val, target_y_val


def unscale_values(values_norm, scaler):
    # Go from normalised scale to original scale
    if hasattr(values_norm, "numpy"):
        values_norm = values_norm.numpy()
    values_norm = np.asarray(values_norm)
    original_shape = values_norm.shape
    flattened = values_norm.flatten().reshape(-1, 1)
    unscaled = scaler.inverse_transform(flattened)
    return unscaled.reshape(original_shape)


def prepare_phys_arrays(x, y, time_scaler, flux_scaler):
    # prepare physical unit arrays
    x_phys = (np.array(x) if not hasattr(x, "numpy") else x.numpy()).copy()
    y_phys = (np.array(y) if not hasattr(y, "numpy") else y.numpy()).copy()

    x_phys[:, 0] = unscale_values(x_phys[:, 0], time_scaler)
    y_phys[:, 0] = unscale_values(y_phys[:, 0], flux_scaler)
    return x_phys, y_phys


def get_phys_updown_errs(y, y_err):
    # get bounds from errors
    y = (np.array(y) if not hasattr(y, "numpy") else y.numpy()).copy()
    y_err = (np.array(y_err) if not hasattr(y_err, "numpy") else y_err.numpy()).copy()

    y_phys_down_bound = (y[:, 0] - y_err[:, 0]).reshape(y.shape)
    y_phys_up_bound = (y[:, 0] + y_err[:, 0]).reshape(y.shape)

    return y_phys_down_bound, y_phys_up_bound


def get_context_indices(
    full_mjd,
    peak_mjd,
    num_points,
    strategy="random",
    prepeak_maskcutoff=50,  # 100,
    postpeak_maskcutoff=75,  # 150
    seed=seed_val,
):
    rng = np.random.default_rng(seed)
    total_points = len(full_mjd)
    indices = np.arange(total_points)

    if total_points == 0:
        return np.zeros(num_points, dtype=int)

    chosen_idx = []

    # 1. choose pts as per strategy
    if strategy == "random":
        chosen_idx = rng.choice(
            indices, size=min(num_points, total_points), replace=False
        )

    elif strategy == "peak_focused":
        # Priority: +/- 2 days of peak
        peak_mask = (full_mjd >= peak_mjd - 2) & (full_mjd <= peak_mjd + 2)
        peak_indices = indices[peak_mask]
        # Force 1 peak point if available
        if len(peak_indices) > 0:
            p_idx = rng.choice(peak_indices, 1)
            chosen_idx.append(p_idx[0])

        # Secondary Pool: Broad window (Defined by cutoffs)
        # Fill up to num_points using this window first
        remaining = num_points - len(chosen_idx)
        if remaining > 0:
            # LAZY CALCULATION: Only compute broad mask if we actually need more points
            broad_mask = (full_mjd >= peak_mjd - prepeak_maskcutoff) & (
                full_mjd <= peak_mjd + postpeak_maskcutoff
            )
            broad_indices = indices[broad_mask]

            # Exclude already chosen (the peak point) to avoid duplicates
            broad_indices = np.setdiff1d(broad_indices, chosen_idx)

            if len(broad_indices) > 0:
                count = min(remaining, len(broad_indices))
                chosen_idx.extend(rng.choice(broad_indices, count, replace=False))

    elif strategy == "extrapolation":
        # Strict Pre-Peak only
        pre_peak_mask = full_mjd < peak_mjd
        pre_peak_indices = indices[pre_peak_mask]

        count = min(num_points, len(pre_peak_indices))
        if count > 0:
            chosen_idx.extend(rng.choice(pre_peak_indices, count, replace=False))

    # Convert to numpy array for set operations
    chosen_idx = np.array(chosen_idx, dtype=int)

    # 2. If not enough pts from strategy, sample randomly from ANYWHERE else in the light curve
    if len(chosen_idx) < num_points:
        num_needed = num_points - len(chosen_idx)
        available_indices = np.setdiff1d(indices, chosen_idx)
        if len(available_indices) > 0:
            count = min(num_needed, len(available_indices))
            fillers = rng.choice(available_indices, count, replace=False)
            chosen_idx = np.concatenate([chosen_idx, fillers])

    # 3. Oversample only if the object itself
    # is extremely short (shorter than num_points)
    if len(chosen_idx) < num_points:
        num_needed = num_points - len(chosen_idx)
        padding = rng.choice(indices, size=num_needed, replace=True)
        chosen_idx = np.concatenate([chosen_idx, padding])

    return np.sort(chosen_idx).astype(int)


def create_evaluation_batch(
    X_ragged,
    y_ragged,
    lc_lazy,
    num_points,
    strategy,
    time_scaler,
    seed=seed_val,
    pbar=True,
):
    context_x_list, context_y_list = [], []

    peak_rels_lazy = (
        lc_lazy.select(pl.col("objid"), pl.col("mjd").list.min().alias("mjd_min"))
        .sort("objid")
        .collect()
        .iter_rows()
    )

    iterator = enumerate(peak_rels_lazy)
    iterator = (
        tqdm(
            iterator, total=X_ragged.shape[0], leave=False, desc=f"{strategy} class no."
        )
        if pbar
        else iterator
    )

    splits = X_ragged.row_splits.numpy()
    assert (splits == y_ragged.row_splits.numpy()).all()

    for i, (snid, peak_rel) in iterator:
        x_norm = X_ragged._values[splits[i] : splits[i + 1]].numpy()
        y_norm = y_ragged._values[splits[i] : splits[i + 1]].numpy()
        mjd_rel = unscale_values(x_norm[:, 0], time_scaler).flatten()
        # assert np.isclose(mjd_rel, lc_train_og[i].select("mjd").item().to_numpy()).all()

        indices = get_context_indices(mjd_rel, peak_rel, num_points, strategy, seed)

        context_x_list.append(x_norm[indices])
        context_y_list.append(y_norm[indices])

    return np.array(context_x_list, dtype=np.float32), np.array(
        context_y_list, dtype=np.float32
    )


# Code taken from tdastro: https://github.com/lincc-frameworks/tdastro/
# Thanks to the LINCC Frameworks team!!!

# AB definition is zp=8.9 for 1 Jy
MAG_AB_ZP_NJY = 8.9 + 2.5 * 9


def mag2flux(mag: npt.ArrayLike) -> npt.ArrayLike:
    """Convert AB magnitude to bandflux in nJy

    Parameters
    ----------
    mag : ndarray of float
        The magnitude to convert to bandflux.

    Returns
    -------
    bandflux : ndarray of float
        The bandflux corresponding to the input magnitude.
    """
    return np.power(10.0, -0.4 * (mag - MAG_AB_ZP_NJY))


def flux2mag(flux: npt.ArrayLike) -> npt.ArrayLike:
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(flux) + MAG_AB_ZP_NJY
    # mag[flux <= 0] = np.nan
    return mag
