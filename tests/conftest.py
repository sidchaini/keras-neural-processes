import numpy as np
import pytest


# --- Shared data fixtures for tests ---


@pytest.fixture
def sample_gp_data():
    """Generate sample GP-like single-channel data."""
    batch_size = 3
    num_points = 30
    x_dim = 1
    y_dim = 1

    x = np.linspace(-2, 2, num_points).reshape(1, num_points, x_dim)
    x = np.repeat(x, batch_size, axis=0)
    y = np.sin(x * np.pi) + 0.1 * np.random.randn(batch_size, num_points, y_dim)

    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def multi_channel_gp_data():
    """Generate multi-channel GP-like data in long format."""
    batch_size = 2
    num_points_per_channel = 20
    num_channels = 3
    total_points = num_points_per_channel * num_channels

    x = np.zeros((batch_size, total_points, 2))
    y = np.zeros((batch_size, total_points, 1))

    for batch_idx in range(batch_size):
        for ch in range(num_channels):
            start_idx = ch * num_points_per_channel
            end_idx = (ch + 1) * num_points_per_channel

            time_points = np.linspace(-2, 2, num_points_per_channel)
            x[batch_idx, start_idx:end_idx, 0] = time_points
            x[batch_idx, start_idx:end_idx, 1] = ch

            if ch == 0:
                y[batch_idx, start_idx:end_idx, 0] = np.sin(time_points * np.pi)
            elif ch == 1:
                y[batch_idx, start_idx:end_idx, 0] = np.cos(time_points * np.pi)
            else:
                y[batch_idx, start_idx:end_idx, 0] = time_points**2

    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def sample_data_1d():
    """Generate sample 1D function data for testing (broader batch)."""
    batch_size = 4
    num_points = 50
    x_dim = 1
    y_dim = 1

    x = np.linspace(-2, 2, num_points).reshape(1, num_points, x_dim)
    x = np.repeat(x, batch_size, axis=0)
    y = np.sin(x * np.pi) + 0.1 * np.random.randn(batch_size, num_points, y_dim)

    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def sample_data_multi_channel():
    """Generate sample multi-channel data for testing (2 channels)."""
    batch_size = 4
    num_points_per_channel = 25
    num_channels = 2
    total_points = num_points_per_channel * num_channels

    x = np.zeros((batch_size, total_points, 2))
    y = np.zeros((batch_size, total_points, 1))

    for batch_idx in range(batch_size):
        for ch in range(num_channels):
            start_idx = ch * num_points_per_channel
            end_idx = (ch + 1) * num_points_per_channel

            time_points = np.linspace(-2, 2, num_points_per_channel)
            x[batch_idx, start_idx:end_idx, 0] = time_points
            x[batch_idx, start_idx:end_idx, 1] = ch

            if ch == 0:
                y[batch_idx, start_idx:end_idx, 0] = np.sin(time_points * np.pi)
            else:
                y[batch_idx, start_idx:end_idx, 0] = np.cos(time_points * np.pi)

    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def context_target_data(sample_data_1d):
    """Generate context and target sets from sample data."""
    x, y = sample_data_1d
    num_points = x.shape[1]

    num_context = 10
    context_indices = np.random.choice(num_points, num_context, replace=False)

    context_x = x[:, context_indices, :]
    context_y = y[:, context_indices, :]
    target_x = x
    target_y = y

    return context_x, context_y, target_x, target_y


@pytest.fixture
def training_data():
    """Generate larger training data."""
    batch_size = 8
    num_points = 40
    x_dim = 1
    y_dim = 1

    x = np.linspace(-2, 2, num_points).reshape(1, num_points, x_dim)
    x = np.repeat(x, batch_size, axis=0)
    y = np.sin(x * np.pi) + 0.1 * np.random.randn(batch_size, num_points, y_dim)

    return x.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def small_training_data():
    """Generate smaller training data for faster unit tests."""
    batch_size = 4
    num_points = 20
    x_dim = 1
    y_dim = 1

    x = np.linspace(-1, 1, num_points).reshape(1, num_points, x_dim)
    x = np.repeat(x, batch_size, axis=0)
    y = np.sin(x * np.pi * 2) + 0.05 * np.random.randn(batch_size, num_points, y_dim)

    return x.astype(np.float32), y.astype(np.float32)
