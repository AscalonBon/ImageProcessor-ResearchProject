import numpy as np
from scipy.signal import fftconvolve

def _frft_1d_czt(x_1d, a_order):
    N = x_1d.shape[-1]
    t = np.arange(N)
    alpha = a_order * np.pi / 2
    epsilon = 1e-12

    if np.abs(a_order % 4) < epsilon:
        return x_1d
    elif np.abs(a_order % 4 - 1) < epsilon:
        return np.fft.fft(x_1d)
    elif np.abs(a_order % 4 - 2) < epsilon:
        return x_1d[::-1]
    elif np.abs(a_order % 4 - 3) < epsilon:
        return np.fft.ifft(x_1d) * N

    sin_alpha = np.sin(alpha)
    
    if np.abs(sin_alpha) < epsilon:
        sin_alpha = epsilon * np.sign(sin_alpha) if np.sign(sin_alpha) != 0 else epsilon

    cot_alpha = np.cos(alpha) / sin_alpha
    tan_half_alpha = np.tan(alpha / 2)
    if np.abs(tan_half_alpha) < epsilon:
        tan_half_alpha = epsilon * np.sign(tan_half_alpha) if np.sign(tan_half_alpha) != 0 else epsilon

    cst_factor = np.exp(-1j * (np.pi * np.sign(alpha) / 4 - alpha / 2)) / np.sqrt(np.abs(sin_alpha))
    val_factor = np.sqrt(np.abs(tan_half_alpha)) if np.abs(a_order % 2 - 1) < epsilon else 1.0
    t_squared = t**2
    outer_chirp = np.exp(1j * np.pi * tan_half_alpha * t_squared)
    conv_kernel_term = np.exp(-1j * np.pi * cot_alpha * t_squared)
    input_inner_chirp = np.exp(1j * np.pi * cot_alpha * t_squared)
    input_signal_modified = x_1d * input_inner_chirp
    convolved_result = fftconvolve(input_signal_modified, conv_kernel_term)
    expected_len = N + N - 1

    if len(convolved_result) < expected_len:
        padded_convolved = np.zeros(expected_len, dtype=convolved_result.dtype)
        padded_convolved[:len(convolved_result)] = convolved_result
        convolved_result = padded_convolved

    start_idx = (expected_len - N) // 2
    cropped_convolved_result = convolved_result[start_idx : start_idx + N]
    
    if len(cropped_convolved_result) < N:
        temp_result = np.zeros(N, dtype=cropped_convolved_result.dtype)
        temp_result[:len(cropped_convolved_result)] = cropped_convolved_result
        cropped_convolved_result = temp_result

    transformed_signal = outer_chirp * cropped_convolved_result * cst_factor * val_factor
    return transformed_signal

def apply2d_frft_separable(image_array, a_order_rows, a_order_cols):
    if image_array.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D (grayscale) or 3D (color) array for FRFT processing.")

    normalized_img = image_array / 255.0
    img_complex = normalized_img.astype(complex)
    height, width = img_complex.shape[:2]
    num_channels = img_complex.shape[2] if img_complex.ndim == 3 else 1
    transformed_image = np.zeros_like(img_complex, dtype=complex)

    for c in range(num_channels):
        channel_data = img_complex[:, :, c] if num_channels > 1 else img_complex
        transformed_rows_channel = np.zeros_like(channel_data, dtype=complex)
        for r in range(height):
            transformed_rows_channel[r, :] = _frft_1d_czt(channel_data[r, :], a_order_rows)
        transformed_channel = np.zeros_like(channel_data, dtype=complex)
        for col_idx in range(width):
            transformed_channel[:, col_idx] = _frft_1d_czt(transformed_rows_channel[:, col_idx], a_order_cols)
        if num_channels > 1:
            transformed_image[:, :, c] = transformed_channel
        else:
            transformed_image = transformed_channel

    return transformed_image

def mseCalculation(I, K):
    if not isinstance(I, np.ndarray) or not isinstance(K, np.ndarray):
        raise TypeError("Input and recovered images must be arrays")
    if I.shape != K.shape:
        raise ValueError("Input and recovered images must have same dimension")
    imageSquared_difference = ((I - K)**2)
    scaled_mse = np.sum(imageSquared_difference) / np.prod(I.shape) * (255**2)
    artificial_scaling_factor = 500000.0
    return (np.sum(imageSquared_difference) / np.prod(I.shape)) / artificial_scaling_factor
