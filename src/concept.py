import numpy as np

# Function to calculate a single element of the Fractional Fourier Transform (FRFT) kernel
def frftKernel(x, u, phi):
    """
    Calculates a single element of the Fractional Fourier Transform (FRFT) kernel.

    Args:
        x (float or np.ndarray): Input coordinate(s).
        u (float or np.ndarray): Output coordinate(s).
        phi (float): The fractional order angle (a * pi / 2).

    Returns:
        complex or np.ndarray: The kernel value(s).
    """
    sgn_phi = np.sign(phi)
    # Calculate the scaling factor tau.
    # np.sin(phi) can be zero if phi is a multiple of pi, which leads to division by zero.
    # For common FRFT orders, phi is often not a multiple of pi, but for robust
    # implementations, special handling for these cases (e.g., close to DFT, identity, or flip)
    # would be necessary. For this exercise, we assume phi is not exactly a multiple of pi.
    tau = np.exp(-1j * (np.pi * sgn_phi / 4 - phi / 2)) / np.sqrt(np.abs(np.sin(phi)))

    # Main formula for the FRFT kernel
    # 1/np.tan(phi) can also lead to issues if phi is a multiple of pi/2 (but not pi).
    # This corresponds to standard Fourier Transform (phi = pi/2) where tan is undefined.
    # These cases are generally handled by specific algorithms or approximations.
    kernel = tau * np.exp(
        1j * np.pi * (
            x**2 * (1 / np.tan(phi)) -
            2 * x * u / np.sin(phi) +
            u**2 * (1 / np.tan(phi))
        )
    )
    return kernel

# Function to generate the full FRFT kernel matrix for a given size and order
def generate_frftKernel(size, phi):
    """
    Generates a 1D FRFT kernel matrix of a given size.
    The returned matrix K will be such that output_vector = K @ input_vector.
    Therefore, K[u, x] where u is output index (row) and x is input index (column).

    Args:
        size (int): The dimension of the kernel (N x N).
        phi (float): The fractional order angle.

    Returns:
        np.ndarray: The N x N complex FRFT kernel matrix.
    """
    # 1D indices for input (x) and output (u) coordinates
    # Both range from 0 to size-1
    input_coords = np.arange(size)
    output_coords = np.arange(size)

    # Create 2D meshgrids. For K[u,x], U_mesh should represent 'u' (output) values (rows)
    # and X_mesh should represent 'x' (input) values (columns).
    # With indexing='ij', output_mesh[i,j] = output_coords[i], input_mesh[i,j] = input_coords[j].
    U_mesh, X_mesh = np.meshgrid(output_coords, input_coords, indexing='ij')

    # Calculate the kernel matrix using the frftKernel function
    # frftKernel(x, u, phi) where x is the input coordinate and u is the output coordinate.
    kernel_matrix = frftKernel(X_mesh, U_mesh, phi)
    return kernel_matrix

# Function to apply the 2D Fractional Fourier Transform (separable)
def apply2d_frft_separable(image_array, a_order_rows, a_order_cols):
    """
    Applies the 2D Separable Fractional Fourier Transform to a grayscale image.
    This implementation performs the FRFT first along rows with `a_order_rows`,
    then along columns with `a_order_cols`.

    Args:
        image_array (np.ndarray): The 2D grayscale input image (real or complex values).
        a_order_rows (float): The fractional order 'a' for the row-wise transform.
        a_order_cols (float): The fractional order 'a' for the column-wise transform.

    Returns:
        np.ndarray: The 2D complex-valued transformed image.
    """
    if image_array.ndim != 2:
        raise ValueError("Input image must be a 2D (grayscale) array for FRFT processing.")

    # Calculate the angles phi for row and column transforms
    phi_rows = a_order_rows * np.pi / 2
    phi_cols = a_order_cols * np.pi / 2

    # Ensure the image data is complex for FRFT operations
    img_complex = image_array.astype(complex)

    height, width = image_array.shape

    # Phase 1: Apply FRFT along rows (width dimension)
    # Each row is treated as a 1D signal. The kernel for this transform will be 'width x width'.
    kernel_width = generate_frftKernel(width, phi_rows)

    # Initialize array to store results after row-wise transform
    transformed_rows = np.zeros_like(img_complex, dtype=complex)
    # Apply the kernel to each row. The matrix multiplication `kernel @ vector`
    # applies the transform to the row vector.
    for r in range(height):
        transformed_rows[r, :] = kernel_width @ img_complex[r, :]

    # Phase 2: Apply FRFT along columns (height dimension)
    # Each column of `transformed_rows` is treated as a 1D signal. The kernel for this
    # transform will be 'height x height'.
    kernel_height = generate_frftKernel(height, phi_cols)

    # Initialize array to store final 2D transformed image
    transformed_image = np.zeros_like(img_complex, dtype=complex)
    # Apply the kernel to each column.
    for c in range(width):
        transformed_image[:, c] = kernel_height @ transformed_rows[:, c]

    return transformed_image

# Function to calculate the Mean Squared Error (MSE)
def mseCalculation(I, K):
    """
    Calculates the Mean Squared Error (MSE) between an input image (ground truth)
    and a recovered or compared image.

    Args:
        I (np.ndarray): The input image (ground truth, or original).
        K (np.ndarray): The recovered or compared image (e.g., transformed and then inverse transformed).

    Returns:
        float: The calculated MSE value.

    Raises:
        TypeError: If I or K are not numpy arrays.
        ValueError: If I and K have different dimensions.
    """
    if not isinstance(I, np.ndarray) or not isinstance(K, np.ndarray):
        raise TypeError("Input and recovered images must be arrays")

    if I.shape != K.shape:
        raise ValueError("Input and recovered images must have same dimension")

    M, N = I.shape # Get dimensions of the image (rows, columns)

    # Calculate the squared difference between the two images element-wise
    imageSquared_difference = (I - K)**2

    # Calculate MSE using the formula: MSE = sum(squared_difference) / (M*N)
    MSE = np.sum(imageSquared_difference) / (M * N)
    return MSE

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Fractional Fourier Transform and MSE Calculation ---")

    # Sample input image (K_original_image) for transformations and MSE calculation.
    # This array is provided by the user.
    K_original_image = np.array([
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55],
        [12, 22, 32, 42, 52],
        [18, 28, 38, 48, 58],
        [11, 21, 31, 41, 51]
    ], dtype=np.float64)

    # --- Table 1: Computation of mean square error by changing the order of phase 1 ---
    print("\n" + "="*80)
    print("Table 1: Computation of mean square error by changing the order of phase 1")
    print(" (Phase 2 is constant at -0.25)")
    print("="*80)
    print(f"{'Sr. No.':<10} {'Fractional Order':<30} {'Mean Square Error':<25}")
    print(f"{'':<10} {'Phase 1':<15} {'Phase 2':<15}")
    print("-" * 80)

    # Fractional orders for Phase 1 (from user's table image)
    phase1_orders_table1 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    constant_phase2_order = -0.25 # Constant order for Phase 2 (from user's table image)

    results_table1 = []
    for i, a1_order in enumerate(phase1_orders_table1):
        try:
            # 1. Perform Forward FRFT
            # Apply FRFT to rows with a1_order, then to columns with constant_phase2_order
            transformed_img_forward = apply2d_frft_separable(K_original_image, a1_order, constant_phase2_order)

            # 2. Perform Inverse FRFT for Reconstruction
            # For separable inverse FRFT, negate orders and swap application order:
            # First, inverse transform columns with -constant_phase2_order,
            # then inverse transform rows with -a1_order.
            reconstructed_img = apply2d_frft_separable(transformed_img_forward, -constant_phase2_order, -a1_order)

            # 3. Calculate MSE between original and reconstructed image
            # Taking the real part as the original image is real and reconstructed
            # might have small imaginary components due to numerical precision.
            mse_val = mseCalculation(K_original_image, np.real(reconstructed_img))

            results_table1.append((i + 1, a1_order, constant_phase2_order, mse_val))
        except (TypeError, ValueError, np.linalg.LinAlgError) as e:
            # Catch potential errors like division by zero if phi is a multiple of pi,
            # or issues with numerical stability.
            results_table1.append((i + 1, a1_order, constant_phase2_order, f"Error: {e}"))

    # Print results for Table 1
    for sr_no, p1_order, p2_order, mse in results_table1:
        # Format MSE value to 4 decimal places, or print error string if applicable
        mse_str = f"{mse:.4f}" if isinstance(mse, (float, np.float64)) else str(mse)
        print(f"{sr_no:<10} {p1_order:<15.2f} {p2_order:<15.2f} {mse_str:<25}")
    print("="*80)


    # --- Table 2: Computation of mean square error by changing the order of phase 2 ---
    print("\n" + "="*80)
    print("Table 2: Computation of mean square error by changing the order of phase 2")
    print(" (Phase 1 is constant at 0.25)")
    print("="*80)
    print(f"{'Sr. No.':<10} {'Fractional Order':<30} {'Mean Square Error':<25}")
    print(f"{'':<10} {'Phase 1':<15} {'Phase 2':<15}")
    print("-" * 80)

    constant_phase1_order = 0.25 # Constant order for Phase 1 (from user's table image)
    # Fractional orders for Phase 2 (from user's table image)
    phase2_orders_table2 = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90]

    results_table2 = []
    for i, a2_order in enumerate(phase2_orders_table2):
        try:
            # 1. Perform Forward FRFT
            # Apply FRFT to rows with constant_phase1_order, then to columns with a2_order
            transformed_img_forward = apply2d_frft_separable(K_original_image, constant_phase1_order, a2_order)

            # 2. Perform Inverse FRFT for Reconstruction
            # Inverse: negate orders and swap application order
            reconstructed_img = apply2d_frft_separable(transformed_img_forward, -a2_order, -constant_phase1_order)

            # 3. Calculate MSE between original and reconstructed image
            mse_val = mseCalculation(K_original_image, np.real(reconstructed_img))

            results_table2.append((i + 1, constant_phase1_order, a2_order, mse_val))
        except (TypeError, ValueError, np.linalg.LinAlgError) as e:
            results_table2.append((i + 1, constant_phase1_order, a2_order, f"Error: {e}"))

    # Print results for Table 2
    for sr_no, p1_order, p2_order, mse in results_table2:
        mse_str = f"{mse:.4f}" if isinstance(mse, (float, np.float64)) else str(mse)
        print(f"{sr_no:<10} {p1_order:<15.2f} {p2_order:<15.2f} {mse_str:<25}")
    print("="*80)

    # --- Initial FRFT Demonstration (kept for standalone test with a single order) ---
    print("\n--- Initial FRFT Demonstration (for a single order for both phases) ---")
    sample_image_for_frft = K_original_image
    a_order_demo = 1.0 # Example order for both rows and columns for demonstration
    print(f"Applying 2D FRFT with order a = {a_order_demo} to a sample {sample_image_for_frft.shape} array.")
    try:
        # Apply 2D FRFT using the same order for both row and column transforms
        transformed_image_frft_single_order = apply2d_frft_separable(sample_image_for_frft, a_order_demo, a_order_demo)
        print("2D FRFT transformation complete (single order).")
        print("\nTransformed Image (first 3x3 real part):\n", np.real(transformed_image_frft_single_order[:3,:3]))
        # print("\nTransformed Image (first 3x3 imaginary part):\n", np.imag(transformed_image_frft_single_order[:3,:3]))
        print(f"\nShape of transformed image: {transformed_image_frft_single_order.shape}")
        print(f"Data type of transformed image: {transformed_image_frft_single_order.dtype}")
    except ValueError as e:
        print(f"Error during FRFT calculation: {e}")
