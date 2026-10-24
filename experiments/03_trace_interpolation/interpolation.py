#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import tqdm # For progress bar
from mpl_toolkits.axes_grid1 import make_axes_locatable # Import for flexible colorbar placement

# Imports from rpure.py - assuming these are available in your environment
from rpure import build_resunetplusplus, Discriminator, blockUNet, read_npy


cmap = 'seismic' # Changed colormap to seismic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Seed Setting for Reproducibility ---
def set_seed(seed: int = 1) -> None:
    """
    Sets random seed for reproducibility across numpy, random, and torch.
    Ensures that experiments can be replicated.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch.cuda.current_device())
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()


# Normalize seismic slice data, including clipping.
# Returns normalized data and the clipping value used.
def normalize_slice(slice_data, q=99.99):
    """
    Normalize seismic slice by clipping extreme values symmetrically
    around 0 based on the given quantile (default: 99.9%).
    Returns the normalized data and the clip_val used for normalization.
    """
    # Flatten non-NaN values to compute percentile
    valid_data = slice_data[~np.isnan(slice_data)].flatten()

    if valid_data.size == 0:
        print('Empty or all-NaN slice encountered!!! Returning zeros.')
        return np.zeros_like(slice_data, dtype=np.float32), 0.0 # Return zeros and a clip_val of 0

    # Calculate the clipping value based on the absolute values' percentile
    clip_val = np.percentile(np.abs(valid_data), q)

    if clip_val == 0:
        print('Slice with zero percentile (or all zero data) encountered!!! Returning zeros.')
        return np.zeros_like(slice_data, dtype=np.float32), 0.0 # Return zeros and a clip_val of 0

    # Clip values symmetrically and then normalize
    clipped = np.clip(slice_data, -clip_val, clip_val)
    normalized = clipped / clip_val

    return normalized.astype(np.float32), clip_val # Return normalized data and clip_val

def denormalize_slice(normalized_data, clip_val):
    """
    Denormalize data using the original clip_val.
    Ensures the output NumPy array is 2D for single-channel images.
    """
    if isinstance(normalized_data, torch.Tensor):
        processed_data_np = normalized_data.detach().cpu().numpy().astype(np.float32)
    else:
        processed_data_np = np.asarray(normalized_data, dtype=np.float32)

    # Explicitly remove any singleton dimensions to ensure 2D for single images
    processed_data_np = np.squeeze(processed_data_np)

    # If after squeezing, it became 0D (scalar), make it 2D
    if processed_data_np.ndim == 0:
        processed_data_np = np.atleast_2d(processed_data_np)

    return processed_data_np * clip_val

def get_amplitude_spectrum(data, dt=6000):
    """
    Calculates the amplitude spectrum (magnitude of FFT).
    data: 2D numpy array (e.g., a seismic slice) or 1D numpy array (a single trace)
    dt: sampling interval (e.g., time step)
    
    The amplitude spectrum is scaled such that the peaks correspond to the
    amplitudes of the sinusoidal components in the time-domain signal.
    """
    
    dt = dt / 1000000.0 ## Convert microseconds to seconds
    
    data = np.squeeze(data)    

    if data.size == 0:
        return np.array([0]), np.array([0])

    if data.ndim == 2:
        fft_data = np.fft.fft(data, axis=1)    
        n_samples = data.shape[1]
        
    elif data.ndim == 1:
        fft_data = np.fft.fft(data)
        n_samples = data.shape[0]
    else:
        raise ValueError(f"Input data must be 1D or 2D for amplitude spectrum calculation. Received {data.ndim}D data with shape {data.shape}.")

    if n_samples == 0:
        return np.array([0]), np.array([0])

    amplitude_spectrum = np.abs(fft_data[:, :n_samples//2 + 1]) if data.ndim == 2 else np.abs(fft_data[:n_samples//2 + 1])
    
    scaling_factor = np.zeros(amplitude_spectrum.shape[-1], dtype=np.float32)
    scaling_factor[0] = 1.0 / n_samples
    for k in range(1, n_samples // 2):
        if k < scaling_factor.shape[-1]:
            scaling_factor[k] = 2.0 / n_samples
    if n_samples % 2 == 0 and n_samples // 2 < scaling_factor.shape[-1]:
        scaling_factor[n_samples // 2] = 1.0 / n_samples
            
    if data.ndim == 2:
        amplitude_spectrum = amplitude_spectrum * scaling_factor[np.newaxis, :]
    else:
        amplitude_spectrum = amplitude_spectrum * scaling_factor

    freqs = np.fft.rfftfreq(n_samples, d=dt)    
    avg_amplitude_spectrum = np.mean(amplitude_spectrum, axis=0) if data.ndim == 2 else amplitude_spectrum
    
    return freqs, avg_amplitude_spectrum

def calculate_normalized_cross_correlation(array1, array2):
    """
    Calculates the normalized cross-correlation coefficient between two 2D arrays.
    """
    # Explicitly convert inputs to NumPy arrays to prevent TypeError with torch.Tensor.std()
    array1_np = array1.detach().cpu().numpy() if isinstance(array1, torch.Tensor) else np.asarray(array1)
    array2_np = array2.detach().cpu().numpy() if isinstance(array2, torch.Tensor) else np.asarray(array2)

    flat_array1 = array1_np.flatten()
    flat_array2 = array2_np.flatten()

    mask = ~np.isnan(flat_array1) & ~np.isnan(flat_array2)
    flat_array1 = flat_array1[mask]
    flat_array2 = flat_array2[mask]

    if len(flat_array1) == 0 or np.std(flat_array1) == 0 or np.std(flat_array2) == 0:
        return 0.0

    corr_coeff = np.corrcoef(flat_array1, flat_array2)[0, 1]
    return corr_coeff

def cal_snr(real_image, fake_image):
    real_image_np = real_image.detach().cpu().numpy()
    fake_image_np = fake_image.detach().cpu().numpy()
    
    noise = real_image_np - fake_image_np
    signal_power = np.mean(real_image_np**2)
    noise_power = np.mean(noise**2)
    
    # Guard against division by zero or log of zero/negative
    if noise_power <= 1e-8: # Add a small epsilon to noise_power
        if signal_power <= 1e-8: # If both are near zero, SNR is ill-defined, return 0 or a very low number
            return 0.0
        return float('inf') # If noise is effectively zero but signal exists, SNR is infinite
    
    snr_val = 10 * np.log10(signal_power / noise_power)
    return snr_val


# --- Seismic Dataset Class ---
class SeismicDataset(Dataset):
    def __init__(self, filename='04p07ful_Realized_1.npy'):
        """
        Initializes the SeismicDataset for interpolating multiple intermediate slices.

        Args:
            filename (str): Path to the numpy file containing the seismic volume.
                            e.g., '04p07ful_Realized_1.npy'.
        """
        self.data = read_npy(filename)

        print(f"Loaded '{filename}' with shape: {self.data.shape}")

    def __len__(self):
        """
        Returns the total number of effective samples in the dataset for training.
        Each sample requires:
        - Input slices at indices `i` and `i+4`.
        - Target slices at indices `i+1`, `i+2`, `i+3`.

        Constraints:
        - `0 <= i`
        - `i+4 < self.data.shape[0]`

        Thus, `i` can range from `0` to `self.data.shape[0] - 5`.
        The number of samples is `(self.data.shape[0] - 5) - 0 + 1 = self.data.shape[0] - 4`.
        """
        if self.data.shape[0] < 5: # Need at least 5 slices (0, 1, 2, 3, 4) to form one sample
            return 0
        return self.data.shape[0] - 4

    def __getitem__(self, idx):
        """
        Retrieves an input-target pair for the interpolation task.
        The `idx` from DataLoader is a sequential index (0, 1, 2, ...).
        This `idx` directly corresponds to the starting index `i` for the sample.

        The `a_tensor` (input) will be a 2-channel tensor:
        - Slice `idx` (conceptually slice #1)
        - Slice `idx + 4` (conceptually slice #5)

        The `b_tensor` (target) will be a 3-channel tensor:
        - Slice `idx + 1` (conceptually slice #2)
        - Slice `idx + 2` (conceptually slice #3)
        - Slice `idx + 3` (conceptually slice #4)

        Args:
            idx (int): The relative index of the sample (from 0 to len-1) provided by DataLoader.

        Returns:
            tuple: A tuple containing (a_tensor, b_tensor, input_slices_denormalized_for_plot, b_clip_val, original_b_patches_denormalized).
                    - a_tensor: The concatenated input slices (normalized, transposed, 2-channel PyTorch tensor).
                    - b_tensor: The concatenated target patches (normalized, transposed, 3-channel PyTorch tensor).
                    - input_slices_denormalized_for_plot: List of raw, transposed input patches (for plotting).
                    - b_clip_val: The clip value used for normalizing the target slices (common for all 3).
                    - original_b_patches_denormalized: List of raw, transposed target patches (for plotting).
        """
        # Input slice indices
        input_slice_1_idx = idx
        input_slice_5_idx = idx + 4

        # Target slice indices
        target_slice_2_idx = idx + 1
        target_slice_3_idx = idx + 2
        target_slice_4_idx = idx + 3

        # --- Construct a_tensor (Input: slice idx and slice idx+4) ---
        a_tensors_list = []
        input_slices_denormalized_for_plot = []
        
        # Process input slice 1
        raw_input_1_2d = np.squeeze(self.data[input_slice_1_idx])
        input_1_transposed = raw_input_1_2d.T
        normalized_input_1, clip_val_input_1 = normalize_slice(input_1_transposed)
        a_tensors_list.append(torch.tensor(normalized_input_1, dtype=torch.float32).unsqueeze(0))
        input_slices_denormalized_for_plot.append(denormalize_slice(normalized_input_1, clip_val_input_1))

        # Process input slice 5
        raw_input_5_2d = np.squeeze(self.data[input_slice_5_idx])
        input_5_transposed = raw_input_5_2d.T
        normalized_input_5, clip_val_input_5 = normalize_slice(input_5_transposed)
        a_tensors_list.append(torch.tensor(normalized_input_5, dtype=torch.float32).unsqueeze(0))
        input_slices_denormalized_for_plot.append(denormalize_slice(normalized_input_5, clip_val_input_5))

        # Concatenate the 2 input slices to form the a_tensor
        a_tensor = torch.cat(a_tensors_list, dim=0) # Shape: 2 x H x W

        # --- Process b_tensor (Target: slice idx+1, idx+2, idx+3) ---
        b_tensors_list = []
        original_b_patches_denormalized = []
        
        # We need a common clip_val for all target slices for consistent denormalization
        # Let's use the clip_val from the middle target slice (slice idx+2) as a representative
        # or calculate a global one for the 3 target slices.
        # For simplicity and consistency with previous logic, let's normalize each independently
        # and then use a representative clip_val for denormalization later for plotting.
        # Or, we can calculate one clip_val for the entire 3-slice block if we want to treat them as one volume.
        # For now, let's normalize each target slice individually and use the clip_val of the middle target slice for denormalization in plotting.
        
        # Target slice 2
        raw_b_patch_2_2d = np.squeeze(self.data[target_slice_2_idx])
        b_patch_2_transposed = raw_b_patch_2_2d.T
        normalized_b_2, clip_val_b_2 = normalize_slice(b_patch_2_transposed)
        b_tensors_list.append(torch.tensor(normalized_b_2, dtype=torch.float32).unsqueeze(0))
        original_b_patches_denormalized.append(denormalize_slice(normalized_b_2, clip_val_b_2))

        # Target slice 3 (middle target slice, representative for b_clip_val)
        raw_b_patch_3_2d = np.squeeze(self.data[target_slice_3_idx])
        b_patch_3_transposed = raw_b_patch_3_2d.T
        normalized_b_3, b_clip_val = normalize_slice(b_patch_3_transposed) # This b_clip_val will be used for all target denormalization in plotting
        b_tensors_list.append(torch.tensor(normalized_b_3, dtype=torch.float32).unsqueeze(0))
        original_b_patches_denormalized.append(denormalize_slice(normalized_b_3, b_clip_val)) # Use the common b_clip_val

        # Target slice 4
        raw_b_patch_4_2d = np.squeeze(self.data[target_slice_4_idx])
        b_patch_4_transposed = raw_b_patch_4_2d.T
        normalized_b_4, clip_val_b_4 = normalize_slice(b_patch_4_transposed)
        b_tensors_list.append(torch.tensor(normalized_b_4, dtype=torch.float32).unsqueeze(0))
        original_b_patches_denormalized.append(denormalize_slice(normalized_b_4, b_clip_val)) # Use the common b_clip_val

        # Concatenate the 3 target slices to form the b_tensor
        b_tensor = torch.cat(b_tensors_list, dim=0) # Shape: 3 x H x W

        # Return all necessary components
        # Note: b_clip_val is a single scalar for denormalizing all 3 output slices
        return a_tensor, b_tensor, input_slices_denormalized_for_plot, b_clip_val, original_b_patches_denormalized


class GANLoss(torch.nn.Module):
    def __init__(self, gan_mode, device, real_label_smooth=0.9, fake_label_smooth=0.1):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented.")
        
        # Parameters for two-sided label smoothing
        self.real_label_smooth = real_label_smooth
        self.fake_label_smooth = fake_label_smooth

    def __call__(self, input, target_is_real):
        if self.gan_mode == 'lsgan':
            if target_is_real:
                target_tensor = torch.full_like(input, 1.0, device=self.device)
                # Apply real label smoothing: target = target_tensor * self.real_label_smooth + (1.0 - self.real_label_smooth) * torch.rand_like(input) * 0.1
                # Simplified label smoothing: target is within a range around 1.0 (or 0.0)
                target_tensor = target_tensor * (0.9 + (0.1 * torch.rand_like(input)))
            else:
                target_tensor = torch.full_like(input, 0.0, device=self.device)
                # Apply fake label smoothing: target = target_tensor + self.fake_label_smooth + (torch.rand_like(input) * 0.1 - 0.05)
                # Simplified label smoothing: target is within a range around 0.0
                target_tensor = target_tensor + (0.0 + (0.1 * torch.rand_like(input) - 0.05)) # Target between -0.05 and 0.05

            return self.loss(input, target_tensor)
        
        # This line should ideally not be reached if gan_mode is 'lsgan'
        # and other modes are not implemented.
        return self.loss(input, target_tensor)


def spectral_convergence(x, y, eps=1e-8, phase_weight=0, low_freq_weight_sc=1.0, freq_decay_rate_sc=1):
    """
    Computes spectral convergence between two tensors x and y using their RFFT,
    including both magnitude and an explicit phase component.
    Measures the difference in their frequency magnitudes and phases.
    Assumes x, y shape: (Batch, Channels, Height, Width).
    
    Innovation: This version applies a frequency-dependent weighting to the
    magnitude and phase differences. Low-frequency components are given more
    weight to emphasize matching broader structural information.
    
    Args:
        x (torch.Tensor): The first tensor (e.g., predicted data).
        y (torch.Tensor): The second tensor (e.g., real target data).
        eps (float): A small epsilon value for numerical stability to avoid division by zero.
        phase_weight (float): Weight for the phase component of the loss.
        low_freq_weight_sc (float): Multiplier for the lowest frequency components
                                    in spectral convergence. A value > 1.0 increases
                                    their importance. Default is 2.0.
        freq_decay_rate_sc (float): Controls how quickly the weight decays from
                                    `low_freq_weight_sc` to 1.0 as frequency increases.
                                    Higher value means faster decay. Default is 5.0.
    
    Returns:
        torch.Tensor: The combined spectral convergence loss.
    """
    # Compute Real Fast Fourier Transform (RFFT) for both tensors
    X_complex = torch.fft.rfftn(x, dim=(-2, -1)) # Complex output
    Y_complex = torch.fft.rfftn(y, dim=(-2, -1))

    # Determine frequency map dimensions
    H_freq, W_freq = X_complex.shape[-2], X_complex.shape[-1]
    
    # Generate normalized frequency coordinates for both dimensions.
    # These coordinates range from 0 (DC component) to 1 (Nyquist frequency).
    freq_y = torch.linspace(0, 1, H_freq, device=x.device)
    freq_x = torch.linspace(0, 1, W_freq, device=x.device)
    
    # Create a 2D grid of normalized frequency coordinates.
    grid_x, grid_y = torch.meshgrid(freq_x, freq_y, indexing='xy')
    
    # Calculate the 'magnitude' of the frequency vector for each point in the grid.
    freq_magnitude = torch.sqrt(grid_x**2 + grid_y**2)
    
    # Construct the frequency weighting mask.
    # The weight starts at `low_freq_weight_sc` for DC (freq_magnitude=0) and
    # decays exponentially towards 1.0 as `freq_magnitude` increases.
    weight_mask = 1.0 + (low_freq_weight_sc - 1.0) * torch.exp(-freq_decay_rate_sc * freq_magnitude)
    
    # Expand the weight mask to match the batch and channel dimensions of the
    # frequency-transformed tensors for proper broadcasting.
    weight_mask = weight_mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H_freq, W_freq)

    # --- Magnitude Spectral Convergence ---
    # Get magnitudes
    X_mag = torch.abs(X_complex)
    Y_mag = torch.abs(Y_complex)
    
    # Compute weighted difference in magnitudes
    diff_mag = Y_mag - X_mag
    weighted_diff_mag = diff_mag * weight_mask
    
    # Flatten weighted difference for norm calculation per sample in batch
    weighted_diff_mag_flat = weighted_diff_mag.reshape(weighted_diff_mag.shape[0], -1)
    X_mag_flat = X_mag.reshape(X_mag.shape[0], -1) # Denominator remains unweighted
    
    # Compute Frobenius (L2) norm for numerator and denominator for magnitude
    num_mag = torch.norm(weighted_diff_mag_flat, p='fro', dim=1)
    den_mag = torch.norm(X_mag_flat, p='fro', dim=1)
    mag_spectral_loss = (num_mag / (den_mag + eps)).mean()

    # --- Phase Spectral Convergence (Innovation) ---
    # Calculate phase angles using atan2
    X_phase = torch.atan2(X_complex.imag, X_complex.real)
    Y_phase = torch.atan2(Y_complex.imag, Y_complex.real)

    # Compute weighted difference in phases
    diff_phase = Y_phase - X_phase
    weighted_diff_phase = diff_phase * weight_mask # Apply same frequency weight to phase
    
    # Flatten weighted difference for norm calculation
    weighted_diff_phase_flat = weighted_diff_phase.reshape(weighted_diff_phase.shape[0], -1)
    X_phase_flat = X_phase.reshape(X_phase.shape[0], -1) # Denominator remains unweighted

    # Compute Frobenius (L2) norm for numerator and denominator for phase
    num_phase = torch.norm(weighted_diff_phase_flat, p='fro', dim=1)
    den_phase = torch.norm(X_phase_flat, p='fro', dim=1)
    
    # Handle cases where den_phase might be zero (e.g., if X_phase_flat is all zeros)
    phase_spectral_loss = (num_phase / (den_phase + eps)).mean()

    # Combine magnitude and phase losses with a tunable weight
    total_spectral_loss = mag_spectral_loss + phase_weight * phase_spectral_loss
    
    return total_spectral_loss


def feature_matching_loss(D, real, fake):
    """
    Calculates feature matching loss by extracting intermediate feature maps
    from the Discriminator for both real and fake data, then computing L1 loss
    between corresponding feature maps. This encourages the Generator to produce
    outputs that are structurally similar to real data at various scales.
    """
    features_real = []
    features_fake = []
    hooks = []

    # Define a hook function to capture the output of Conv2d layers for real data
    def hook_real(module, input, output):
        if isinstance(module, nn.Conv2d): # Ensure it's a Conv2d layer
            features_real.append(output)
    # Register hooks on all Conv2d layers of the Discriminator
    for module in D.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_real))
    # Perform a forward pass with real data to populate `features_real`
    _ = D(real)
    # Remove hooks immediately after use to prevent interference
    for h in hooks: h.remove()

    # Define a hook function to capture the output of Conv2d layers for fake data
    hooks = [] # Reset hooks list
    def hook_fake(module, input, output):
        if isinstance(module, nn.Conv2d): # Ensure it's a Conv2d layer
            features_fake.append(output)
    # Register new hooks for fake data
    for module in D.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fake))
    # Perform a forward pass with fake data to populate `features_fake`
    _ = D(fake)
    # Remove hooks
    for h in hooks: h.remove()

    # Compute L1 loss over all collected feature maps
    loss = 0.0
    for fr, ff in zip(features_real, features_fake):
        # Ensure feature map sizes match before computing loss (e.g., if architectures vary)
        if fr.shape != ff.shape:
            # If shapes don't match, interpolate the fake features to match real features' size
            # Ensure interpolation does not lead to an error or unexpected behavior for feature maps.
            # Using align_corners=False for bilinear interpolation is generally recommended.
            ff_interp = F.interpolate(ff, size=fr.shape[2:], mode='bilinear', align_corners=False)
            loss += F.l1_loss(ff_interp, fr) # L1 distance between real and fake feature maps
        else:
            loss += F.l1_loss(ff, fr) # L1 distance between real and fake feature maps
    return loss


# --- Main Script Execution ---
if __name__ == "__main__":
    # Enable high precision for matrix multiplications (can speed up training on compatible GPUs)
    torch.set_float32_matmul_precision('high')

    # Flag to resume training from a checkpoint
    resume_training = False # Set to True to enable resuming
    #resume_training = True # Set to True to enable resuming

    # Hyper-parameters for loss weights
    lambda_pix  = 10.0  # Weight for pixel-wise L1 loss
    lambda_spec = 1     # Weight for spectral convergence loss
    lambda_fm   = 0.1   # Weight for feature-matching loss

    # Initialize Dataset and DataLoaders
    # Using the same dataset for train and val for simplicity,
    # but in a real scenario, you'd have separate validation data.
    DATA_FILENAME = '04p07ful_Realized_1.npy'
    dataset = SeismicDataset(filename=DATA_FILENAME)
    dataset_val = SeismicDataset(filename=DATA_FILENAME) # Using same dataset for val for simplicity

    # Define training and validation indices using SubsetRandomSampler.
    # Ensure there are enough samples for both train and validation splits.
    total_samples = dataset.__len__()
    if total_samples == 0:
        print("Error: No valid samples found in the dataset. Exiting training.")
        exit()

    # Split data: 85% train, 15% validation
    train_split_point = int(0.85 * total_samples) # 85% for training
    if train_split_point == 0 and total_samples > 0: # Ensure at least one training sample if possible
        train_split_point = 1
    if total_samples - train_split_point < 1: # Ensure at least one validation sample
        if train_split_point > 0: # If total_samples is small, ensure val_idx is not empty
            train_split_point = total_samples - 1
        else:
            train_split_point = total_samples # No train/val split possible if total_samples is 0 or 1

    train_idx = list(range(train_split_point))
    train_sampler = SubsetRandomSampler(train_idx)

    val_idx = list(range(train_split_point, total_samples))
    val_sampler = SubsetRandomSampler(val_idx)

    # --- Batch sizes and Gradient Accumulation ---
    # Actual batch size due to GPU constraints
    train_batch, val_batch = 1, 1
    print(f"Actual batch sizes: Train={train_batch}, Val={val_batch}")

    # Number of mini-batches to accumulate gradients over before performing an optimizer step.
    # This simulates a larger effective batch size.
    gradient_accumulation_steps = 4 # Example: Simulates a batch size of 4 with actual batch size 1
    print(f"Gradient accumulation steps: {gradient_accumulation_steps} (simulating effective batch size of {train_batch * gradient_accumulation_steps})")


    # num_workers > 0 enables multi-process data loading, speeding up I/O
    # pin_memory=True speeds up data transfer to GPU
    train_loader = DataLoader(dataset, batch_size=train_batch, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=val_batch, sampler=val_sampler, num_workers=4, pin_memory=True)

    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(val_idx)}")
    print(f"Number of training batches (actual): {len(train_loader)}")
    print(f"Number of validation batches (actual): {len(val_loader)}")

    # Instantiate Generator (G) and Discriminator (D) models
    # Generator now takes 2 input channels and outputs 3 channels
    G = build_resunetplusplus(in_channels=2, out_channels=3, features=16).to(device)
    D = Discriminator(3, 8).to(device) # Discriminator now processes 3-channel images

    # Define Optimizers for G and D with specified learning rate and betas
    optimizer_G = optim.Adam(G.parameters(), lr=0.0009, betas=(0.5, 0.999)) # Adjusted G LR
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Learning Rate Schedulers (ReduceLROnPlateau is common for GANs)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)

    # Define GAN Loss (LSGAN is specified)
    criterion = GANLoss('lsgan', device, real_label_smooth=0.9, fake_label_smooth=0.1)

    # Lists to store training loss history
    trainloss_G, trainloss_D = [], []
    # Lists for individual generator loss components
    trainloss_G_adv, trainloss_G_pix, trainloss_G_spec, trainloss_G_fm = [], [], [], []
    train_snr_history = [] # To store average SNR per epoch

    # Lists for validation metrics
    val_loss_G_history, val_loss_D_history = [], []
    val_snr_history = []
    val_ncc_history = []

    # Training Loop Parameters
    num_epochs = 800
    p = 2 # Visualization frequency (e.g., plot every 'p' epochs)

    # Directories for saving plots and models
    images_training_dir = './images_training_interpolation' # Directory to save training plots
    os.makedirs(images_training_dir, exist_ok=True)
    checkpoint_dir = './models_interpolation'
    checkpoint_filename = 'best_model_interpolation_gan.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure models directory exists

    start_epoch = 0
    best_G_loss = float('inf')
    best_val_snr = -float('inf') # Track best validation SNR for saving model

    # --- Resume Training Logic ---
    if resume_training and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(checkpoint['G_state'])
        D.load_state_dict(checkpoint['D_state'])
        optimizer_G.load_state_dict(checkpoint['G_opt'])
        optimizer_D.load_state_dict(checkpoint['D_opt'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state']) 
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        best_G_loss = checkpoint['best_G_loss']
        best_val_snr = checkpoint.get('best_val_snr', -float('inf')) # Load best_val_snr
        trainloss_G = checkpoint['metrics'].get('trainloss_G', [])
        trainloss_D = checkpoint['metrics'].get('trainloss_D', [])
        trainloss_G_adv = checkpoint['metrics'].get('trainloss_G_adv', [])
        trainloss_G_pix = checkpoint['metrics'].get('trainloss_G_pix', [])
        trainloss_G_spec = checkpoint['metrics'].get('trainloss_G_spec', [])
        trainloss_G_fm = checkpoint['metrics'].get('trainloss_G_fm', [])
        train_snr_history = checkpoint['metrics'].get('train_snr_history', [])
        val_loss_G_history = checkpoint['metrics'].get('val_loss_G_history', [])
        val_loss_D_history = checkpoint['metrics'].get('val_loss_D_history', [])
        val_snr_history = checkpoint['metrics'].get('val_snr_history', [])
        val_ncc_history = checkpoint['metrics'].get('val_ncc_history', [])
        print(f"Resumed from epoch {start_epoch}, with best G loss: {best_G_loss:.4f}, best Val SNR: {best_val_snr:.2f} dB")
    else:
        print("Starting new training (no checkpoint found or resume_training is False).")

    # Optional: Enable anomaly detection for debugging NaNs in gradients
    # torch.autograd.set_detect_anomaly(True) 

    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        total_loss_D, total_loss_G, total_train_snr = 0.0, 0.0, 0.0
        # Initialize total for individual generator losses for the current epoch
        total_loss_G_adv, total_loss_G_pix, total_loss_G_spec, total_loss_G_fm = 0.0, 0.0, 0.0, 0.0

        G.train(); D.train() # Set models to training mode

        # Initialize counters for gradient accumulation
        d_accumulated_batches = 0
        g_accumulated_batches = 0

        for i, (inputs, real, input_slices_denormalized_for_plot, b_clip_val, original_b_patches_denormalized) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}"):
            inp = inputs.to(device)
            rea = real.to(device)

            # --- NaN/Inf Check for Inputs (Crucial Guard) ---
            if torch.isnan(inp).any() or torch.isinf(inp).any():
                print(f"\033[91mWARNING: NaN or Inf detected in inputs at epoch {epoch+1}, batch {i}. Skipping batch.\033[0m")
                continue
            if torch.isnan(rea).any() or torch.isinf(rea).any():
                print(f"\033[91mWARNING: NaN or Inf detected in real targets at epoch {epoch+1}, batch {i}. Skipping batch.\033[0m")
                continue

            # --- Update Discriminator (D) ---
            # No zero_grad() here, as we are accumulating
            
            # 1. Train D with real images
            real_score = D(rea)
            loss_D_real = criterion(real_score, True) # Use GANLoss with label smoothing
            loss_D_real_scaled = loss_D_real / gradient_accumulation_steps # Scale loss for accumulation

            # 2. Train D with fake images
            fake = G(inp)
            fake_score = D(fake.detach()) # Detach fake to prevent G's gradients from flowing through D here
            loss_D_fake = criterion(fake_score, False) # Use GANLoss with label smoothing
            loss_D_fake_scaled = loss_D_fake / gradient_accumulation_steps # Scale loss for accumulation

            # Combine D losses and perform backpropagation
            loss_D = loss_D_real_scaled + loss_D_fake_scaled
            
            # Guard against NaN/Inf in loss before backward pass
            if torch.isnan(loss_D).any() or torch.isinf(loss_D).any():
                print(f"\033[91mWARNING: NaN or Inf detected in Discriminator loss at epoch {epoch+1}, batch {i}. Skipping D backward pass for this batch.\033[0m")
            else:
                loss_D.backward()
                d_accumulated_batches += 1

            # --- Update Generator (G) ---
            # No zero_grad() here, as we are accumulating

            # Generate fake image again (important: do not detach for G's update)
            fake = G(inp)
            fake_score = D(fake)

            # Calculate Generator's adversarial loss
            loss_G_adv = criterion(fake_score, True) # G wants D to classify fake as real
            loss_G_adv_scaled = loss_G_adv / gradient_accumulation_steps

            # Calculate Pixel-wise L1 loss
            loss_G_pix = lambda_pix * F.l1_loss(fake, rea) # L1 loss between generated and real
            loss_G_pix_scaled = loss_G_pix / gradient_accumulation_steps

            # Calculate Spectral Convergence Loss with original parameters
            loss_G_spec = lambda_spec * spectral_convergence(
                fake, rea,
                low_freq_weight_sc=1.0, # Reverted to previous effective value
                freq_decay_rate_sc=1.0, # Reverted to previous effective value
                phase_weight=0 # Keep phase_weight 0 as you want to focus on amplitudes
            )
            loss_G_spec_scaled = loss_G_spec / gradient_accumulation_steps

            # Calculate Feature Matching Loss
            loss_G_fm = lambda_fm * feature_matching_loss(D, rea, fake)
            loss_G_fm_scaled = loss_G_fm / gradient_accumulation_steps

            # Combine Generator losses
            loss_G = (
                loss_G_adv_scaled +
                loss_G_pix_scaled +
                loss_G_spec_scaled +
                loss_G_fm_scaled
            )

            # Guard against NaN/Inf in loss before backward pass
            if torch.isnan(loss_G).any() or torch.isinf(loss_G).any():
                print(f"\033[91mWARNING: NaN or Inf detected in Generator loss at epoch {epoch+1}, batch {i}. Skipping G backward pass for this batch.\033[0m")
            else:
                loss_G.backward()
                g_accumulated_batches += 1

            # Accumulate losses for logging (use unscaled losses for logging purposes)
            if not (torch.isnan(loss_D_real).any() or torch.isinf(loss_D_real).any() or \
                    torch.isnan(loss_D_fake).any() or torch.isinf(loss_D_fake).any()):
                total_loss_D += (loss_D_real.item() + loss_D_fake.item())
            
            if not (torch.isnan(loss_G_adv).any() or torch.isinf(loss_G_adv).any() or \
                    torch.isnan(loss_G_pix).any() or torch.isinf(loss_G_pix).any() or \
                    torch.isnan(loss_G_spec).any() or torch.isinf(loss_G_spec).any() or \
                    torch.isnan(loss_G_fm).any() or torch.isinf(loss_G_fm).any()):
                total_loss_G += (loss_G_adv.item() + loss_G_pix.item() + loss_G_spec.item() + loss_G_fm.item())
                total_loss_G_adv += loss_G_adv.item()
                total_loss_G_pix += loss_G_pix.item()
                total_loss_G_spec += loss_G_spec.item()
                total_loss_G_fm += loss_G_fm.item()

            # Calculate SNR for logging (optional, if you want to track image quality)
            # Ensure fake and rea are not NaN/Inf before calculating SNR
            if not (torch.isnan(fake).any() or torch.isinf(fake).any() or \
                    torch.isnan(rea).any() or torch.isinf(rea).any()):
                current_snr = cal_snr(rea, fake)
                total_train_snr += current_snr
            else:
                total_train_snr += 0.0 # Add 0 if NaN/Inf

            # --- Perform Optimizer Step and Zero Grads (Gradient Accumulation Logic) ---
            # Update D parameters
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if d_accumulated_batches > 0: # Only step if gradients were accumulated
                    torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0) # Gradient clipping for Discriminator
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                    d_accumulated_batches = 0 # Reset counter
                else:
                    pass # print(f"  Batch {i+1}: No D optimizer step (no gradients accumulated or skipped due to NaN).")

                if g_accumulated_batches > 0: # Only step if gradients were accumulated
                    torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0) # Gradient clipping for Generator
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    g_accumulated_batches = 0 # Reset counter
                else:
                    pass # print(f"  Batch {i+1}: No G optimizer step (no gradients accumulated or skipped due to NaN).")


        # Average losses for the epoch
        # Guard against division by zero if train_loader is empty
        num_train_batches = len(train_loader)
        if num_train_batches == 0:
            print(f"No training batches processed in epoch {epoch+1}. Skipping train loss calculation.")
            continue

        # When logging, divide by the total number of actual mini-batches
        avg_loss_D = total_loss_D / num_train_batches
        avg_loss_G = total_loss_G / num_train_batches
        avg_loss_G_adv = total_loss_G_adv / num_train_batches
        avg_loss_G_pix = total_loss_G_pix / num_train_batches
        avg_loss_G_spec = total_loss_G_spec / num_train_batches
        avg_loss_G_fm = total_loss_G_fm / num_train_batches
        avg_train_snr = total_train_snr / num_train_batches

        trainloss_D.append(avg_loss_D)
        trainloss_G.append(avg_loss_G)
        trainloss_G_adv.append(avg_loss_G_adv)
        trainloss_G_pix.append(avg_loss_G_pix)
        trainloss_G_spec.append(avg_loss_G_spec)
        trainloss_G_fm.append(avg_loss_G_fm)
        train_snr_history.append(avg_train_snr) # Store SNR history

        print(f"Epoch {epoch+1} Train | D Loss: {avg_loss_D:.4f} | G Loss: {avg_loss_G:.4f} | "
              f"G_Adv: {avg_loss_G_adv:.4f} | G_Pix: {avg_loss_G_pix:.4f} | "
              f"G_Spec: {avg_loss_G_spec:.4f} | G_FM: {avg_loss_G_fm:.4f} | Train SNR: {avg_train_snr:.2f} dB")

        # --- Validation Loop ---
        G.eval(); D.eval() # Set models to evaluation mode
        val_total_loss_D, val_total_loss_G, val_total_snr, val_total_ncc = 0.0, 0.0, 0.0, 0.0
        val_total_G_adv, val_total_G_pix, val_total_G_spec, val_total_G_fm = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad(): # Disable gradient calculations for validation
            for i, (inputs, real, input_slices_denormalized_for_plot, b_clip_val, original_b_patches_denormalized) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation Epoch {epoch+1}"):
                inp = inputs.to(device)
                rea = real.to(device)

                if torch.isnan(inp).any() or torch.isinf(inp).any() or \
                   torch.isnan(rea).any() or torch.isinf(rea).any():
                    print(f"\033[91mWARNING: NaN or Inf detected in validation data at epoch {epoch+1}, batch {i}. Skipping validation batch.\033[0m")
                    continue

                # Discriminator loss for validation
                real_score_val = D(rea)
                loss_D_real_val = criterion(real_score_val, True)
                fake_val = G(inp)
                fake_score_val = D(fake_val.detach())
                loss_D_fake_val = criterion(fake_score_val, False)
                val_loss_D = loss_D_real_val + loss_D_fake_val
                val_total_loss_D += val_loss_D.item()

                # Generator loss for validation
                fake_score_val_G = D(fake_val)
                loss_G_adv_val = criterion(fake_score_val_G, True)
                loss_G_pix_val = lambda_pix * F.l1_loss(fake_val, rea)
                loss_G_spec_val = lambda_spec * spectral_convergence(
                    fake_val, rea,
                    low_freq_weight_sc=1.0, # Reverted to previous effective value
                    freq_decay_rate_sc=1.0, # Reverted to previous effective value
                    phase_weight=0
                )
                loss_G_fm_val = lambda_fm * feature_matching_loss(D, rea, fake_val)
                val_loss_G = loss_G_adv_val + loss_G_pix_val + loss_G_spec_val + loss_G_fm_val
                val_total_loss_G += val_loss_G.item()

                val_total_G_adv += loss_G_adv_val.item()
                val_total_G_pix += loss_G_pix_val.item()
                val_total_G_spec += loss_G_spec_val.item()
                val_total_G_fm += loss_G_fm_val.item()

                # Calculate SNR for validation
                current_val_snr = cal_snr(rea, fake_val)
                val_total_snr += current_val_snr

                # Denormalize generated fake images for plotting
                # fake_val has shape (1, 3, H, W)
                denormalized_fake_slices = [denormalize_slice(fake_val.squeeze(0)[j], b_clip_val.cpu().numpy().item()) for j in range(fake_val.shape[1])]
                
                # Original real slices are already denormalized in original_b_patches_denormalized
                
                # Calculate residues
                denormalized_residue_slices = [
                    original_b_patches_denormalized[j] - denormalized_fake_slices[j]
                    for j in range(len(original_b_patches_denormalized))
                ]

                # Initialize batch_ncc for this batch
                batch_ncc = 0.0 
                # Calculate NCC for each output slice
                for j in range(len(original_b_patches_denormalized)):
                    batch_ncc += calculate_normalized_cross_correlation(original_b_patches_denormalized[j], denormalized_fake_slices[j])
                batch_ncc /= len(original_b_patches_denormalized) # Average NCC over the 3 slices
                val_total_ncc += batch_ncc


                # Plotting for validation (e.g., first few samples or periodically)
                if (epoch + 1) % p == 0 and i == 0: # Plot first validation sample every 'p' epochs
                    
                    # Determine common color scale limits across all images to be plotted
                    all_images_to_plot = []
                    all_images_to_plot.extend(input_slices_denormalized_for_plot)
                    all_images_to_plot.extend(original_b_patches_denormalized)
                    all_images_to_plot.extend(denormalized_fake_slices)
                    all_images_to_plot.extend(denormalized_residue_slices)

                    # Flatten all data to find global min/max, handling potential empty arrays
                    # Added isinstance check to ensure img is a NumPy array before accessing .size
                    all_data_flattened = np.concatenate([
                        img.flatten()
                        for img in all_images_to_plot
                        if isinstance(img, np.ndarray) and img.size > 0
                    ])
                    
                    if all_data_flattened.size > 0:
                        global_min = all_data_flattened.min()
                        global_max = all_data_flattened.max()
                    else:
                        global_min, global_max = -1.0, 1.0 # Default if no data

                    # Adjust global min/max for residue plots if they are expected to be centered around zero
                    # For seismic data, it's often good to have a symmetric color scale around zero for both data and residue.
                    # Let's make the color scale symmetric around zero for better visualization of positive/negative amplitudes.
                    abs_max = max(abs(global_min), abs(global_max))
                    plot_vmin, plot_vmax = -abs_max, abs_max


                    fig, axs = plt.subplots(4, 3, figsize=(20, 20)) # 4 rows, 3 columns

                    # Row 1: Input Slices
                    # Explicitly squeeze to ensure 2D array for imshow
                    im_input1 = axs[0, 0].imshow(np.squeeze(input_slices_denormalized_for_plot[0]), cmap=cmap, aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
                    axs[0, 0].set_title(f'Input Slice #1')
                    axs[0, 0].axis('off')

                    im_input5 = axs[0, 1].imshow(np.squeeze(input_slices_denormalized_for_plot[1]), cmap=cmap, aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
                    axs[0, 1].set_title(f'Input Slice #5')
                    axs[0, 1].axis('off')
                    
                    axs[0, 2].axis('off') # Empty cell for alignment or overall title

                    # Rows 2, 3, 4: Original, Generated, Residue for slices #2, #3, #4
                    titles_suffix = ["#2", "#3", "#4"]
                    for j in range(3): # Iterate through the 3 output slices
                        # Original
                        axs[j+1, 0].imshow(np.squeeze(original_b_patches_denormalized[j]), cmap=cmap, aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
                        axs[j+1, 0].set_title(f'Original Slice {titles_suffix[j]}')
                        axs[j+1, 0].axis('off')

                        # Generated
                        axs[j+1, 1].imshow(np.squeeze(denormalized_fake_slices[j]), cmap=cmap, aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
                        axs[j+1, 1].set_title(f'Generated Slice {titles_suffix[j]} (Epoch {epoch+1})')
                        axs[j+1, 1].axis('off')

                        # Residue
                        im_residue = axs[j+1, 2].imshow(np.squeeze(denormalized_residue_slices[j]), cmap=cmap, aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
                        axs[j+1, 2].set_title(f'Residue {titles_suffix[j]}')
                        axs[j+1, 2].axis('off')

                    # Add a single, thin color bar to the extreme right of the entire plot area
                    # Use the last residue image as the mappable for the colorbar
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
                    fig.colorbar(im_residue, cax=cbar_ax) # Use im_residue as the mappable for the colorbar
                    
                    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the colorbar
                    plt.show() # Display the plot inline
                    plt.savefig(os.path.join(images_training_dir, f'epoch_{epoch+1}_val_interpolation_comparison.png'))
                    plt.close(fig)

        # Average validation losses and metrics for the epoch
        num_val_batches = len(val_loader)
        if num_val_batches == 0:
            print(f"No validation batches processed in epoch {epoch+1}. Skipping validation loss calculation.")
        else:
            avg_val_loss_D = val_total_loss_D / num_val_batches
            avg_val_loss_G = val_total_loss_G / num_val_batches
            avg_val_snr = val_total_snr / num_val_batches
            avg_val_ncc = val_total_ncc / num_val_batches

            val_loss_D_history.append(avg_val_loss_D)
            val_loss_G_history.append(avg_val_loss_G)
            val_snr_history.append(avg_val_snr)
            val_ncc_history.append(avg_val_ncc)

            print(f"Epoch {epoch+1} Val   | D Loss: {avg_val_loss_D:.4f} | G Loss: {avg_val_loss_G:.4f} | "
                  f"Val SNR: {avg_val_snr:.2f} dB | Val NCC: {avg_val_ncc:.4f}")

        # Update learning rate schedulers (using validation G loss)
        scheduler_G.step(avg_val_loss_G)
        scheduler_D.step(avg_val_loss_D)
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        print(f"Current LR (G): {current_lr_G:.6f}, Current LR (D): {current_lr_D:.6f}")

        # --- Save Checkpoint (Best Validation SNR) ---
        # Save model if validation SNR improves (or if it's the first epoch)
        if avg_val_snr > best_val_snr:
            best_val_snr = avg_val_snr
            print(f"Saving best model at epoch {epoch+1} with Val SNR: {best_val_snr:.2f} dB (and G Loss: {avg_loss_G:.4f})")
            checkpoint_data = {
                'epoch': epoch,
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'G_opt': optimizer_G.state_dict(),
                'D_opt': optimizer_D.state_dict(),
                'scheduler_G_state': scheduler_G.state_dict(),
                'scheduler_D_state': scheduler_D.state_dict(),
                'best_G_loss': avg_loss_G, # Keeping track of G loss at best SNR
                'best_val_snr': best_val_snr,
                'metrics': {
                    'trainloss_G': trainloss_G,
                    'trainloss_D': trainloss_D,
                    'trainloss_G_adv': trainloss_G_adv,
                    'trainloss_G_pix': trainloss_G_pix,
                    'trainloss_G_spec': trainloss_G_spec,
                    'trainloss_G_fm': trainloss_G_fm,
                    'train_snr_history': train_snr_history,
                    'val_loss_G_history': val_loss_G_history,
                    'val_loss_D_history': val_loss_D_history,
                    'val_snr_history': val_snr_history,
                    'val_ncc_history': val_ncc_history,
                }
            }
            torch.save(checkpoint_data, checkpoint_path)
        else:
            print(f"Model not saved at epoch {epoch+1} (Val SNR {avg_val_snr:.2f} dB not better than best {best_val_snr:.2f} dB).")

    print("\nTraining complete!")


# In[ ]:




