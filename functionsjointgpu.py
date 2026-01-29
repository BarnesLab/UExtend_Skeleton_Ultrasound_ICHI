# docker run -it -p 8080:8080 -p 8888:8888 -v /Users/shashwat/workspace/stroke-shape:/home/jovyan/work jupyter/datascience-notebook

import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda:1')
device = torch.device('cuda:1')

import geomstats.backend as gs
import numpy as np
import pandas as pd
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.visualization import KendallDisk, KendallSphere
import fdasrsf
import matplotlib.pyplot as plt
from geomstats.geometry.matrices import Matrices
import scipy
from scipy.signal import butter
from joblib import Parallel, delayed
import random
from tqdm.notebook import tqdm
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf


preshape = PreShapeSpace(29, 3)
preshape.equip_with_group_action("rotations")
preshape.equip_with_quotient_structure()


def OPA_gpu(A_t, B_t, reflect=False):
    
    """ Alignes A to B """       
    if reflect:
        # Correctly transpose A_t and B_t to enable proper matrix multiplication
        A = A_t.permute(2, 1, 0)  # (200, 3, 29)
        B = B_t.permute(2, 1, 0)  # (200, 3, 29)
        
        # Transpose A to get (200, 29, 3)
        A_transposed = A.permute(0, 2, 1)  # (200, 29, 3)

        # Now perform batch matrix multiplication across the 'sample' dimension
        product = B @ A_transposed
        u, sigma, v_t = torch.linalg.svd(product, full_matrices=False)
        R = u @ v_t

        # Apply R to the original A matrix, properly transposed to match dimensions
        aligned = R @ A
        
        # returning numpy array
        return aligned.permute(2, 1, 0)  # Back to (29, 3, 200)
    else:
        # converting (29, 3, 200) to (200, 29, 3)
        A = A_t.permute(2, 0, 1)
        B = B_t.permute(2, 0, 1)
        
        # aligning 
        aligned =  Matrices.align_matrices(A, B)
        # returning numpy array
        return aligned.permute(1, 2, 0)  # Back to (29, 3, 200)



def log_gpu(p1, p2):
    # p1: landmarks x ambient
    # p2: landmarks x ambient
    p1 = p1.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    p2 = p2.permute(2, 0, 1)
    result = preshape.quotient.metric.log(p2, p1)
    return result.permute(1, 2, 0)
    # return preshape.metric.log(p1, p2)

def log_gpu_frechet(p1, p2):
    # p1: (29, 3, 200), a single set of landmarks across time
    # p2: (130, 29, 3, 200), a batch of sets of landmarks across time
    
    # First, add a new dimension at the beginning of p1, then expand to match p2's size
    p1_expanded = p1.unsqueeze(0).expand(p2.shape[0], -1, -1, -1)  # Expand p1 to (130, 29, 3, 200)
    
    # Permute to match the expected dimensions for processing
    p1_expanded = p1_expanded.permute(0, 3, 1, 2)  # (130, 200, 29, 3)
    p2 = p2.permute(0, 3, 1, 2)  # (130, 200, 29, 3)
    
    # Calculate log map in a vectorized form across all samples and time points
    result = preshape.quotient.metric.log(p2, p1_expanded)  # Assumes this function can handle the batch dimension
    # Permute back to the original dimensions
    result = result.permute(2, 3, 1, 0)  # Back to (29, 3, 200, 130)

    return result


def exp_gpu(p, v):
    # p: landmarks x ambient
    p = p.permute(2, 0, 1)
    v = v.permute(2, 0, 1)
    result = preshape.metric.exp(v, p)
    return result.permute(1, 2, 0)



def parallel_gpu(v, p1, p2, n_steps=10):
    v = v.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    p1 = p1.permute(2, 0, 1) 
    p2 = p2.permute(2, 0, 1)
    result = preshape.quotient.metric.parallel_transport(tangent_vec=v, base_point=p1, end_point=p2, n_steps=2)
    return result.permute(1, 2, 0)


def cov_der_gpu(beta_t, delta_t, c):
    """Given a function, calculate a derivative in the tangent space of beta(t)"""
    beta_dot_t = torch.zeros_like(beta_t)

    # Compute the log differences for all t except the last one, in parallel
    beta_dot_t[:, :, :-1] = log_gpu(beta_t[:, :, :-1], beta_t[:, :, 1:]) / delta_t

    # Handle the last point separately if required
    pt = parallel_gpu(beta_dot_t[:, :, -2].unsqueeze(2), beta_t[:, :, -2].unsqueeze(2), beta_t[:, :, -1].unsqueeze(2))
    beta_dot_t[:, :, -1] = pt.squeeze(2)

    return beta_dot_t


def parallel_vf_gpu(v, beta, c):
    """Transports the entire vector field v to the tangent spaces at beta to the tangent space of reference point c"""
    c = c.unsqueeze(2)
    
    # Use parallel_gpu to process the entire batch
    parallel_vf = parallel_gpu(v, beta, c)
    
    return parallel_vf
    
    
def rotate_trajectory_align_gpu(mu, traj, reflect=False):
    """Batch-aligns trajectories with mu using the modified OPA."""
    # Call the batched OPA directly
    traj_aligned = OPA_gpu(traj, mu, reflect=reflect)
    return traj_aligned



def srvf_gpu(beta_dot_t, delta_t):

    # Calculating norm of each slice along the first two axes
    norms = torch.linalg.norm(beta_dot_t, dim=(0, 1), keepdim=True)

    thresh = 0.0000001
    # Clamping the norm values to avoid division by zero or very small numbers
    norms = torch.clamp(norms, min=thresh)

    # Normalize beta_dot_t by norms
    q_t = beta_dot_t / norms

    return q_t


# def tsrvf(beta_t, beta_emg_t, delta_t, c):
def tsrvf(beta_t, delta_t, c):   
    # beta_dot = cov_der(beta_t, delta_t)

    # beta_parallel = parallel_vf(beta_dot, beta, c)
    beta_dot = cov_der_gpu(beta_t, delta_t, c)
    # beta_emg_dot = np.gradient(beta_emg_t, axis=1) / delta_t.cpu().numpy()
    # beta_emg_dot = torch.from_numpy(beta_emg_dot).to(device)

    beta_dot_c = parallel_vf_gpu(beta_dot, beta_t, c)
    beta_dot_c = torch.reshape(beta_dot_c, (beta_dot.shape[0] * beta_dot.shape[1], beta_dot.shape[2]))
    # beta_joint_c = torch.cat((beta_dot_c, beta_emg_dot), dim=0)

    # q_t = srvf_gpu(beta_joint_c, delta_t)
    q_t = srvf_gpu(beta_dot_c, delta_t)

    return q_t


# def temporal_align(mu, mu_emg, beta, beta_emg, delta_t):
def temporal_align(mu, beta, delta_t):
    c = mu[:, :, 0]

    """ mu, beta are two rotationally aligned trajectories """
    # q_mu = tsrvf(mu, mu_emg, delta_t, c)
    # q_beta = tsrvf(beta, beta_emg, delta_t, c)

    q_mu = tsrvf(mu, delta_t, c)
    q_beta = tsrvf(beta, delta_t, c)
    
    # q_mu_flat = q_mu.reshape(-1, q_mu.shape[2])
    # q_beta_flat = q_beta.reshape(-1, q_mu.shape[2])

    gamma_inv = cf.optimum_reparam_curve(q_mu.cpu().numpy(), q_beta.cpu().numpy(), method="DP")
    return gamma_inv


def compose_emg(beta_emg, t, gamma):
    i = 1
    beta_emg_gamma = np.zeros((beta_emg.shape[0], gamma.shape[0]))

    for j in range(gamma.shape[0]):
        while t[i] < gamma[j]:
            i = i + 1
        delta_t = (gamma[j] - t[i - 1]) / (t[i] - t[i - 1])
        beta_emg_gamma[:, j] = beta_emg[:, i - 1] + delta_t * (beta_emg[:, i] - beta_emg[:, i - 1])
    return beta_emg_gamma

def compose_gpu(beta, t, gamma):
    # Convert numpy arrays to torch tensors and move to GPU
    gamma = torch.from_numpy(gamma).to(device)

    # Pre-compute indices i for each gamma using searchsorted
    i = torch.searchsorted(t, gamma, right=True)
    i = torch.clamp(i, min=1, max=t.shape[0] - 1)  # Ensure indices are valid

    # Calculate delta_t for all gamma simultaneously
    delta_t = (gamma - t[i - 1]) / (t[i] - t[i - 1])

    # Prepare batch operations
    p1 = beta[:, :, i - 1]  # Points from beta corresponding to i-1
    p2 = beta[:, :, i]      # Points from beta corresponding to i

    # Batch compute the log map of the paths
    log_v = log_gpu(p1, p2)  # Perform batch log map
    v = log_v * delta_t.unsqueeze(0).unsqueeze(0)  # Apply delta_t scaling

    # Batch compute the exponential map
    beta_gamma = exp_gpu(p1, v)  # Perform batch exponential map

    return beta_gamma


# def temporal_rotation_align(mu, mu_emg, beta, beta_emg, t, iterations=10, tol=10 ** (-5)):
def temporal_rotation_align(mu, beta, t, iterations=10, tol=10 ** (-5)):
    prev_error = -10000
    delta_t = t[1] - t[0]
    history = []

    beta_hat = beta
    # beta_emg_hat = beta_emg

    for iteration in tqdm(range(iterations)):
        error = torch.norm(mu - beta_hat).item() #+ np.linalg.norm(mu_emg - beta_emg_hat)
        history.append(error)

        beta_hat = rotate_trajectory_align_gpu(mu, beta_hat)

        # gamma_inv = temporal_align(mu, mu_emg, beta_hat, beta_emg_hat, delta_t)
        gamma_inv = temporal_align(mu, beta_hat, delta_t)

        beta_hat = compose_gpu(beta_hat, t, gamma_inv)
        # beta_emg_hat = compose_emg(beta_emg, t.cpu().numpy(), gamma_inv)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

    # return beta_hat, beta_emg_hat, gamma_inv, history
    return beta_hat, gamma_inv, history



# def parallel_align(mu, mu_emg, betas, betas_emg, t):
def parallel_align(mu, betas, t):
    N = len(betas)

    def align(n):
        # return temporal_rotation_align(mu, mu_emg, betas[n], betas_emg[n], t)
        return temporal_rotation_align(mu, betas[n], t)

    results = Parallel(n_jobs=-1)(delayed(align)(n) for n in range(N))

    # betas_aligned, betas_emg_aligned, gammas, temp_histories = zip(*results)
    betas_aligned, gammas, temp_histories = zip(*results)

    return (
        list(betas_aligned),
        # list(betas_emg_aligned),
        list(gammas),
        list(temp_histories),
    )


# def frechet_joint(betas, betas_emg, t, iterations=50, plot=True, tol=10 ** (-5)):
def frechet_joint(betas, t, iterations=50, plot=True, tol=10 ** (-5)):
    epsilon = 0.1
    prev_error = -10000

    history = []

    N = len(betas)
    # idx = np.random.choice(range(len(betas_resampled_healthy)))
    idx = 0
    mu = betas[idx]
    # mu_emg = betas_emg_healthy[idx]
                
    mu = torch.from_numpy(mu).to(device)
    betas = [torch.from_numpy(beta).to(device) for beta in betas]
    t = torch.from_numpy(t).to(device)

    
    for iteration in tqdm(range(iterations)):
        # we use the original betas here to avoid repeated sampling of same time series, leading to loss of information
        # betas_aligned, emgs_aligned, gammas, temp_histories = parallel_align(mu, mu_emg, betas, betas_emg, t)
        betas_aligned, gammas, temp_histories = parallel_align(mu, betas, t)

        betas_aligned_torch = torch.stack(betas_aligned, dim=0)
        # tangent_emg = np.zeros((mu_emg.shape[0], mu_emg.shape[1], N))
        # mean_emg_tangent_vec = np.zeros(mu_emg.shape)
        
        # make sure to shoot a vector to the aligned betas
        tangent_vec = log_gpu_frechet(mu, betas_aligned_torch)
        mean_tangent_vec = torch.mean(tangent_vec, dim=3)

        # for tt in range(mu.shape[2]):
            # for n in range(N):
                # make sure to shoot a vector to the aligned betas
                # tangent_emg[:, tt, n] = emgs_aligned[n][:, tt] - mu_emg[:, tt]  # for emg
                # mean_emg_tangent_vec[:, tt] += tangent_emg[:, tt, n] / float(N)  # for emg

        # mu_emg = mu_emg + epsilon * mean_emg_tangent_vec
        mu = exp_gpu(mu, epsilon * mean_tangent_vec)
        error = (torch.linalg.norm(mean_tangent_vec)**2).item() #+ np.linalg.norm(mean_emg_tangent_vec)**2
        history.append(error)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

        if plot:
            plt.figure()
            plt.plot(history)
            plt.show()
            
    mu = mu.cpu().numpy()
    betas_aligned = [beta.cpu().numpy() for beta in betas_aligned]
    tangent_vec = tangent_vec.cpu().numpy()

    # return mu, mu_emg, betas_aligned, emgs_aligned, gammas, tangent_vec, tangent_emg, history
    return mu, betas_aligned, gammas, tangent_vec, history


def preprocess(x):
    """Removes translations and scaling from a k (landmarks) x m (ambient dimension, eg. 2 for 2d shapes)"""
    mu = x.mean(axis=0)
    for i in range(x.shape[0]):
        x[i, :] = x[i, :] - mu
    x = x / np.linalg.norm(x, ord="fro")
    return x

def preprocess_temporal(beta):
    """Mean centers data and removes scaling for kendall shape space"""
    # betas (K, M, T)
    for t in range(beta.shape[2]):
        beta[:, :, t] = preprocess(beta[:, :, t])
    return beta


def process_kinematic(data, gamma_t):
    pids = data.keys()
    betas_resampled = []

    for i, pid in enumerate(pids):
        beta = preprocess_temporal(data[pid])
        t = torch.linspace(0, 1, steps=beta.shape[2])

        beta_resampled = compose_gpu(torch.from_numpy(beta).to(device), t, gamma_t).cpu().numpy()
        betas_resampled.append(beta_resampled)

    return betas_resampled


def calculate_rms_over_windows(sig_arr, column_name, sampling_frequency, window_size_ms):
    """
    Calculate RMS values over a specified window size for a given sensor signal.

    Parameters:
    sig_arr: Array containing the sensor signal.
    column_name (str): Name of the column with the sensor signal.
    sampling_frequency (int): Sampling frequency of the signal in Hz.
    window_size_ms (int): Size of the window in milliseconds.

    Returns:
    numpy array: array of RMS values.
    """
    df_signal = pd.DataFrame(sig_arr, columns=['signal'])
    df_copy = df_signal.copy()

    # Calculate the number of samples per window
    samples_per_window = int(sampling_frequency * (window_size_ms / 1000))

    # Function to calculate RMS for a window
    def calculate_rms(window):
        return np.sqrt(np.mean(window**2))

    # Use rolling window to calculate RMS
    df_copy["RMS"] = df_copy[column_name].rolling(window=samples_per_window, min_periods=1).apply(calculate_rms, raw=True)

    return df_copy["RMS"].values


def bandpass_filter(signal, sampling_frequency, lowcut, highcut, order=4):
    nyquist = 0.5 * sampling_frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    return filtered_signal



def process_emg(data, gamma_t):
    pids = data.keys()
    betas_resampled = []
    time_points = gamma_t.shape[0]

    for i, pid in enumerate(pids):
        beta = data[pid]
        beta_rms = beta.copy()
        t = np.linspace(0, 1, beta.shape[1])

        for muscle in range(beta.shape[0]):
            signal = beta[muscle,:]
            signal_filtered= bandpass_filter(signal, 1000, 10, 200)
            rms = calculate_rms_over_windows(signal_filtered, "signal", 1000, 50)
            beta_rms[muscle, :] = rms
      
        beta_resampled = compose_emg(beta_rms, t, gamma_t)
        a = -cf.calculatecentroid(beta_resampled)             # 14
        beta_resampled += np.tile(a, (time_points, 1)).T      # 14x200  
        
        betas_resampled.append(beta_resampled)

    return betas_resampled