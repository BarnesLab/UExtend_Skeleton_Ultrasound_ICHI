import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda:1')
device = torch.device('cuda:1')

import geomstats.backend as gs
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.visualization import KendallDisk, KendallSphere
import fdasrsf
import matplotlib.pyplot as plt
from geomstats.geometry.matrices import Matrices

preshape = PreShapeSpace(17, 3)
# preshape = PreShapeSpace(3, 2)
preshape.equip_with_group_action("rotations")
preshape.equip_with_quotient_structure()


def procrustes_align_gpu(A_t, B_t, reflect=False):
    # Convert (29, 3, 200) to (200, 29, 3)
    source_batch = A_t.permute(2, 0, 1)
    target_batch = B_t.permute(2, 0, 1)

    aligned_batch = Matrices.align_matrices(source_batch, target_batch)
    return aligned_batch.permute(1, 2, 0)  # Back to (29, 3, 200)


def align_trajectory_orientation_gpu(mu, traj, reflect=False):
    """Batch-align a trajectory to the reference orientation using Procrustes alignment."""
    aligned_trajectory = procrustes_align_gpu(traj, mu, reflect=reflect)
    return aligned_trajectory


def shape_log_map_gpu(p1, p2):
    # p1: landmarks x ambient
    # p2: landmarks x ambient
    permuted_p1 = p1.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    permuted_p2 = p2.permute(2, 0, 1)
    log_map = preshape.quotient.metric.log(permuted_p2, permuted_p1)
    return log_map.permute(1, 2, 0)
    # return preshape.metric.log(p1, p2)


def batched_frechet_log_map_gpu(p1, p2):
    # p1: (29, 3, 200), a single set of landmarks across time
    # p2: (130, 29, 3, 200), a batch of sets of landmarks across time

    # Add a new dimension to p1, then expand it to match p2's batch size
    expanded_p1 = p1.unsqueeze(0).expand(p2.shape[0], -1, -1, -1)

    # Permute to match the expected dimensions for processing
    expanded_p1 = expanded_p1.permute(0, 3, 1, 2)  # (130, 200, 29, 3)
    permuted_p2 = p2.permute(0, 3, 1, 2)  # (130, 200, 29, 3)

    # Calculate log maps in vectorized form across samples and time points
    log_maps = preshape.quotient.metric.log(permuted_p2, expanded_p1)
    log_maps = log_maps.permute(2, 3, 1, 0)  # Back to (29, 3, 200, 130)

    return log_maps


def shape_exp_map_gpu(p, v):
    # p: landmarks x ambient
    permuted_point = p.permute(2, 0, 1)
    permuted_tangent_vector = v.permute(2, 0, 1)
    exp_map = preshape.metric.exp(permuted_tangent_vector, permuted_point)
    return exp_map.permute(1, 2, 0)


def parallel_transport_shape_gpu(v, p1, p2, n_steps=10):
    permuted_vector = v.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    permuted_start_point = p1.permute(2, 0, 1)
    permuted_end_point = p2.permute(2, 0, 1)
    transported_vector = preshape.quotient.metric.parallel_transport(
        tangent_vec=permuted_vector,
        base_point=permuted_start_point,
        end_point=permuted_end_point,
        n_steps=2,
    )
    return transported_vector.permute(1, 2, 0)


def normalize_shape_configuration(x):
    """Removes translations and scaling from a k (landmarks) x m (ambient dimension, eg. 2 for 2d shapes)"""
    centroid = x.mean(axis=0)
    for landmark_idx in range(x.shape[0]):
        x[landmark_idx, :] = x[landmark_idx, :] - centroid
    x = x / np.linalg.norm(x, ord="fro")
    return x


def normalize_shape_trajectory(data):
    """Mean centers data and removes scaling for kendall shape space"""
    for time_idx in range(data.shape[2]):
        data[:, :, time_idx] = normalize_shape_configuration(data[:, :, time_idx])

    return data


def covariant_trajectory_derivative_gpu(beta_t, delta_t, c):
    """Given a function, calculate a derivative in the tangent space of beta(t)"""
    trajectory_derivative = torch.zeros_like(beta_t)

    # Compute the log differences for all time points except the last one
    trajectory_derivative[:, :, :-1] = (
        shape_log_map_gpu(beta_t[:, :, :-1], beta_t[:, :, 1:]) / delta_t
    )

    # Handle the last point separately
    transported_last_derivative = parallel_transport_shape_gpu(
        trajectory_derivative[:, :, -2].unsqueeze(2),
        beta_t[:, :, -2].unsqueeze(2),
        beta_t[:, :, -1].unsqueeze(2),
    )
    trajectory_derivative[:, :, -1] = transported_last_derivative.squeeze(2)

    return trajectory_derivative


def transport_trajectory_vector_field_gpu(v, beta, c):
    """Transports the entire vector field v to the tangent spaces at beta to the tangent space of reference point c"""
    reference_point = c.unsqueeze(2)

    transported_vector_field = parallel_transport_shape_gpu(v, beta, reference_point)
    return transported_vector_field


def compute_srvf_representation_gpu(beta_dot_t, delta_t):
    # Calculate the norm of each slice along the first two axes
    derivative_norms = torch.linalg.norm(beta_dot_t, dim=(0, 1), keepdim=True)

    minimum_norm = 0.0000001
    derivative_norms = torch.clamp(derivative_norms, min=minimum_norm)

    normalized_srvf = beta_dot_t / derivative_norms
    return normalized_srvf


def compute_transport_srvf_gpu(beta_t, delta_t, c):
    trajectory_derivative = covariant_trajectory_derivative_gpu(beta_t, delta_t, c)
    transported_derivative = transport_trajectory_vector_field_gpu(trajectory_derivative, beta_t, c)
    trajectory_tsrvf = compute_srvf_representation_gpu(transported_derivative, delta_t)

    return trajectory_tsrvf


def warp_shape_trajectory_gpu(beta, t, gamma):
    # Convert numpy arrays to torch tensors and move to GPU
    gamma_tensor = torch.from_numpy(gamma).to(device)

    # Locate the time interval containing each gamma value
    interval_indices = torch.searchsorted(t, gamma_tensor, right=True)
    interval_indices = torch.clamp(
        interval_indices,
        min=1,
        max=t.shape[0] - 1,
    )

    # Compute interpolation weights within each interval
    interpolation_weights = (
        (gamma_tensor - t[interval_indices - 1])
        / (t[interval_indices] - t[interval_indices - 1])
    )

    left_points = beta[:, :, interval_indices - 1]
    right_points = beta[:, :, interval_indices]

    segment_log_vectors = shape_log_map_gpu(left_points, right_points)
    scaled_log_vectors = segment_log_vectors * interpolation_weights.unsqueeze(0).unsqueeze(0)

    reparameterized_trajectory = shape_exp_map_gpu(left_points, scaled_log_vectors)
    return reparameterized_trajectory


def estimate_temporal_reparameterization(mu, beta, delta_t):
    reference_point = mu[:, :, 0]

    """mu, beta are two rotationally aligned trajectories"""
    reference_tsrvf = compute_transport_srvf_gpu(mu, delta_t, reference_point)
    trajectory_tsrvf = compute_transport_srvf_gpu(beta, delta_t, reference_point)

    reference_tsrvf_flat = reference_tsrvf.reshape(-1, reference_tsrvf.shape[2])
    trajectory_tsrvf_flat = trajectory_tsrvf.reshape(-1, reference_tsrvf.shape[2])

    inverse_warp = fdasrsf.curve_functions.optimum_reparam_curve(
        reference_tsrvf_flat.cpu().numpy(),
        trajectory_tsrvf_flat.cpu().numpy(),
        method="DP",
    )

    return inverse_warp


from tqdm.notebook import tqdm


def align_shape_trajectory_spatiotemporal(mu, beta, t, iterations=10, tol=10 ** (-5), reflect=False):
    previous_error = -10000
    time_step = t[1] - t[0]
    error_history = []

    aligned_trajectory = beta

    for iteration_idx in tqdm(range(iterations)):
        alignment_error = torch.norm(mu - aligned_trajectory)
        alignment_error = alignment_error.item()
        error_history.append(alignment_error)

        aligned_trajectory = align_trajectory_orientation_gpu(
            mu,
            aligned_trajectory,
            reflect=reflect,
        )

        inverse_warp = estimate_temporal_reparameterization(mu, aligned_trajectory, time_step)
        aligned_trajectory = warp_shape_trajectory_gpu(aligned_trajectory, t, inverse_warp)

        if abs(alignment_error - previous_error) < tol:
            break
        else:
            previous_error = alignment_error

    return aligned_trajectory, inverse_warp, error_history


from joblib import Parallel, delayed


def align_trajectory_collection_parallel(mu, betas, t):
    num_trajectories = len(betas)

    def align(trajectory_idx):
        return align_shape_trajectory_spatiotemporal(mu, betas[trajectory_idx], t)

    alignment_results = Parallel(n_jobs=-1)(
        delayed(align)(trajectory_idx)
        for trajectory_idx in range(num_trajectories)
    )

    aligned_trajectories, warping_functions, alignment_histories = zip(
        *alignment_results
    )

    return (
        list(aligned_trajectories),
        list(warping_functions),
        list(alignment_histories),
    )


def resample_kinematic_trajectories(data, gamma_t):
    participant_ids = data.keys()
    resampled_trajectories = []

    for participant_idx, participant_id in enumerate(participant_ids):
        trajectory = normalize_shape_trajectory(data[participant_id])
        time_grid = torch.linspace(0, 1, steps=trajectory.shape[2])

        resampled_trajectory = warp_shape_trajectory_gpu(
            torch.from_numpy(trajectory).to(device),
            time_grid,
            gamma_t,
        ).cpu().numpy()
        resampled_trajectories.append(resampled_trajectory)

    return resampled_trajectories


def estimate_frechet_mean_trajectory(betas, t, mu_init, iterations=50, plot=True, tol=10 ** (-5)):
    original_betas = np.copy(betas)

    step_size = 0.1
    previous_error = -10000
    error_history = []

    num_trajectories = len(betas)

    # Quotient translation and scaling
    # for trajectory_idx in range(num_trajectories):
    #     betas[trajectory_idx] = normalize_shape_trajectory(betas[trajectory_idx])

    frechet_mean = torch.from_numpy(mu_init).to(device)
    trajectory_tensors = [torch.from_numpy(beta).to(device) for beta in betas]
    time_grid = torch.from_numpy(t).to(device)

    for iteration_idx in tqdm(range(iterations)):
        aligned_trajectories, warping_functions, alignment_histories = align_trajectory_collection_parallel(
            frechet_mean,
            trajectory_tensors,
            time_grid,
        )

        aligned_trajectories_tensor = torch.stack(aligned_trajectories, dim=0)

        # Compute all tangent vectors at once
        tangent_vectors = batched_frechet_log_map_gpu(
            frechet_mean,
            aligned_trajectories_tensor,
        )

        # Calculate the mean tangent vector across the last dimension
        mean_tangent_vector = torch.mean(tangent_vectors, dim=3)

        # Update the Frechet mean
        frechet_mean = shape_exp_map_gpu(
            frechet_mean,
            step_size * mean_tangent_vector,
        )

        mean_update_error = torch.linalg.norm(mean_tangent_vector) ** 2
        mean_update_error = mean_update_error.item()
        error_history.append(mean_update_error)

        if abs(mean_update_error - previous_error) < tol:
            break
        else:
            previous_error = mean_update_error

        if plot:
            plt.figure()
            plt.plot(error_history)
            plt.show()

    frechet_mean = frechet_mean.cpu().numpy()
    aligned_trajectories = [
        trajectory.cpu().numpy()
        for trajectory in aligned_trajectories
    ]
    tangent_vectors = tangent_vectors.cpu().numpy()

    return (
        frechet_mean,
        aligned_trajectories,
        warping_functions,
        tangent_vectors,
        error_history,
    )
