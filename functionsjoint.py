# docker run -it -p 8080:8080 -p 8888:8888 -v /Users/shashwat/workspace/stroke-shape:/home/jovyan/work jupyter/datascience-notebook

from scipy import ndimage
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.visualization import KendallDisk, KendallSphere
import geomstats.backend as gs
import fdasrsf
import matplotlib.pyplot as plt
import pandas as pd
from geomstats.geometry.matrices import Matrices

preshape = PreShapeSpace(29, 3)
# preshape = PreShapeSpace(3, 2)
preshape.equip_with_group_action("rotations")
preshape.equip_with_quotient_structure()


def OPA(A_t, B_t):
    # Aligns A matrix to match B matrix
    # A_t: landmarks x dim
    # B_t: landmarks x dim
    return Matrices.align_matrices(A_t, B_t)
    

# def log(p1, p2):
#   # Aligns p2 to p1
#   """ Straightens the pearl beads in a tangent space pointing towards p2 """
#   s = p2.shape
#   p2 = OPA(p2, p1)
#   p2_v = p2.flatten()#p2.reshape(s[0]*s[1])
#   p1_v = p1.flatten()#p1.reshape(s[0]*s[1])

#   dot = np.clip(np.dot(p1_v, p2_v), -1, 1)
#   theta = np.arccos(dot)

#   # TO calculate the limit and prevent blowup when both vectors are exactly equal
#   if theta==0:
#     frac = 1
#   else:
#     frac = (theta/np.sin(theta))

#   t = (p2_v - np.dot(p2_v, p1_v)*p1_v)*frac
#   return t.reshape(s)

# def exp(p, v, theta=None):
#   """ Pushes the object on the manifold """
#   if not theta:
#     theta = np.linalg.norm(v)
#   p2 = np.cos(theta)*p + np.sin(theta)*v/np.linalg.norm(v)
#   return p2/np.linalg.norm(p2, ord='fro')


def log(p1, p2):
    # p1: landmarks x ambient
    # p2: landmarks x ambient
    return preshape.quotient.metric.log(p2, p1)
    # return preshape.metric.log(p1, p2)


def exp(p, v):
    # p: landmarks x ambient
    return preshape.metric.exp(v, p)


# def parallel(v, X, Y):
#   Y_hat = OPA(Y, X)
#   return v - 2*np.trace(np.transpose(v)@Y_hat)*((X+Y_hat)/(np.linalg.norm(X+Y_hat)**2))


def parallel(v, p1, p2, n_steps=10):
    return preshape.quotient.metric.parallel_transport(tangent_vec=v, base_point=p1, end_point=p2, n_steps=2)


def parallel_custom(v, p1, p2):
    pass


def random_beta(t, m=2, k=3, sigma=4):
    """Generate smooth triangular trajectories on kendall shape space"""
    T = t.shape[0]

    # Generate random configurations of triangles
    beta = np.random.normal(size=(k, m, T), scale=10)

    # Smoothen these configurations to show that no two are too far apart
    beta = ndimage.gaussian_filter1d(beta, axis=2, sigma=sigma)

    for tt in range(0, T):
        # remove mean centering + scaling from these matrices
        beta[:, :, tt] = preprocess(beta[:, :, tt])

        if tt > 0:
            # rotational alignment to previous
            beta[:, :, tt] = OPA(beta[:, :, tt], beta[:, :, tt - 1])

    return beta


def random_rotated_temporal_beta(t, beta=None):
    if beta is None:
        beta = random_beta(t, sigma=5)
    gamma = gamma_simulate(t, alpha=random.uniform(0, 4))
    # Random reparameterization of time
    rot_beta_hat = compose(beta, t, gamma)

    # Random rotation
    R = random_rotation_matrix()

    # Multiply each shape by a random rotation matrix
    for idx in range(t.shape[0]):
        rot_beta_hat[:, :, idx] = rot_beta_hat[:, :, idx] @ R

    return beta, rot_beta_hat


def preprocess_temporal(beta):
    """Mean centers data and removes scaling for kendall shape space"""
    # betas (K, M, T)
    for t in range(beta.shape[2]):
        beta[:, :, t] = preprocess(beta[:, :, t])

    return beta


def preprocess(x):
    """Removes translations and scaling from a k (landmarks) x m (ambient dimension, eg. 2 for 2d shapes)"""
    mu = x.mean(axis=0)
    for i in range(x.shape[0]):
        x[i, :] = x[i, :] - mu
    x = x / np.linalg.norm(x, ord="fro")
    return x


def cov_der(beta_t, delta_t, c):
    # Align to c
    # beta_align_t = np.copy(beta_t)

    # beta_align_t[:,:,0] = OPA(beta_align_t[:,:,0], c) # align first point in fiber to c

    """Given a function, calculate a derivative in the tangent space of beta(t)"""
    beta_dot_t = np.zeros(beta_t.shape)

    for t in range(beta_t.shape[2] - 1):
        # beta_dot_t[:,:,t] = log(beta_align_t[:,:,t], beta_align_t[:,:,t+1])/delta_t # aligns remaining point to each other
        beta_dot_t[:, :, t] = log(beta_t[:, :, t], beta_t[:, :, t + 1]) / delta_t

    beta_dot_t[:, :, -1] = parallel(beta_dot_t[:, :, -2], beta_t[:, :, -2], beta_t[:, :, -1])

    return beta_dot_t


def cov_int(beta_dot_c_t, delta_t, c):
    """How to use trapz integration"""

    beta_t = np.zeros(beta_dot_c_t.shape)
    # print(c.shape, beta_t.shape)
    beta_t[:, :, 0] = c  # Start from reference

    for t in range(beta_t.shape[2] - 1):
        beta_t[:, :, t + 1] = exp(
            beta_t[:, :, t],
            parallel(beta_dot_c_t[:, :, t], c, beta_t[:, :, t]) * delta_t,
        )

    return beta_t


def parallel_vf(v, beta, c):
    """Transports the vector field v to the tangent space at beta[t] to tangent space of reference point c"""
    parallel_vf = np.zeros(v.shape)

    for t in range(v.shape[2]):
        parallel_vf[:, :, t] = parallel(v[:, :, t], beta[:, :, t], c)

    return parallel_vf


def srvf(beta_dot_t, delta_t):
    # beta_dot_t = np.gradient(beta_t, axis=0)/delta_t # get the gradient of the function

    # Calculating norm of columns and divide
    norm = np.sqrt(np.linalg.norm(beta_dot_t, axis=0))

    q_t = np.zeros(beta_dot_t.shape)

    thresh = 0.0000001

    # print(norm)
    for i in range(beta_dot_t.shape[1]):
        if norm[i] > thresh:
            q_t[:, i] = beta_dot_t[:, i] / norm[i]
        else:
            q_t[:, i] = beta_dot_t[:, i] * thresh

    # q_t = np.transpose([beta_dot_t[:,:,i]/norm[i] for i in range(beta_dot_t.shape[2])])
    return q_t


def tsrvf(beta_t, beta_emg_t, delta_t, c):
    # beta_dot = cov_der(beta_t, delta_t)

    # beta_parallel = parallel_vf(beta_dot, beta, c)
    beta_dot = cov_der(beta_t, delta_t, c)
    beta_emg_dot = np.gradient(beta_emg_t, axis=1) / delta_t

    beta_dot_c = parallel_vf(beta_dot, beta_t, c).reshape((beta_dot.shape[0] * beta_dot.shape[1], beta_dot.shape[2]))
    # beta_joint_c = np.vstack([beta_dot_c/np.linalg.norm(beta_dot_c), beta_emg_dot/np.linalg.norm(beta_emg_dot)])
    beta_joint_c = beta_emg_dot

    # normalize here or in srvf
    q_t = srvf(beta_joint_c, delta_t)

    return q_t


# need to write for emg
def tsrvf_to_beta(q_t, delta_t, c):
    norm = np.linalg.norm(q_t, axis=(0, 1))

    beta_dot_c_hat = np.zeros(q_t.shape)

    for l in range(q_t.shape[2]):
        beta_dot_c_hat[:, :, l] = q_t[:, :, l] * norm[l]

    beta_hat_t = cov_int(beta_dot_c_hat, delta_t, c)

    return beta_hat_t


import random


def compose(beta, t, gamma):
    """Discuss with Arafat, then move to top"""
    i = 1

    beta_gamma = np.zeros((beta.shape[0], beta.shape[1], gamma.shape[0]))

    for j in range(gamma.shape[0]):
        while t[i] < gamma[j]:
            i = i + 1
        delta_t = (gamma[j] - t[i - 1]) / (t[i] - t[i - 1])
        beta_gamma[:, :, j] = exp(beta[:, :, i - 1], log(beta[:, :, i - 1], beta[:, :, i]) * delta_t)

    return beta_gamma


def compose_emg(beta_emg, t, gamma):
    i = 1

    beta_emg_gamma = np.zeros((beta_emg.shape[0], gamma.shape[0]))

    for j in range(gamma.shape[0]):
        while t[i] < gamma[j]:
            i = i + 1
        delta_t = (gamma[j] - t[i - 1]) / (t[i] - t[i - 1])
        beta_emg_gamma[:, j] = beta_emg[:, i - 1] + delta_t * (beta_emg[:, i] - beta_emg[:, i - 1])

    return beta_emg_gamma


def rotate_trajectory_align(mu, traj):
    """Aligns beta with mu"""
    traj_aligned = np.zeros(shape=traj.shape)

    for tt in range(traj.shape[2]):
        # print(tt, traj_aligned.shape)
        traj_aligned[:, :, tt] = OPA(traj[:, :, tt], mu[:, :, tt])

    return traj_aligned


def temporal_align(mu, mu_emg, beta, beta_emg, delta_t):
    c = mu[:, :, 0]

    """ mu, beta are two rotationally aligned trajectories """
    q_mu = tsrvf(mu, mu_emg, delta_t, c)
    q_beta = tsrvf(beta, beta_emg, delta_t, c)

    # q_mu_flat = q_mu.reshape(-1, q_mu.shape[2])
    # q_beta_flat = q_beta.reshape(-1, q_mu.shape[2])

    gamma_inv = fdasrsf.curve_functions.optimum_reparam_curve(q_mu, q_beta, method="DP")

    return gamma_inv


from tqdm.notebook import tqdm


def temporal_rotation_align(mu, mu_emg, beta, beta_emg, t, iterations=10, tol=10 ** (-5)):
    prev_error = -10000
    delta_t = t[1] - t[0]
    history = []

    beta_hat = beta
    beta_emg_hat = beta_emg

    for iteration in tqdm(range(iterations)):
        error = np.linalg.norm(mu - beta_hat) + np.linalg.norm(mu_emg - beta_emg_hat)
        history.append(error)

        beta_hat = rotate_trajectory_align(mu, beta_hat)

        gamma_inv = temporal_align(mu, mu_emg, beta_hat, beta_emg_hat, delta_t)

        beta_hat = compose(beta_hat, t, gamma_inv)
        beta_emg_hat = compose_emg(beta_emg, t, gamma_inv)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

    return beta_hat, beta_emg_hat, gamma_inv, history


from joblib import Parallel, delayed


def parallel_align(mu, mu_emg, betas, betas_emg, t):
    N = len(betas)

    def align(n):
        return temporal_rotation_align(mu, mu_emg, betas[n], betas_emg[n], t)

    results = Parallel(n_jobs=-1)(delayed(align)(n) for n in range(N))

    betas_aligned, betas_emg_aligned, gammas, temp_histories = zip(*results)

    return (
        list(betas_aligned),
        list(betas_emg_aligned),
        list(gammas),
        list(temp_histories),
    )


# def parallel_align(mu, betas, t):
#     N = len(betas)

#     betas_aligned = []
#     gammas = []
#     temp_histories = []

#     for n in range(N):
#       beta_aligned, gamma_inv, temp_history = temporal_rotation_align(mu, betas[n], t)
#       betas_aligned.append(beta_aligned)
#       gammas.append(gamma_inv)
#       temp_histories.append(temp_history)

#     return betas_aligned, gammas, temp_histories


def process_kinematic(data, gamma_t):
    pids = data.keys()
    betas_resampled = []

    for i, pid in enumerate(pids):
        beta = preprocess_temporal(data[pid])
        t = np.linspace(0, 1, beta.shape[2])
        beta_resampled = compose(beta, t, gamma_t)
        betas_resampled.append(beta_resampled)

    return betas_resampled


def calculate_rms_over_windows(df, column_name, sampling_frequency, window_size_ms):
    """
    Calculate RMS values over a specified window size for a given sensor signal.

    Parameters:
    df (pd.DataFrame): DataFrame containing the sensor signal.
    column_name (str): Name of the column with the sensor signal.
    sampling_frequency (int): Sampling frequency of the signal in Hz.
    window_size_ms (int): Size of the window in milliseconds.

    Returns:
    pd.DataFrame: DataFrame with the original signal and corresponding RMS values.
    """
    df_signal = pd.DataFrame(df, columns=['signal'])
    df_copy = df_signal.copy()

    # Calculate the number of samples per window
    samples_per_window = int(sampling_frequency * (window_size_ms / 1000))
    # print(f"Samples per window: {samples_per_window}")

    # Function to calculate RMS for a window
    def calculate_rms(window):
        return np.sqrt(np.mean(window**2))

    # Use rolling window to calculate RMS
    df_copy["RMS"] = df_copy[column_name].rolling(window=samples_per_window, min_periods=1).apply(calculate_rms, raw=True)

    # Drop NaN values resulting from the rolling operation
    # df_cleaned = df.dropna(subset=['RMS'])

    return df_copy["RMS"].values


import scipy
from scipy.signal import butter

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

    for i, pid in enumerate(pids):
        beta = data[pid]
        t = np.linspace(0, 1, beta.shape[1])

        # try:
        for muscle in range(beta.shape[0]):
            
            #processed_emg = nk.emg_process(beta[muscle, :], sampling_rate=1000)
            #import pdb; pdb.set_trace()
            #amplitude = processed_emg[0]["EMG_Amplitude"]
            signal = beta[muscle,:]
            signal_filtered= bandpass_filter(signal, 1000, 10, 200)
            rms = calculate_rms_over_windows(signal_filtered, "signal", 1000, 50)

            beta[muscle, :] = rms
            #calculate_rms_over_windows(sensor_signal, "signal", sampling_frequency, window_size_ms)

        beta_resampled = compose_emg(beta, t, gamma_t)
        betas_resampled.append(beta_resampled)

        # except Exception as e:
        # print(e, pid)

    return betas_resampled


# def frechet_joint(betas, betas_emg, t, betas_resampled_healthy, betas_emg_healthy, iterations=50, plot=True, tol=10 ** (-5)):

def frechet_joint(betas, t, betas_resampled_healthy, iterations=50, plot=True, tol=10 ** (-5)):
    epsilon = 0.01
    prev_error = -10000

    history = []

    N = len(betas)

    # betas[0] K, M, T, betas_emg[0] F, T

    # Quotient translation and scaling
    # for n in range(N):
    # betas[n] = preprocess_temporal(betas[n])

    # need to preprocess betas_emg

    idx = np.random.choice(range(len(betas_resampled_healthy)))
    mu = betas_resampled_healthy[idx]
    # mu_emg = betas_emg_healthy[idx]

    for iteration in tqdm(range(iterations)):
        # we use the original betas here to avoid repeated sampling of same time series, leading to loss of information
        betas_aligned, emgs_aligned, gammas, temp_histories = parallel_align(mu, mu_emg, betas, betas_emg, t)

        tangent_vec = np.zeros((mu.shape[0], mu.shape[1], mu.shape[2], N))
        # tangent_emg = np.zeros((mu_emg.shape[0], mu_emg.shape[1], N))

        mean_tangent_vec = np.zeros(mu.shape)
        # mean_emg_tangent_vec = np.zeros(mu_emg.shape)

        for tt in range(mu.shape[2]):
            for n in range(N):
                # make sure to shoot a vector to the aligned betas
                tangent_vec[:, :, tt, n] = log(mu[:, :, tt], betas_aligned[n][:, :, tt])
                # tangent_emg[:, tt, n] = emgs_aligned[n][:, tt] - mu_emg[:, tt]  # for emg

                mean_tangent_vec[:, :, tt] += tangent_vec[:, :, tt, n] / float(N)
                # mean_emg_tangent_vec[:, tt] += tangent_emg[:, tt, n] / float(N)  # for emg

            mu[:, :, tt] = exp(mu[:, :, tt], epsilon * mean_tangent_vec[:, :, tt])
        
        # mu_emg = mu_emg + epsilon * mean_emg_tangent_vec

        error = np.linalg.norm(mean_tangent_vec) ** 2 #+ np.linalg.norm(mean_emg_tangent_vec) ** 2
        history.append(error)
        print(history)

        if abs(error - prev_error) < tol:
            break
        else:
            prev_error = error

        if plot:
            plt.figure()
            plt.plot(history)
            plt.show()

    return mu, mu_emg, betas_aligned, emgs_aligned, gammas, tangent_vec, tangent_emg, history


def gamma_simulate(t, alpha=3):
    """
    This function maps a value between 0 and 1 to a value between 0 and 1
    using a smooth monotonic function.

    Args:
        t: A floating-point number between 0 and 1.

    Returns:
        A floating-point number between 0 and 1.
    """
    # Use a sigmoid function (shifted and scaled) for smoothness.
    return np.power(t, alpha)


def random_rotation_matrix():
    # Generate a random angle between 0 and 2*pi radians
    theta = np.random.uniform(0, 2 * np.pi)

    # Compute the components of the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Create the rotation matrix
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    return rotation_matrix


def plot_triangle(ax, x, edgecolor="black"):
    ax.fill(x[:, 0], x[:, 1], facecolor="None", edgecolor=edgecolor, linewidth=1)


def plot_triangle_scatter(ax, x, edgecolor="black"):
    ax.fill(x[:, 0], x[:, 1], facecolor="None", edgecolor=edgecolor, linewidth=0.2)
    ax.scatter(x[:, 0], x[:, 1], color=edgecolor, linewidth=1)


def plot_triangle_scatter_3d(ax, x, edgecolor="black"):
    # ax.fill(x[:,0], x[:,1], x[:,2], facecolor='None', edgecolor=edgecolor, linewidth=0.2)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], linewidth=1)


from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.show()


def plot_triangle_time_static(beta, ax, edgecolor=None, cmap="viridis"):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    three_d = beta.shape[1] == 3

    if edgecolor is None:
        color_linspace = np.linspace(0, 1, beta.shape[2])

        colormap = plt.get_cmap(cmap)  # You can choose any colormap available in Matplotlib

        # Step 3: Map the linspace values to colors
        colors = colormap(color_linspace)

        for t in range(beta.shape[2]):
            if three_d:
                plot_triangle_scatter_3d(ax, beta[:, :, t], edgecolor=colors[t])
            else:
                plot_triangle_scatter(ax, beta[:, :, t], edgecolor=colors[t])

    else:
        for t in range(beta.shape[2]):
            if three_d:
                plot_triangle_scatter_3d(ax, beta[:, :, t], edgecolor=edgecolor)
            else:
                plot_triangle_scatter(ax, beta[:, :, t], edgecolor=edgecolor)


def plot_triangle_time(fig, beta, ax):
    (line,) = ax.plot([], [], "black")  # Initialize an empty line for plotting triangles

    def update(frame):
        # ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # ax.clear()
        plot_triangle_scatter(ax, beta[:, :, frame], edgecolor="black")

    ani = FuncAnimation(
        fig,
        update,
        frames=beta.shape[2],
        blit=False,
        repeat=True,
        interval=100,
    )
    return ani
