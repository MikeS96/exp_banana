from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata

from .utils import ExpSE2, SE2


def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('font', size=14)
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)


def visualize_motion(trajectory: np.ndarray, mov: str):
    """
    Visualize one single motion of the robot
    """
    fig, ax = plt.subplots()
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], ls='--', lw=3,
            color=(0.125, 0.4, 0.811), label='Trajectory', zorder=0)
    # Draw agent
    plt.hlines(0, 0, 0.025, colors=(1, 1, 1), alpha=1.0, lw=2.5, zorder=5)
    agent = plt.Circle((0, 0), 0.025, color=(1.0, 0.466, 0.0), alpha=1.0, lw=2, fill=False, zorder=10)
    ax.set_aspect(1)
    ax.add_artist(agent)
    # Miscellaneous
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    # Hide grid lines
    plt.legend(loc='upper left')
    plt.title('Robot trajectory sample', fontsize=22)
    bounds = [-0.25, 0.25] if mov == 'linear' else [-0.25, 1.1]
    plt.ylim(bounds)
    plt.show()


def visualize_k_motions(final_states: np.ndarray, n_trials: int, agent: object,
                        mean_cartesian: np.ndarray = None, cov_cartesian: np.ndarray = None,
                        se2_poses: List[SE2] = None, mean_exp: object = None, cov_exp: np.ndarray = None,
                        sigmas: float = 2):
    """
    Visualize robot displacement in cartesian coordinates with error curves
    """
    fig, ax = plt.subplots()
    # Extract one noiseless trajectory and plot
    motion = agent.integrate_motion(D=0)
    ax.plot(motion[:, 0], motion[:, 1], ls='--', lw=3,
            color=(0.75, 0.75, 0.75), label='Noiseless motion', zorder=2)
    # Plot trajectory
    ax.scatter(final_states[:, 0], final_states[:, 1], lw=2, marker='o',
               color=(0.125, 0.4, 0.811), label='Final states', zorder=0)
    # Draw agent
    plt.hlines(0, 0, 0.025, colors=(1, 1, 1), alpha=1.0, lw=2.5, zorder=5)
    agent = plt.Circle((0, 0), 0.025, color=(1.0, 0.466, 0.0), alpha=1.0, lw=2, fill=False, zorder=10)
    ax.set_aspect(1)
    ax.add_artist(agent)
    # Plot cartesian contours if provided
    if mean_cartesian is not None and cov_cartesian is not None:
        plot_contours_cartesian(mean_cartesian, cov_cartesian, sigmas, ax, cmap="cool")
    # Plot exp contours if provided
    if mean_exp is not None and cov_exp is not None:
        plot_contours_exp(se2_poses, mean_exp, cov_exp, sigmas, ax, cartesian_coordinates=final_states, cmap='cool')
    # Miscellaneous
    ax.legend()
    # Hide grid lines
    ax.grid(False)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    plt.legend(loc='upper left')
    plt.title('Robot trajectory integrated {} times in Car. coordinates'.format(n_trials), fontsize=22)
    plt.tight_layout()
    plt.show()


def visualize_k_motions_exp(se2_poses: np.ndarray, n_trials: int,
                            mean_exp: object = None, cov_exp: np.ndarray = None, sigmas: float = 2):
    """
    Visualize robot displacement in exponential coordinates with error curves
    """
    fig, ax = plt.subplots()
    # Compute exp coordinates of poses
    exp_poses = [ExpSE2(pose_matrix=g) for g in se2_poses]
    taus = np.asarray([t.tau for t in exp_poses])
    # Plot trajectory
    ax.scatter(taus[:, 0], taus[:, 1], lw=2, marker='o',
               color=(0.125, 0.4, 0.811), label='Final states Exp. coordinates', zorder=1)
    # Plot contours if provided
    if mean_exp is not None and cov_exp is not None:
        plot_contours_exp(se2_poses, mean_exp, cov_exp, sigmas, ax, taus=taus, cmap='cool')
    # Miscellaneous
    ax.legend()
    # Hide grid lines
    ax.grid(False)
    ax.set_xlabel(r'$v_1$', fontsize=18)
    ax.set_ylabel(r'$v_2$', fontsize=18)
    plt.legend(loc='upper left')
    plt.title('Robot trajectory integrated {} times in Exp coordinates'.format(n_trials), fontsize=22)
    plt.tight_layout()
    plt.show()


def plot_contours_cartesian(mean: np.ndarray,
                            cov: np.ndarray,
                            sigmas: int,
                            ax: object,
                            cmap: str = 'autumn_r'):
    """
    Plot error curves in cartesian coordinates for pdf defined in Euclidian space
    """
    # Generating a meshgrid complacent with the n-sigma boundary
    mean_x, mean_y = mean[0], mean[1]
    sigma_x, sigma_y = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    x = np.linspace(-sigmas * sigma_x, sigmas * sigma_x, 200)
    y = np.linspace(-sigmas * sigma_y, sigmas * sigma_y, 200)
    # Center grid on mean of distribution
    x, y = np.meshgrid(x + mean_x, y + mean_y)
    pos = np.dstack((x, y))
    # Generate density for each point in grid
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)
    ax.contour(x, y, z, zorder=3, linewidths=0.5, colors='white', alpha=0.8,
               levels=[rv.pdf(np.asarray([(c * sigma_x) + mean_x,
                                          (c * sigma_y) + mean_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])
    ax.contourf(x, y, z, zorder=3, cmap=cmap, alpha=0.3,
                levels=[rv.pdf(np.asarray([(c * sigma_x) + mean_x,
                                           (c * sigma_y) + mean_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])


def plot_contours_exp_sample(se2_poses: List[SE2],
                             mean: object,
                             cov: np.ndarray,
                             sigmas: int,
                             ax: object,
                             cartesian_coordinates: Optional[List[np.ndarray]] = None,
                             taus: Optional[List[np.ndarray]] = None,
                             cmap: str = 'Reds'):
    # Exp mean
    exp_mean = ExpSE2(pose_matrix=mean)
    # Obtain stdv around x-y
    sigma_x, sigma_y = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    # Define Exp distribution
    rv = multivariate_normal(mean=np.zeros(3), cov=cov)
    # Sample exp distribution n times
    samples_exp = rv.rvs(30000)
    # Map all samples to SE(2)
    cartesian_coordinates = list()
    for s in samples_exp:
        cartesian_coordinates.append(mean.compose(ExpSE2(tau=s).exp_max()).g[:2, -1])
    cartesian_coordinates = np.asarray(cartesian_coordinates)
    # Define marginal distribution for x-y in exp coordinates
    rv = multivariate_normal(np.zeros(2), cov[:2, :2])
    # Generate density for each point in grid - data is centered so mean is zero
    # This is effectively computing the marginal v pdf without orientation
    z = rv.pdf(samples_exp[:, :2])
    if taus is not None:
        # Uncenter exp samples
        x, y = samples_exp[:, 0] + exp_mean.tau[0], samples_exp[:, 1] + exp_mean.tau[1]
    elif cartesian_coordinates is not None:
        x, y = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1]
    ax.tricontour(x, y, z, zorder=3, linewidths=0.5, colors='white', alpha=0.8,
                  levels=[rv.pdf(np.asarray([c * sigma_x, c * sigma_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])
    ax.tricontourf(x, y, z, zorder=3, cmap=cmap, alpha=0.3,
                   levels=[rv.pdf(np.asarray([c * sigma_x, c * sigma_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])


def plot_contours_exp(se2_poses: List[SE2],
                      mean: object,
                      cov: np.ndarray,
                      sigmas: int,
                      ax: object,
                      cartesian_coordinates: Optional[List[np.ndarray]] = None,
                      taus: Optional[List[np.ndarray]] = None,
                      cmap: str = 'Reds'):
    # Invert mean
    m_1 = mean.invert()
    # Obtain stdv around x-y
    sigma_x, sigma_y = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    # Center samples around mean
    centered = list()
    for g in se2_poses:
        centered.append(ExpSE2(pose_matrix=m_1.compose(g)).tau)

    centered = np.asarray(centered)
    # Generate density for each point in grid - data is centered so mean is zero
    # This is effectively computing the marginal v pdf without orientation
    rv = multivariate_normal(np.zeros(2), cov[:2, :2])
    z = rv.pdf(centered[:, :2])
    if taus is not None:
        x, y = taus[:, 0], taus[:, 1]
    elif cartesian_coordinates is not None:
        x, y = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1]
    ax.tricontour(x, y, z, zorder=3, linewidths=0.5, colors='white', alpha=0.8,
                  levels=[rv.pdf(np.asarray([c * sigma_x, c * sigma_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])
    ax.tricontourf(x, y, z, zorder=3, cmap=cmap, alpha=0.3,
                   levels=[rv.pdf(np.asarray([c * sigma_x, c * sigma_y])) for c in [2.0, 1.5, 1.0, 0.5, 0.0]])
