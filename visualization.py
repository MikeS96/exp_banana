import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)


def visualize_motion(trajectory: np.ndarray, mov: str):
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
    plt.show(block=False)


def visualize_k_motions(final_states: np.ndarray, n_trials: int, agent: object,
                        mean_cartesian: np.ndarray = None, cov_cartesian: np.ndarray = None, sigmas: float = 2):
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
    # Plot contours if provided
    if mean_cartesian is not None and cov_cartesian is not None:
        plot_contours(mean_cartesian, cov_cartesian, sigmas, ax)
    # Miscellaneous
    ax.legend()
    # Hide grid lines
    ax.grid(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.legend(loc='upper left')
    plt.title('Robot trajectory integrated {} times'.format(n_trials), fontsize=22)
    plt.tight_layout()
    plt.show()


def plot_contours(mean: np.ndarray,
                  cov: np.ndarray,
                  sigmas: int,
                  ax: object,
                  cmap: str = 'autumn_r'):
    # Generating a meshgrid complacent with the 3-sigma boundary
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
    ax.contour(x, y, z, zorder=2, cmap=cmap,
               levels=[rv.pdf(np.asarray([(c * sigma_x) + mean_x,
                                          (c * sigma_y) + mean_y])) for c in [2.5, 2.0, 1.5, 1.0, 0.5, 0.0]])
