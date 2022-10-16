from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from src.visualization import visualize_motion, visualize_k_motions, visualize_k_motions_exp, set_default
from src.utils import compute_marginalize_cartesian, compute_mean_exp, compute_cov_exp, SE2, Agent

set_default(figsize=(10, 6))


def uncertainty_propagation(agent: Agent, t: float, D: int, T: int = 1, mov: str = 'linear'):
    """
    Propagate the uncertainty giving the transmition model
    """
    # Init mean and cov
    mean = np.eye(3)
    cov = np.zeros((3, 3))
    # Obtain agent params
    arc = agent.arc
    rate = agent.rate
    l = agent.l
    r = agent.r
    # Propagate uncertainty
    if mov == 'linear':
        # Both wheels have equal angular rotation
        w = agent.w1
        # Compute mean E.q (32)
        mean[0, 2] = r * w * t
        # Compute covariance E.q (33)
        cov[0, 0] = 0.5 * (D * (r ** 2) * t)
        cov[1, 1] = (2 * D * (w ** 2) * (r ** 4) * (t ** 3)) / (3 * (l ** 2))
        cov[1, 2] = cov[2, 1] = (D * w * (r ** 3) * (t ** 2)) / l ** 2
        cov[2, 2] = (2 * D * (r ** 2) * t) / l ** 2
    elif mov == 'arc':
        # Compute mean E.q (34)
        mean[0, 0] = np.cos(rate * t)
        mean[0, 1] = -np.sin(rate * t)
        mean[0, 2] = arc * np.sin(rate * t)
        mean[1, 0] = np.sin(rate * t)
        mean[1, 1] = np.cos(rate * t)
        mean[1, 2] = arc * (1 - np.cos(rate * t))
        # Compute covariance E.q (35)
        c = (D * r ** 2) / (l ** 2 * rate)
        # Sigma 11
        cov[0, 0] = (c / 8) * ((4 * arc ** 2 + l ** 2) * (2 * rate * t + np.sin(2 * rate * t)) +
                               16 * arc ** 2 * (rate * t - 2 * np.sin(rate * t)))
        # Sigma 12, 21
        cov[0, 1] = cov[1, 0] = (-c / 2) * (((4 * arc ** 2) * (-1 + np.cos(rate * t))) + l ** 2) * np.sin(rate * t / 2) ** 2
        # Sigma 13, 31
        cov[0, 2] = cov[2, 0] = (2 * c * arc) * (rate * T - np.sin(rate * t))
        # Sigma 22
        cov[1, 1] = (-c / 8) * ((4 * arc ** 2) + l ** 2) * (-2 * rate * t + np.sin(2 * rate * t))
        # Sigma 23, 32
        cov[1, 2] = cov[2, 1] = (-2 * c * arc) * (-1 + np.cos(rate * t))
        # Sigma 33
        cov[2, 2] = 2 * c * rate
    else:
        raise NotImplementedError('Select one valid time of movement')

    return SE2(pose_matrix=mean), cov


def propagation():
    # Set Diffusion coefficient
    D = 1
    T = 1  # Total displacement time
    mov = 'arc'
    # Create agent
    agent = Agent(mov=mov)
    # Integrate 10K times the trajectory
    n_trials = 10000
    last_state = np.zeros((n_trials, 3))
    for i in range(n_trials):
        last_state[i] = agent.integrate_motion(D=D)[-1]

    # Visualize final states
    visualize_k_motions(last_state, n_trials, agent=agent)
    #################
    ## Compute Exp ##
    #################
    group_poses = [SE2(pose=p) for p in last_state]
    exp_mean, exp_cov = uncertainty_propagation(agent, t=T, D=D, T=T, mov=mov)
    # Visualize Exp. distribution in Exp. coordiantes
    visualize_k_motions_exp(group_poses, n_trials, mean_exp=exp_mean, cov_exp=exp_cov, sigmas=3)
    # Visualize Exp. distribution in Euc. coordinates
    visualize_k_motions(last_state, n_trials, agent=agent,
                        se2_poses=group_poses, mean_exp=exp_mean, cov_exp=exp_cov, sigmas=5)


if __name__ == '__main__':
    propagation()
