from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from src.visualization import visualize_motion, visualize_k_motions, visualize_k_motions_exp, set_default
from src.utils import compute_marginalize_cartesian, compute_mean_exp, compute_cov_exp, SE2, Agent

set_default(figsize=(10, 6))


def distribution():
    mov = 'arc'
    # Create agent
    agent = Agent(mov=mov)
    # Sample one trajectory and visualize
    trajectory = agent.integrate_motion()
    visualize_motion(trajectory, mov)
    # Integrate 10K times the trajectory
    n_trials = 10000
    last_state = np.zeros((n_trials, 3))
    for i in range(n_trials):
        last_state[i] = agent.integrate_motion(D=1)[-1]

    # Visualize final states
    visualize_k_motions(last_state, n_trials, agent=agent)
    #######################
    ## Compute Cartesian ##
    #######################
    xy_mean, xy_cov = compute_marginalize_cartesian(last_state)
    # Visualize Euc. distribution in Euc. coordinates
    visualize_k_motions(last_state, n_trials, agent=agent, mean_cartesian=xy_mean, cov_cartesian=xy_cov, sigmas=4)
    #################
    ## Compute Exp ##
    #################
    group_poses = [SE2(pose=p) for p in last_state]
    exp_mean = compute_mean_exp(group_poses)
    exp_cov = compute_cov_exp(group_poses, exp_mean)
    # Visualize Exp. distribution in Exp. coordiantes
    visualize_k_motions_exp(group_poses, n_trials, mean_exp=exp_mean, cov_exp=exp_cov, sigmas=3)
    # Visualize Exp. distribution in Euc. coordinates
    visualize_k_motions(last_state, n_trials, agent=agent,
                        se2_poses=group_poses, mean_exp=exp_mean, cov_exp=exp_cov, sigmas=5)


if __name__ == '__main__':
    distribution()
