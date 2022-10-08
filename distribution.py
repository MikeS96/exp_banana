from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)


set_default(figsize=(10, 6))


class Agent:
    """
    args:
        @param r: Radius of wheel
        @param l: Baseline of the vehicle
        @param arc: Arc of curvature
        @param rate: Rate of spin
        @param v: Linear speed of the agent
        @param mov: Whether if displacement is along a straight line or an arc [linear, arc]
        @param seed: Random generator seed
    """

    def __init__(self,
                 r: float = 0.033,
                 l: float = 0.2,
                 arc: float = 1.0,
                 rate: float = np.pi / 2,
                 v: float = 1.0,
                 mov: str = 'linear',
                 seed: int = 1234):
        self.r = r
        self.l = l
        self.arc = arc
        self.rate = rate
        self.v = v
        self.mov = mov
        self.rng = np.random.default_rng(seed)
        # Compute wheel speed
        if mov == 'linear':
            self.w1 = self.w2 = self.v / self.r
        elif mov == 'arc':
            self.w1 = self.rate / self.r * (self.arc + (self.l / 2))
            self.w2 = self.rate / self.r * (self.arc - (self.l / 2))
        else:
            raise NotImplementedError("Please select one valid motion -> linear or arc")

    def integrate_motion(self,
                         dt: float = 0.001,
                         T: int = 1.0,
                         D: int = 1.0):
        """
        Integrate stochastic process.

        Args:
            dt: Timestep
            T: Duration of movement
            D: Diffusion coefficient
        """
        # Number of integration steps
        steps = int(T / dt)
        # Init state of agent x=0, y=0, theta=0
        state = np.zeros((steps, 3), dtype=float)
        state[0] = np.asarray([0, 0, 0])
        # Generate brownian noise
        dw1 = self.rng.normal(loc=0, scale=np.sqrt(dt), size=steps)
        dw2 = self.rng.normal(loc=0, scale=np.sqrt(dt), size=steps)
        for step in range(1, steps):
            state_t_1 = state[step - 1]
            # Deterministic motion
            det_motion = dt * np.asarray([(self.r / 2) * (self.w1 + self.w2) * np.cos(state_t_1[-1]),
                                          (self.r / 2) * (self.w1 + self.w2) * np.sin(state_t_1[-1]),
                                          (self.r / self.l) * (self.w1 - self.w2)])
            # Stochastic motion
            H = np.asarray([[(self.r / 2) * np.cos(state_t_1[-1]), (self.r / 2) * np.cos(state_t_1[-1])],
                            [(self.r / 2) * np.sin(state_t_1[-1]), (self.r / 2) * np.sin(state_t_1[-1])],
                            [self.r / self.l, -self.r / self.l]]) * np.sqrt(D)
            stoch_motion = H @ np.asarray([[dw1[step]], [dw2[step]]]).squeeze()
            # Compute delta motion
            dx = det_motion + stoch_motion
            # Compute total motion
            state[step] = state_t_1 + dx
        return state


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
    plt.show()


def visualize_k_motions(final_states: np.ndarray, n_trials: int, agent: Agent):
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


def main():
    mov = 'arc'
    # Create agent
    agent = Agent(mov=mov)
    # Sample one trajectory and visualize
    trajectory = agent.integrate_motion()
    visualize_motion(trajectory, mov)
    # Integrate 10K times the trajectory
    n_trials = 2500
    last_state = np.zeros((n_trials, 3))
    for i in range(n_trials):
        last_state[i] = agent.integrate_motion()[-1]
    # Visualize final states
    visualize_k_motions(last_state, n_trials, agent=agent)


if __name__ == '__main__':
    main()
