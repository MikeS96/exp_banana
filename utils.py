from typing import Optional
import numpy as np


class SE2:
    def __init__(self, pose: Optional[np.ndarray] = None, pose_matrix: Optional[np.ndarray] = None):
        if pose_matrix is not None:
            self.g = pose_matrix
        elif pose is not None:
            x = pose[0]
            y = pose[1]
            theta = pose[2]
            rot_matrix = np.asarray([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
            g = np.eye(3)
            g[:2, :2] = rot_matrix
            g[0, -1], g[1, -1] = x, y
            self.g = g
        else:
            raise NotImplementedError("Provide a pose in cartesian coordinates or in SE(2)")

    def __str__(self):
        msg = '[[{}  {}  {}]\n' \
              ' [{}  {}  {}]\n' \
              ' [{}  {}  {}]]'.format(self.g[0, 0].round(3), self.g[0, 1].round(3), self.g[0, 2].round(3),
                                      self.g[1, 0].round(3), self.g[1, 1].round(3), self.g[1, 2].round(3),
                                      self.g[2, 0].round(3), self.g[2, 1].round(3), self.g[2, 2].round(3))
        return msg

    def __repr__(self):
        msg = '[[{}  {}  {}]\n' \
              ' [{}  {}  {}]\n' \
              ' [{}  {}  {}]]'.format(self.g[0, 0].round(3), self.g[0, 1].round(3), self.g[0, 2].round(3),
                                      self.g[1, 0].round(3), self.g[1, 1].round(3), self.g[1, 2].round(3),
                                      self.g[2, 0].round(3), self.g[2, 1].round(3), self.g[2, 2].round(3))
        return msg

    def invert(self, update: bool = False):
        # Invert matrix
        g = np.eye(3)
        g[:2, :2] = self.g[:2, :2].T
        g[:2, -1] = (-self.g[:2, :2].T @ self.g[:2, -1][:, np.newaxis]).squeeze()
        if update:
            self.g = g
        return SE2(pose_matrix=g)

    def compose(self, g_):
        # Compound transformations and return a new one
        new_g = self.g @ g_.g
        return SE2(pose_matrix=new_g)


class ExpSE2:
    def __init__(self, pose_matrix: Optional[SE2] = None, tau: Optional[np.ndarray] = None):
        if tau is not None:
            self.tau = tau
        elif pose_matrix is not None:
            self.g = pose_matrix.g
            self.tau = self.log_map()
        else:
            raise NotImplementedError("Provide a pose in SE(2) or exp coordinates")

    @staticmethod
    def skew_symmetric_so2(theta):
        return np.asarray([[0, -theta], [theta, 0]])

    def hat_se2(self, tau: Optional[np.ndarray] = None):
        tau = tau if tau is not None else self.tau
        tau_hat = np.zeros((3, 3))
        tau_hat[:2, :2] = self.skew_symmetric_so2(tau[-1])
        tau_hat[0, -1], tau_hat[1, -1] = tau[0], tau[1]
        return tau_hat

    def vee_se2(self, tau_hat):
        return np.asarray([tau_hat[0], tau_hat[1], tau_hat[2]])

    def exp_max(self):
        theta = self.tau[-1]
        # Compute Jacobian SE(2)
        jac = (np.sin(theta) / theta) * np.eye(2) + \
              ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        # Obtain rotation matrix
        g = np.eye(3)
        traslation = (jac @ self.tau[:2][:, np.newaxis]).squeeze()
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        g[:2, :2] = rotation
        g[:2, -1] = traslation

        return SE2(pose_matrix=g)

    def log_map(self):
        # Compute logmap SO(2)
        theta = np.arctan2(self.g[1, 0], self.g[0, 0])
        # Compute Jacobian SE(2)
        jac = (np.sin(theta) / theta) * np.eye(2) + \
              ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        # Compute translation component
        rho = (np.linalg.inv(jac) @ self.g[:2, -1]).squeeze()
        tau = np.asarray([rho[0], rho[1], theta])
        return tau

    def __str__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg

    def __repr__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg


def compute_marginalize_cartesian(samples: np.ndarray):
    """
    Compute mean and convariance of cartesian coordinates and marginalize heading.
    Marginalize heading from the joint gaussian distribution. Assume heading is in the last position.
    https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf - 2.3.2
    """
    # Compute mean
    mean = np.mean(samples, axis=0)
    # Compute covariance
    centered = samples - mean
    cov = (1 / samples.shape[0]) * np.sum(centered[:, :, np.newaxis] @ centered[:, np.newaxis, :], axis=0)
    # Marginalize heading
    marginal_cov = cov[:2, :2]
    marginal_mean = mean[:2]
    return marginal_mean, marginal_cov


def compute_mean_exp(samples: SE2, n_iter: int = 5):
    """
    Compute mean in exponential coordinates
    """
    # Initial mean guess (at identity)
    m = SE2(pose=np.zeros(3))
    N = len(samples)
    for n in range(5):
        m_1 = m.invert()
        tau_sum = np.zeros(3)
        for g in samples:
            tau_sum += ExpSE2(pose_matrix=m_1.compose(g)).tau
        tau_sum /= N
        m = m.compose(ExpSE2(tau=tau_sum).exp_max())
    return m


def compute_cov_exp(samples: SE2, mean: SE2):
    """
    Compute covariance in exponential coordinates
    """
    N = len(samples)
    # Invert mean
    m_1 = mean.invert()
    # Compute centered samples
    y = list()
    for g in samples:
        y.append(ExpSE2(pose_matrix=m_1.compose(g)).tau)
    y = np.asarray(y)
    # Compute covariance matrix 
    cov = (1 / N) * np.sum(y[:, :, np.newaxis] @ y[:, np.newaxis, :], axis=0)
    return cov
