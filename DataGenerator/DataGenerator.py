from scipy.integrate import odeint
from typing import List
import numpy as np
import math


def Xdot(state, t, a, b):
    theta1 = state[0]
    theta2 = state[1]
    dot_theta1 = state[2]
    dot_theta2 = state[3]
    control = np.array([a, b])

    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    g_acc = 9.81

    M = np.array([
        [
            m1 * l1 ** 2 + m2 * (l1 ** 2 + 2 * l1 * l2 * math.cos(theta2) + l2 ** 2),
            m2 * (l1 * l2 * math.cos(theta2) + l2 ** 2)
        ],
        [
            m2 * (l1 * l2 * math.cos(theta2) + l2 ** 2),
            m2 * l2 ** 2
        ]
    ])

    c = np.array([
        -m2 * l1 * l2 * math.sin(theta2) * (2 * dot_theta1 * dot_theta2 + dot_theta2 ** 2),
        m2 * l1 * l2 * math.sin(theta2) * dot_theta1 ** 2
    ])

    g = np.array([(
                          m1 + m2) * l1 * g_acc * math.cos(theta1) + m2 * l2 * g_acc * math.cos(theta1 + theta2),
                  m2 * l2 * g_acc * math.cos(theta1 + theta2)
                  ])

    ddot_theta = np.linalg.inv(M) @ np.transpose(control - c - g)

    return np.array([dot_theta1,
                     dot_theta2,
                     ddot_theta[0],
                     ddot_theta[1]])


def ProperState(data_state):
    theta_1 = data_state[0]
    theta_2 = data_state[1]
    dot_theta_1 = data_state[2]
    dot_theta_2 = data_state[3]

    return np.array(
        [math.cos(theta_1),
         math.sin(theta_1),
         math.cos(theta_2),
         math.sin(theta_2),
         dot_theta_1,
         dot_theta_2])


class DataGenerator:
    def __init__(self, num_steps: float, num_init: int, controls: List[float]):
        self.training_data = None
        self.controls = controls
        self.num_init = num_init
        self.num_steps = num_steps

    def generate(self):
        """
        generate the problem data by performing forward simulation of the system dynamics
        ---------------------------------------------------------------------------------
        :return:
        np.ndarray, dimensions = [len(controls), num_init, num_steps, q_dim]
        """
        self.training_data = []
        t = np.arange(0.0, self.num_steps, 1)

        for control_idx in range(len(self.controls)):
            particular_control_data = []
            u = self.controls[control_idx]
            for j in range(self.num_init):
                initial_state = np.random.rand(4)
                result_ode_int = odeint(Xdot, initial_state, t, args=(u, u))

                traj = np.empty((int(self.num_steps), 6))  # data
                for i in range(int(self.num_steps)):
                    traj[i] = ProperState(result_ode_int[i])

                particular_control_data.append(traj)

            particular_control_data = np.stack(particular_control_data, axis=1)
            self.training_data.append(particular_control_data)

        self.training_data = np.stack(self.training_data)

        data = {
            'x': self.training_data,
            "u": self.controls,
            "t": t,
        }

        return data

class ArrangeData:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def arrange(self, x, u, t):
        assert 2 <= self.num_steps <= len(t)
        n_u, ts, bs = x.shape[0:3]
        x_list = []
        u_list = []

        for u_ind in range(n_u):
            temp = np.zeros((self.num_steps, bs * (ts - self.num_steps + 1), *x.shape[3:]), dtype=np.float32)

            for i in range(ts - self.num_steps + 1):
                temp[:, i * bs:(i + 1) * bs] = x[u_ind, i:i + self.num_steps]

            x_list.append(temp)
            u_array = np.array(u[u_ind:u_ind + 1], dtype=np.float32)
            u_list.append(u_array * np.ones((temp.shape[1], 2), dtype=np.float32))

        t_eval = t[0:self.num_steps]
        return np.concatenate(x_list, axis=1), np.concatenate(u_list, axis=0), t_eval

class TrajDataset:
    def __init__(self, data, num_steps):
        self.x = []
        self.u = []
        self.arranger = ArrangeData(num_steps+1)

        for i in range(data['x'].shape[0]):
            x, u, self.t_eval = self.arranger.arrange(data['x'][i:i + 1], data['u'][i:i + 1], data['t'])
            self.x.append(x)
            self.u.append(u)

        self.u_idx = 0

    def __getitem__(self, index):
        return self.x[self.u_idx][:, index], self.u[self.u_idx][index]

    def __len__(self):
        return self.u[self.u_idx].shape[0]
