from flight_controller import FlightController
from cost_functions import free_flight_cost_function
import sys
sys.path.append('../models')
from models import ResidualDynamicsModel
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prospective_paths(ax, start_state, controller):
    prospective_paths = controller.mppi.states[0]
    for i in range(len(prospective_paths)):
        start_x = start_state[0]
        start_y = start_state[1]
        start_z = start_state[2]

        x = np.concatenate((start_x, prospective_paths[i, :, 0].detach().numpy()), axis=0)
        y = np.concatenate((start_y, prospective_paths[i, :, 1].detach().numpy()), axis=0)
        z = np.concatenate((start_z, prospective_paths[i, :, 2].detach().numpy()), axis=0)
        ax.plot(x, z, y, c='g')
    ax.scatter(start_x, start_z, start_y, c='r', s=10)


if __name__ == '__main__':
    CSV_FILENAME = "../data/processed_tags3_right_wallalltimes.csv"
    index_limit = [100, 100000]
    df = pd.read_csv(CSV_FILENAME)
    state = torch.tensor([[df['position.x'][index_limit[0]], df['position.y'][index_limit[0]], df['position.z'][index_limit[0]], df['R11'][index_limit[0]], df['R21'][index_limit[0]], df['R31'][index_limit[0]], df['R12'][index_limit[0]], df['R22'][index_limit[0]], df['R32'][index_limit[0]], df['vel.x'][index_limit[0]], df['vel.y'][index_limit[0]], df['vel.z'][index_limit[0]], df['ang_vel_x'][index_limit[0]], df['ang_vel_y'][index_limit[0]], df['ang_vel_z'][index_limit[0]]]], dtype=torch.float32).numpy()
    print("Starting State: ", state[0, :3])

    start_state = [np.array([state[0, 0]]), np.array([state[0, 1]]), np.array([state[0, 2]])]

    state_dim = 15
    action_dim = 4
    model = ResidualDynamicsModel(state_dim, action_dim)
    model.load_state_dict(torch.load('../models/multistep_residual_model.pt'))
    model.eval()
    #model.cpu()
    controller = FlightController(model, free_flight_cost_function)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_lst = []
    y_lst = []
    z_lst = []
    for i in range(3000):
        #print("STATE: ", state)
        action = controller.control(state)

        state = torch.from_numpy(state)
        action = torch.from_numpy(action).reshape((-1, action.shape[0]))

        next_state = model(state, action)

        # TODO: Add randomness to next state: Model_prediction + epsilon where epsilon is fron N(0, Sigma)

        if i % 500 == 0:
            plot_prospective_paths(ax, start_state, controller)
        state = next_state.detach().numpy()
        start_state = [np.array([state[0, 0]]), np.array([state[0, 1]]), np.array([state[0, 2]])]

        x_lst.append(state[0, 0])
        y_lst.append(state[0, 1])
        z_lst.append(state[0, 2])

    ax.plot(x_lst, z_lst, y_lst, c='r')
    plt.show()

