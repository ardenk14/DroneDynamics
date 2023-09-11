from flight_controller import FlightController
from cost_functions import free_flight_cost_function
import sys
sys.path.append('../models')
from models import ResidualDynamicsModel
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio

def plot_prospective_paths(ax, start_state, controller):
    prospective_paths = controller.mppi.states[0]
    costs = controller.mppi.cost_total
    best_val, best_ind = torch.topk(costs, 10, largest=False)
    absolute_best = torch.argmin(costs).detach().item()
    for i in best_ind.detach():#range(len(prospective_paths)):
        start_x = start_state[0]
        start_y = start_state[1]
        start_z = start_state[2]

        x = np.concatenate((start_x, prospective_paths[i, :, 0].detach().numpy()), axis=0)
        y = np.concatenate((start_y, prospective_paths[i, :, 1].detach().numpy()), axis=0)
        z = np.concatenate((start_z, prospective_paths[i, :, 2].detach().numpy()), axis=0)
        if i != absolute_best:
            ax.plot(x, z, y, c='g')
        else:
            ax.plot(x, z, y, c='b')
    ax.scatter(start_x, start_z, start_y, c='r', s=10)


if __name__ == '__main__':
    #CSV_FILENAME = "../data/bagfiles/processed_csvs/processed_tags3_right_wallalltimes.csv"
    #index_limit = [100, 100000]
    #df = pd.read_csv(CSV_FILENAME)
    #state = torch.tensor([[df['position.x'][index_limit[0]], df['position.y'][index_limit[0]], df['position.z'][index_limit[0]], df['R11'][index_limit[0]], df['R21'][index_limit[0]], df['R31'][index_limit[0]], df['R12'][index_limit[0]], df['R22'][index_limit[0]], df['R32'][index_limit[0]], df['vel.x'][index_limit[0]], df['vel.y'][index_limit[0]], df['vel.z'][index_limit[0]], df['ang_vel_x'][index_limit[0]], df['ang_vel_y'][index_limit[0]], df['ang_vel_z'][index_limit[0]]]], dtype=torch.float32).numpy()
    
    npz_fp = "../data/Bag1_start0.15_6.npz"
    npz_data = np.load(npz_fp)
    state = torch.tensor([npz_data['data'][0, :12]], dtype=torch.float32).numpy()

    
    print("Starting State: ", state[0, :3])

    start_state = [np.array([state[0, 0]]), np.array([state[0, 1]]), np.array([state[0, 2]])]

    state_dim = 12#15
    action_dim = 16#4
    model = ResidualDynamicsModel(state_dim, action_dim)
    model.load_state_dict(torch.load('../models/multistep_residual_model.pt'))
    model.eval()
    #model.cpu()
    controller = FlightController(model, free_flight_cost_function)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.set_aspect('equal')
    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_ylim3d(bottom=0, top=4) 
    ax.axes.set_zlim3d(bottom=-1, top=2) 
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    x_lst = []
    y_lst = []
    z_lst = []
    frames = []
    for i in range(100):
        #print("STATE: ", state)
        action = controller.control(state)

        state = torch.from_numpy(state)
        action = torch.from_numpy(action).reshape((-1, action.shape[0]))

        #print("GOT HERE")
        next_state = model(state, action)
        print("Next States: ", next_state)

        # TODO: Add randomness to next state: Model_prediction + epsilon where epsilon is fron N(0, Sigma)

        if i % 1 == 0:
            plot_prospective_paths(ax, start_state, controller)
            ax.plot(x_lst, z_lst, y_lst, c='r')

            # Save the current frame as an image file
            filename = f"frames/frame_{i:03d}.png"
            plt.savefig(filename)
            
            # Add the frame to the list
            frames.append(imageio.imread(filename))

        state = next_state.detach().numpy()
        start_state = [np.array([state[0, 0]]), np.array([state[0, 1]]), np.array([state[0, 2]])]

        x_lst.append(state[0, 0])
        y_lst.append(state[0, 1])
        z_lst.append(state[0, 2])

    #ax.plot(x_lst, z_lst, y_lst, c='r')
    #plt.show()

print("Creating animation...")
imageio.mimsave("animation.gif", frames, fps=10)
plt.close()
print("Done!")

