import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import yaml
import torch
from models import  ResidualDynamicsModel, AbsoluteDynamicsModel


"""
Visualize the drone position and orientation data from a CSV file.
Also plot model predictions.
usage: first change filename to plot
python3 plot_dataset.py
"""
CSV_FILENAME = "../data/50hz_processed_tags3_right_wall1681856871.257578_1681856949.932647.csv" #../data/processed_tags3_right_wall_commands_states.csv"
TAG_FILENAME = "../april_tags/tags_all.yaml"

def plot_trajectory(model, filename, tags_filename, index_limit=None, reset_state=False):
    df = pd.read_csv(filename)

    if index_limit is None:
        index_limit = [0, len(df)]

    results = []
    #state = torch.tensor([[df['position.x'][index_limit[0]], df['position.y'][index_limit[0]], df['position.z'][index_limit[0]], df['R11'][index_limit[0]], df['R21'][index_limit[0]], df['R31'][index_limit[0]], df['R12'][index_limit[0]], df['R22'][index_limit[0]], df['R32'][index_limit[0]], df['vel.x'][index_limit[0]], df['vel.y'][index_limit[0]], df['vel.z'][index_limit[0]], df['ang_vel_x'][index_limit[0]], df['ang_vel_y'][index_limit[0]], df['ang_vel_z'][index_limit[0]]]], dtype=torch.float32)
    state = torch.tensor([[df['position.x'][index_limit[0]], df['position.y'][index_limit[0]], df['position.z'][index_limit[0]], df['vel.x'][index_limit[0]], df['vel.y'][index_limit[0]], df['vel.z'][index_limit[0]], df['ang_vel_x'][index_limit[0]], df['ang_vel_y'][index_limit[0]], df['ang_vel_z'][index_limit[0]], df['roll'][index_limit[0]], df['pitch'][index_limit[0]], df['yaw'][index_limit[0]]]], dtype=torch.float32)
    with torch.no_grad():
        for i in range(index_limit[0], index_limit[1]):
            if reset_state:
                #state = torch.tensor([[df['position.x'][i], df['position.y'][i], df['position.z'][i], df['R11'][i], df['R21'][i], df['R31'][i], df['R12'][i], df['R22'][i], df['R32'][i], df['vel.x'][i], df['vel.y'][i], df['vel.z'][i], df['ang_vel_x'][i], df['ang_vel_y'][i], df['ang_vel_z'][i]]], dtype=torch.float32)
                state = torch.tensor([[df['position.x'][i], df['position.y'][i], df['position.z'][i], df['vel.x'][i], df['vel.y'][i], df['vel.z'][i], df['ang_vel_x'][i], df['ang_vel_y'][i], df['ang_vel_z'][i], df['roll'][i], df['pitch'][i], df['yaw'][i]]], dtype=torch.float32)

            action = torch.tensor([[df['commands[0]'][i], df['commands[1]'][i], df['commands[2]'][i], df['commands[3]'][i]]], dtype=torch.float32)

            result = model(state, action)
            results.append(result.cpu().detach().numpy()[0])
            state = result

    with open(tags_filename, 'r') as f:
        data = yaml.safe_load(f)
    
    # Plot the position
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Extract x,y,z location for each id number and plot it
    for bundle in data['tag_bundles']:
        for tag in bundle['layout']:
            id_num = tag['id']
            x = tag['x']
            y = tag['y']
            z = tag['z']
            if x > -1.0:
                ax.scatter(x, z, y, c='r', s=8)

    if index_limit:
        ax.plot(df["position.x"][index_limit[0]:index_limit[1]], df["position.z"][index_limit[0]:index_limit[1]], df["position.y"][index_limit[0]:index_limit[1]], label="True Path")
    else:
        ax.plot(df["position.x"], df["position.z"], df["position.y"])

    ax.plot([i[0] for i in results], [i[2] for i in results], [i[1] for i in results], label="Predicted Path")
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title("Position Prediction with State Replacement")
    ax.legend()
    plt.show()

    print("RESULTS: ", results)

def plot_trajectory(model, filename, reset_state=False):
    npz_data = np.load(filename)
    states = npz_data['data'][:, :12]
    actions = npz_data['data'][:, 12:]
    traj_len = states.shape[0]

    # Plot the position
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    ax.plot(x, y, z, label="True Path")

    res_x = [states[0, 0]]
    res_y = [states[0, 1]]
    res_z = [states[0, 2]]
    results = []
    state = torch.tensor([states[0]], dtype=torch.float32)
    for i in range(traj_len-1):
        action = torch.tensor([actions[i]], dtype=torch.float32)
        #print("STATE SHAPE: ", state.shape)
        #print("ACTION SHAPE: ", action.shape)
        result = model(state, action).cpu().detach().numpy()[0]
        print("STATE: ", state[0, 0:3])
        print("RESULT: ", result)
        state = torch.tensor([result], dtype=torch.float32)
        #state = torch.tensor([states[i+1]], dtype=torch.float32)

        res_x.append(result[0])
        res_y.append(result[1])
        res_z.append(result[2])
        results.append(result)

    results = np.array(results)
    ax.plot(res_x, res_y, res_z, label="Predicted Path")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title("Position Prediction with State Replacement")
    ax.legend()
    plt.show()

    fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6)
    ax1.plot(x)
    ax1.plot(res_x)
    ax1.set_title("X")
    ax2.plot(y)
    ax2.plot(res_y)
    ax2.set_title("Y")
    ax3.plot(z)
    ax3.plot(res_z)
    ax3.set_title("Z")
    ax4.plot(states[:, 3])
    ax4.plot(results[:, 3])
    ax4.set_title("X Ang")
    ax5.plot(states[:, 4])
    ax5.plot(results[:, 4])
    ax5.set_title("Y Ang")
    ax6.plot(states[:, 5])
    ax6.plot(results[:, 5])
    ax6.set_title("Z Ang")
    ax7.plot(states[:, 6])
    ax7.plot(results[:, 6])
    ax7.set_title("X Vel")
    ax8.plot(states[:, 7])
    ax8.plot(results[:, 7])
    ax8.set_title("Y Vel")
    ax9.plot(states[:, 8])
    ax9.plot(results[:, 8])
    ax9.set_title("Z Vel")
    ax10.plot(states[:, 9])
    ax10.plot(results[:, 9])
    ax10.set_title("X Ang Vel")
    ax11.plot(states[:, 10])
    ax11.plot(results[:, 10])
    ax11.set_title("Y Ang Vel")
    ax12.plot(states[:, 11])
    ax12.plot(results[:, 11])
    ax12.set_title("Z Ang Vel")
    plt.show()

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    #ax1.scatter([0.02 * i for i in range(len(actions[:, 0])*20)], actions[:, 0:20])
    #ax2.scatter([0.02 * i for i in range(len(actions[:, 1])*20)], actions[:, 20:40])
    #ax3.scatter([0.02 * i for i in range(len(actions[:, 2])*20)], actions[:, 40:60])
    #ax4.scatter([0.02 * i for i in range(len(actions[:, 3])*20)], actions[:, 60:])
    #plt.show()

def plot_all_data():
    data_filepaths = []
    for i in ['0.15', '0.16', '0.17', '0.18', '0.19']:
        for j in range(19):
            data_filepaths.append("../data/Bag1_start" + i + "_" + str(j) + ".npz")
        for j in range(19):
            data_filepaths.append("../data/Bag2_start" + i + "_" + str(j) + ".npz")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in data_filepaths:
        try:
            npz_data = np.load(i)
        except:
            continue
        states = npz_data['data'][:, :12]
        actions = npz_data['data'][:, 12:]
        traj_len = states.shape[0]

        # Plot the position
        print("Plotting...")
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        ax.plot(x, y, z, label="True Path")
    plt.show()


if __name__ == "__main__":
    state_dim = 12
    action_dim = 16 #4
    #model = AbsoluteDynamicsModel(state_dim, action_dim)
    #model.load_state_dict(torch.load('multistep_absolute_model.pt'))
    model = ResidualDynamicsModel(state_dim, action_dim)
    model.load_state_dict(torch.load('multistep_residual_model.pt'))
    model.eval()
    plot_all_data()
    plot_trajectory(model, "../data/Bag1_start0.15_6.npz")
    plot_trajectory(model, "../data/Bag1_start0.15_7.npz")
    plot_trajectory(model, "../data/Bag1_start0.15_8.npz")
    #plot_trajectory(model, CSV_FILENAME, TAG_FILENAME, [0,200], reset_state=True) #[10000,14000]