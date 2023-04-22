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
CSV_FILENAME = "../data/50hz_processed_tags2_right_wall1681856259.673429_1681856345.057893.csv" #../data/processed_tags3_right_wall_commands_states.csv"
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
                state = torch.tensor([[df['position.x'][i], df['position.y'][i], df['position.z'][i], df['vel.x'][i, df['vel.y'][i], df['vel.z'][i], df['ang_vel_x'][i], df['ang_vel_y'][i], df['ang_vel_z'][i], df['roll'][i], df['pitch'][i], df['yaw'][i]]]], dtype=torch.float32)

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
            ax.scatter(x, z, y, c='r', s=8)

    if index_limit:
        ax.plot(df["position.x"][index_limit[0]:index_limit[1]], df["position.z"][index_limit[0]:index_limit[1]], df["position.y"][index_limit[0]:index_limit[1]])
    else:
        ax.plot(df["position.x"], df["position.z"], df["position.y"])

    ax.plot([i[0] for i in results], [i[2] for i in results], [i[1] for i in results])
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title("Position")
    plt.show()

    print("RESULTS: ", results)

if __name__ == "__main__":
    state_dim = 12
    action_dim = 4
    #model = AbsoluteDynamicsModel(state_dim, action_dim)
    #model.load_state_dict(torch.load('multistep_absolute_model.pt'))
    model = ResidualDynamicsModel(state_dim, action_dim)
    model.load_state_dict(torch.load('multistep_residual_model.pt'))
    model.eval()
    plot_trajectory(model, CSV_FILENAME, TAG_FILENAME, [0,200], reset_state=False) #[10000,14000]