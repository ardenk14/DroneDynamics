import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import yaml
from scipy.spatial.transform import Rotation as R


"""
Visualize the drone position and orientation data from a CSV file.
usage: first change filename to plot
python3 plot_dataset.py
"""
# CSV_FILENAME = "/home/niksridhar/Downloads/processed_csvs/processed_tags1_distorted_commands_states.csv"
CSV_FILENAME = r"../processed_tags3_right_wallalltimes.csv"
TAG_FILENAME = r"../../april_tags/tags_right_wall.yaml" 

def plot_apriltags(tags_filename, ax=None, label=False):
    with open(tags_filename, 'r') as f:
        data = yaml.safe_load(f)
    
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    print("Plotting AprilTags...")
    # Extract x,y,z location for each id number and plot it
    for bundle in data['tag_bundles']:
        for tag in bundle['layout']:
            # fig.title(bundle['name'])
            id_num = tag['id']
            x = tag['x']
            y = tag['y']
            z = tag['z']
            ax.scatter(x, z, y, c='r', s=8)
            if label: ax.text(x, z, y, id_num, size=8, zorder=1, color='k')
    
def plot_trajectory(filename, tags_filename, index_limit=None, ax=None):
    df = pd.read_csv(filename)

    # Plot the position
    print("Plotting Trajectory...")
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    plot_apriltags(tags_filename, ax=ax)

    # trim the trajectory to contain only rows within index limits
    if not index_limit: index_limit = [0, len(df)]
    df = df.iloc[index_limit[0]:index_limit[1]]
    # reset index
    df = df.reset_index(drop=True)

    # df = df[index_limit[0]:index_limit[1]]

    # plot the trajectory
    ax.plot(df["position.x"], df["position.y"], df["position.z"])

    # plot the start and end points
    ax.scatter(df["position.x"][0], df["position.y"][0], df["position.z"][0], c='g', s=8)
    ax.scatter(df["position.x"][len(df)-1], df["position.y"][len(df)-1], df["position.z"][len(df)-1], c='b', s=8)

    # At each point, plot a small coordinate frame indicating the orientation 
    # use quiver to make a 3d arrow
    # ax.quiver(df["position.x"], df["position.y"], df["position.z"], df["orientation.x"], df["orientation.y"], df["orientation.z"], length=0.1, normalize=True)

    print("Plotting Orientation...")
    orient = R.from_euler('xyz', df[["roll", "pitch", "yaw"]].values, degrees=True).as_matrix() #(N,3,3)
    origins = df[["position.x", "position.y", "position.z"]].values #(N,3)
    for i in range(0,len(df)):
        # only  plot orientation if the command_state flag is 1
        if df["command_state"][i] == 1:
            ax.quiver(origins[i, 0], origins[i, 1], origins[i, 2], orient[i, :, 0], orient[i, :, 1], orient[i, :, 2], length=0.08, color=['r', 'g', 'b'], normalize=False)


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Position")
    plt.show()

if __name__ == "__main__":
    # TODO: Make plotting functions more modular
    plot_trajectory(CSV_FILENAME, TAG_FILENAME, [10000,14000])



