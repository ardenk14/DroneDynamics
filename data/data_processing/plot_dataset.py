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

    # plot_apriltags(tags_filename, ax=ax)

    # trim the trajectory to contain only rows within index limits
    if not index_limit: index_limit = [0, len(df)]
    df = df.iloc[index_limit[0]:index_limit[1]]
    # reset index
    df = df.reset_index(drop=True)

    # plot the trajectory
    ax.plot(df["position.x"], df["position.y"], df["position.z"])

    # plot the start and end points
    ax.scatter(df["position.x"][0], df["position.y"][0], df["position.z"][0], c='g', s=10)
    ax.scatter(df["position.x"][len(df)-1], df["position.y"][len(df)-1], df["position.z"][len(df)-1], c='b', s=10)

    # At each point, plot a small coordinate frame indicating the orientation 
    print("Plotting Orientation...")
    # unpack the angular orientation, which is encoded as the first two columns of the rotation matrix
    col1 = np.expand_dims(df[["R11", "R21", "R31"]], axis=2) 
    col2 = np.expand_dims(df[["R12", "R22", "R32"]], axis=2) 
    # create col3 using cross product
    col3 = np.cross(col1, col2, axisa=1, axisb=1, axisc=1)
    # create the rotation matrix (N,3,3)
    orient = np.concatenate((col1, col2, col3), axis=2) #(N,3,3)
    # # check that the columns are orthogonal
    # for i in range(0,orient.shape[0], 100):
    #   print(np.dot(orient[i,:,0], orient[i,:,2].T))
    
    # was from euler state representation
    # try:
    #     orient = R.from_euler('xyz', df[["roll", "pitch", "yaw"]].values, degrees=False).as_dcm() #(N,3,3) # USE .as_dcm() instead of as_matrix() if one doesn't work
    # except AttributeError:
    #     orient = R.from_euler('xyz', df[["roll", "pitch", "yaw"]].values, degrees=False).as_matrix()
    origins = df[["position.x", "position.y", "position.z"]].values #(N,3)


    for i in range(0,len(df)):
        # only  plot orientation if the command_state_flag is 1 (tag detected)
        if df["command_state_flag"][i] == 1:
            ax.quiver(origins[i, 0], origins[i, 1], origins[i, 2], orient[i, :, 0], orient[i, :, 1], orient[i, :, 2], length=0.1, color=['r', 'g', 'b'], normalize=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Position")
    plt.show()

if __name__ == "__main__":
    # TODO: Make plotting functions more modular
    plot_trajectory(CSV_FILENAME, TAG_FILENAME, [1000,4000])



