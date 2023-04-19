import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import yaml


"""
Visualize the drone position and orientation data from a CSV file.
usage: first change filename to plot
python3 plot_dataset.py
"""
# CSV_FILENAME = "/home/niksridhar/Downloads/processed_csvs/processed_tags1_distorted_commands_states.csv"
CSV_FILENAME = "../processed_tags3_right_wall_commands_states.csv"
TAG_FILENAME = "../../april_tags/tags_right_wall.yaml"

def plot_trajectory(filename, tags_filename, index_limit=None):
    df = pd.read_csv(filename)

    with open(tags_filename, 'r') as f:
        data = yaml.safe_load(f)

    
    # print(df.head())
    # print(df.columns)

    # Plot the position
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Extract x,y,z location for each id number and plot it
    for bundle in data['tag_bundles']:
        for tag in bundle['layout']:
            # fig.title(bundle['name'])
            id_num = tag['id']
            x = tag['x']
            y = tag['y']
            z = tag['z']
            ax.scatter(x, z, y, c='r', s=8)

    if index_limit:
        ax.plot(-1*df["position.x"][index_limit[0]:index_limit[1]], -1*df["position.z"][index_limit[0]:index_limit[1]], -1*df["position.y"][index_limit[0]:index_limit[1]])
    else:
        ax.plot(-1*df["position.x"], -1*df["position.z"], -1*df["position.y"])
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title("Position")
    plt.show()

if __name__ == "__main__":
    plot_trajectory(CSV_FILENAME, TAG_FILENAME, [10000,14000])



