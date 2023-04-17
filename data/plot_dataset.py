import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 


"""
Visualize the drone position and orientation data from a CSV file.
usage: first change filename to plot
python3 plot_dataset.py
"""
# CSV_FILENAME = "/home/niksridhar/Downloads/processed_csvs/processed_tags1_distorted_commands_states.csv"
CSV_FILENAME = "/home/niksridhar/Downloads/processed_csvs/processed_tags1_commands_states.csv"

def plot_trajectory(filename, index_limit=None):
    df = pd.read_csv(filename)
    # print(df.head())
    # print(df.columns)

    # Plot the position
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if index_limit:
        ax.plot(df["position.x"][index_limit[0]:index_limit[1]], df["position.y"][index_limit[0]:index_limit[1]], df["position.z"][index_limit[0]:index_limit[1]])
    else:
        ax.plot(df["position.x"], df["position.y"], df["position.z"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Position")
    plt.show()

if __name__ == "__main__":
    plot_trajectory(CSV_FILENAME, [10000,14000])



