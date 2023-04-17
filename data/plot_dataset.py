import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 


"""
Visualize the drone position and orientation data from a CSV file.
"""
CSV_FILENAME = "data.csv"

def plot_trajectory(filename):

    df = pd.read_csv(filename)
    # print(df.head())
    # print(df.columns)

    # Plot the position
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   
    ax.plot(df["position.x"], df["position.y"], df["position.z"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Position")

if __name__ == "__main__":
    plot_trajectory(CSV_FILENAME)
    plt.show()


