import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

"""
plot the tags.yaml file in 3D to make sure it corresponds to the real life positions of the tags
"""

def check_tags_yaml():
    with open('tags_right_wall.yaml', 'r') as f:
        data = yaml.safe_load(f)

    # Extract x,y,z location for each id number and plot it
    for bundle in data['tag_bundles']:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for tag in bundle['layout']:
            # fig.title(bundle['name'])
            id_num = tag['id']
            x = tag['x']
            y = tag['y']
            z = tag['z']
            ax.scatter(x, y,z)
            # plt.annotate(id_num, (x, y))

    # with open('test.yaml', 'r') as f:
    #     data = yaml.safe_load(f)
    #     layout_list = data['layout']

    # # ax = fig.add_subplot(projection='3d')

    # for tag in layout_list:
    #     id_num = tag['id']
    #     x = tag['x']
    #     y = tag['y']
    #     z = tag['z']
    #     ax.scatter(x, y,z)
    #     # ax.annotate(id_num, (x, y,z))
    # Show the plot
    plt.show()

if __name__ == "__main__":
    check_tags_yaml()