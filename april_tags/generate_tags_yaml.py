import check_tags_yaml
from scipy.spatial.transform import Rotation as R

"""
Create a tags.yaml file for the tags in our test space
"""
# YAML_FILE_DIR = r"C:\Users\arthu\OneDrive\Documents\Classwork\ROB498_Robot_Learning_for_Planning_and_Control\project\DroneDynamics\april_tags"
YAML_FILE_NAME = "./tags_right_wall.yaml"
TAG_SPACING_WITHIN_6GROUP = 0.0877 # meters
INCH_TO_METERS = 0.0254

# for fake generation
NUM_SHEETS_RIGHT_WALL_X = 4
NUM_SHEETS_RIGHT_WALL_Y = 4
NUM_TAGS_XAXIS_6GROUP = 3
NUM_TAGS_YAXIS_6GROUP = 2
SHEET_SPACING = 1.5

# key: lowest tag id in group, value: location of lowest tag in group
# x is horizontal, y is vertical, z is depth (assumed to be the same)

group_locations_left={
    102: [-43.5,27.5,13]
    126: [-43.5,51,12.75]
    132: [-45.5,71.25,12.5]
    138: [-45.5,39.75,28.25]
    108: [-45.5,60.75,28.25]
    144: [-45.5,29,41.5]
    114: [-45.5,50.5,41.5]
    120: [-45.5,71.75,41.75]
    538: [-43.5,20.25,81]
    540: [-43.5,48,80.8]
    534: [-43.5,70.75,81]
    544: [-43.5,35,99.30]
    542: [-43.5,57.98,80]
}
group_locations_dict = {
    0:  [0,0],
    12: [0,24],
    18: [0,47.5],
    24: [0,69.75],
    30: [22.25,0],
    36: [22.25,69.35],
    42: [22.25,47.25],
    48: [22.25,23],
    54: [-18.5,0],
    60: [-18.5,24],
    66: [-17.5,46.75],
    72: [-18.1,68.5],
    78: [-36,0],
    84: [-35.5,23.5],
    90: [-35,47.0],
    96: [-35.5,68.6]
}
# convert fron inches to meters
for i in group_locations_dict.keys():
    group_locations_dict[i] = [
        group_locations_dict[i][0]*INCH_TO_METERS,
        group_locations_dict[i][1]*INCH_TO_METERS]

for i in group_locations_left.keys():
    group_locations_left[i] = [
        group_locations_left[i][0]*INCH_TO_METERS,
        group_locations_left[i][1]*INCH_TO_METERS]

# while I don't have the actual tag locations, 
# I'm going to assume that the tag sheets are evenly spaced
def generate_fake_tag_locations():
    """
    Generate the (fake) locations of all the tags in the test space
    """
    for i in range(NUM_SHEETS_RIGHT_WALL_X*NUM_SHEETS_RIGHT_WALL_Y):
        x = (i % NUM_SHEETS_RIGHT_WALL_X) * SHEET_SPACING
        y = (i // NUM_SHEETS_RIGHT_WALL_Y) * SHEET_SPACING
        group_locations_dict[i*6] = [x,y,0]


def quat_to_euler(orientation_quat):
    """ convert orientation from quaternion to euler angles"""
    orientation_euler = R.from_quat(orientation_quat)
    return orientation_euler

class AprilTag():
    def __init__(self, tag_id, size, location, orientation_quat):
        pass
        self.id = tag_id
        self.size = size
        self.location = location # list [x,y,z]
        self.orientation_quat = orientation_quat # [x,y,z,w]
        self.orientation_euler = quat_to_euler(orientation_quat) # (roll, pitch, yaw)

    def generate_tag_dict(self):
        """
        generate a tag dictionary for the tags.yaml file of the form:
        {id: 0, size: 0.06, x: 0.0000, y: 0.0000, z: 0.0000, qw: 1.0000, qx: 0.0000, qy: 0.0000, qz: 0.0000}
        """
        tag_dict = {}
        tag_dict['id'] = self.id
        tag_dict['size'] = self.size
        tag_dict['x'] = self.location[0]
        tag_dict['y'] = self.location[1]
        tag_dict['z'] = self.location[2]
        tag_dict['qw'] = self.orientation_quat[3]
        tag_dict['qx'] = self.orientation_quat[0]
        tag_dict['qy'] = self.orientation_quat[1]
        tag_dict['qz'] = self.orientation_quat[2]
        return tag_dict
    
    def generate_tag_dict_str(self):
        """
        generate a tag dictionary for the tags.yaml file of the form:
        {id: 0, size: 0.06, x: 0.0000, y: 0.0000, z: 0.0000, qw: 1.0000, qx: 0.0000, qy: 0.0000, qz: 0.0000}
        """
        tag_dict = self.generate_tag_dict()
        # print each value to 4-decimal places
        tag_dict_str = "{"
        for key in tag_dict:
            if key == 'id':
                tag_dict_str += key + ": " + str(tag_dict[key]) + ", "
            else:
                tag_dict_str += key + ": " + "{:.4f}".format(tag_dict[key]) + ", "
        tag_dict_str = tag_dict_str[:-2] + "}"
        return tag_dict_str


class TagGroup():
    def __init__(self, tag0_id, tag0_pos):
        self.tag0_id = tag0_id
        self.tag0_pos = tag0_pos
        self.tags = []

        positions = self.generate_6group_positions(tag0_pos)
        ids = self.generate_6group_ids(tag0_id)
        orientations = self.generate_6group_orientations()
        for i in range(6):
            tag = AprilTag(ids[i], 0.06, positions[i], orientations[i])
            self.tags.append(tag)

    def generate_6group_positions(self, tag0_pos):
        """
        Given the position of tag 0, generate the positions of the other 5 tags in the group
        Sheets are laid out in the order: 
        0 1 2
        3 4 5
        """
        positions = []
        for i in range(3):
            x = tag0_pos[0] + (i % 3)*TAG_SPACING_WITHIN_6GROUP
            y = tag0_pos[1]
            z = 0.0
            positions.append([x,y,z])
        for i in range(3):
            x = tag0_pos[0] + (i % 3)*TAG_SPACING_WITHIN_6GROUP
            y = tag0_pos[1] - TAG_SPACING_WITHIN_6GROUP
            z = 0.0
            positions.append([x,y,z])
        return positions
        
    def generate_6group_ids(self, tag0_id):
        """
        Given the id of tag 0, generate the ids of the other 5 tags in the group (by counting upwards)
        """
        ids = []
        for i in range(6):
            ids.append(tag0_id + i)
        return ids

    def generate_6group_orientations(self):
        """
        Generate the orientations of the tags in the group
        (all tags in the group have the same orientation)
        """
        orientation = [0.0,0.0,0.0,1.0] # [x,y,z,w]
        orientations = []
        for i in range(6):
            orientations.append(orientation)
        return orientations
    
    def convert_to_str_no_formatting(self):
        """
        Print the tag group to the terminal
        """
        tag_group_str = ""
        for tag in self.tags:
            tag_str = tag.generate_tag_dict_str()
            tag_group_str = tag_group_str + f"          {tag_str},\n"
        tag_group_str = tag_group_str[:-2]

        return tag_group_str
    
def generate_tags_yaml():
    """
    Generate the tags.yaml file for the tags in our test space
    """
    tag_groups_list = []

    for id in group_locations_dict:
        tag_group = TagGroup(id, group_locations_dict[id])
        tag_groups_list.append(tag_group)
    print("finished generating tag groups, now writing to file")

    with open(YAML_FILE_NAME, 'w') as f:
        # write the header
        f.write("standalone_tags: []\n")
        f.write("tag_bundles:\n  [\n    {\n      name: 'right_wall_bundle',\n      layout:\n        [\n")

        for tag_group in tag_groups_list:
            tag_group_str = tag_group.convert_to_str_no_formatting()
            f.write(f"{tag_group_str},\n")

        # delete the last comma and newline
        f.seek(f.tell() - 2, 0)
        f.truncate()

        # write the footer
        f.write("\n        ]\n    }\n  ]\n")

        print(f"Successfully generated {YAML_FILE_NAME}")

if __name__ == "__main__":
    #generate_fake_tag_locations()
    generate_tags_yaml()
    check_tags_yaml.check_tags_yaml()
    #check_tags_yaml.check_tags_yaml(YAML_FILE_NAME)
