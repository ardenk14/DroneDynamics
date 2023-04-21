import check_tags_yaml
from scipy.spatial.transform import Rotation as R

"""
Create a tags.yaml file for the tags in our test space
"""
# YAML_FILE_DIR = r"C:\Users\arthu\OneDrive\Documents\Classwork\ROB498_Robot_Learning_for_Planning_and_Control\project\DroneDynamics\april_tags"
YAML_FILE_NAME = "./tags_right_wall.yaml"
TAG_SPACING_WITHIN_6GROUP = 0.0877 # meters
TAG_SPACING_WITHIN_2GROUP = 0.1341 # meters
TAG6_SIZE = 0.06
TAG2_SIZE = 0.0897
INCH_TO_METERS = 0.0254

# for fake generation
NUM_SHEETS_RIGHT_WALL_X = 4
NUM_SHEETS_RIGHT_WALL_Y = 4
NUM_TAGS_XAXIS_6GROUP = 3
NUM_TAGS_YAXIS_6GROUP = 2
SHEET_SPACING = 1.5

# key: lowest tag id in group, value: location of lowest tag in group
# x is horizontal, y is vertical, z is depth (assumed to be the same)

# contains all group tag locations
group_locations={
    '6tags_left':{
        102: [-43.5 , 27.5  , 13    ,0,0,1],
        126: [-43.5 , 51    , 12.75 ,0,0,1],
        132: [-43.5 , 71.25 , 12.5  ,0,0,1],
        138: [-45.5 , 39.75 , 28.25 ,0,0,1],
        108: [-45.5 , 60.75 , 28.25 ,0,0,1],
        144: [-45.5 , 29    , 41.5  ,0,0,1],
        114: [-45.5 , 50.5  , 41.5  ,0,0,1],
        120: [-45.5 , 71.75 , 41.75 ,0,0,1],
    },
    '6tags_right':{
          0:  [0     , 0    , 0  ,1,0,0],
         12: [0     , 24   , 0  ,1,0,0],
         18: [0     , 47.5 , 0  ,1,0,0],
         24: [0     , 69.75, 0  ,1,0,0],
         30: [22.25 , 0    , 0  ,1,0,0],
         36: [22.25 , 69.35, 0  ,1,0,0],
         42: [22.25 , 47.25, 0  ,1,0,0],
         48: [22.25 , 23   , 0  ,1,0,0],
         54: [-18.5 , 0    , 0  ,1,0,0],
         60: [-18.5 , 24   , 0  ,1,0,0],
         66: [-17.5 , 46.75, 0  ,1,0,0],
         72: [-18.1 , 68.5 , 0  ,1,0,0],
         78: [-36   , 0    , 0  ,1,0,0],
         84: [-35.5 , 23.5 , 0  ,1,0,0],
         90: [-35   , 47.0 , 0  ,1,0,0],
         96: [-35.5 , 68.6 , 0  ,1,0,0]
    },
    '2tags_left':{
        538: [-43.5 , 20.25, 81   , 0,0,1],
        540: [-43.5 , 48   , 80.8 , 0,0,1],
        534: [-43.5 , 70.75, 81   , 0,0,1],
        544: [-43.5 , 35   , 99.3 , 0,0,1],
        542: [-43.5 , 57.98, 98.8 , 0,0,1],
    }
}

# convert fron inches to meters
for i in group_locations.keys():
    for j in group_locations[i].keys():
        group_locations[i][j] = [
            group_locations[i][j][0]*INCH_TO_METERS, # x (inch -> m)
            group_locations[i][j][1]*INCH_TO_METERS, # y (inch -> m)
            group_locations[i][j][2]*INCH_TO_METERS, # z (inch -> m)
            group_locations[i][j][3], # flag_x
            group_locations[i][j][4], # flag_y
            group_locations[i][j][5]] # flag_z

def quat_to_euler(orientation_quat):
    """ convert orientation from quaternion to euler angles"""
    orientation_euler = R.from_quat(orientation_quat)
    return orientation_euler

class AprilTag():
    def __init__(self, tag_id, size, location, orientation_quat):
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
    def __init__(self, tag0_id, tag0_pos, n_tags=6):
        self.tag0_id = tag0_id
        self.tag0_pos = tag0_pos[:3] # First three coordinates are positions (x,y,z).
        self.flags = tag0_pos[3:]    # Next three positions are flags for left/right wall.
        self.n_tags = n_tags
        self.tags = []
        
        # Checking if the flags correspond to the left 
        # wall or the right wall for quarternion assignment.
        if self.flags==[0,0,1]:
            self.orientation = [0.0, 0.70710678, 0.0, 0.70710678]
        else:
            self.orientation = [0.0, 0.0, 0.0, 1.0]

        if self.n_tags==6:
            positions = self.generate_6group_positions_all()
            tag_size = TAG6_SIZE
        elif self.n_tags==2:
            positions = self.generate_2group_positions()
            tag_size = TAG2_SIZE
        
        ids = self.generate_group_ids(tag0_id)
        orientations = self.generate_group_orientations()
        for i in range(n_tags):
            tag = AprilTag(ids[i],tag_size,positions[i],orientations[i])
            self.tags.append(tag)
    
    def generate_2group_positions(self):
        """
        Given the position of tag 0, generate the positions of the next tag in the group
        Sheets are laid out in the order: 
        0 1
        """
        positions = []
        flag_x,flag_y,flag_z = self.flags
        for i in range(self.n_tags):
            x = self.tag0_pos[0] + flag_x*(i % 2)*TAG_SPACING_WITHIN_6GROUP
            y = self.tag0_pos[1]
            z = self.tag0_pos[2] - flag_z*(i % 2)*TAG_SPACING_WITHIN_6GROUP
            positions.append([x,y,z])
        return positions

    def generate_6group_positions_all(self):
        """
        Given the position of tag 0, generate the positions of the other 5 tags in the group
        Sheets are laid out in the order: 
        0 1 2
        3 4 5
        """
        # TODO: incorporate orientation of tag0
        positions = []
        flag_x,flag_y,flag_z = self.flags
        
        for i in range(3):
            x = self.tag0_pos[0] + flag_x*(i % 3)*TAG_SPACING_WITHIN_6GROUP
            y = self.tag0_pos[1]
            z = self.tag0_pos[2] - flag_z*(i%3)*TAG_SPACING_WITHIN_6GROUP
            positions.append([x,y,z])
        for i in range(3):
            x = self.tag0_pos[0] + flag_x*(i % 3)*TAG_SPACING_WITHIN_6GROUP
            y = self.tag0_pos[1] - TAG_SPACING_WITHIN_6GROUP
            z = self.tag0_pos[2] - flag_z*(i%3)*TAG_SPACING_WITHIN_6GROUP
            positions.append([x,y,z])
        return positions
        
    def generate_group_ids(self, tag0_id):
        """
        Given the id of tag 0, generate the ids of the other (n_tags-1) tags in the group (by counting upwards)
        """
        ids = []
        for i in range(self.n_tags):
            ids.append(tag0_id + i)
        return ids
    def generate_group_orientations(self):
        """
        Generate the orientations of the tags in the group
        (assume all tags in the group have the same orientation)
        """
        orientations = []
        for i in range(self.n_tags):
            orientations.append(self.orientation)
        return orientations
    
    def generate_group6_orientations(self, orientation0=[0.0,0.0,0.0,1.0]):
        """
        Generate the orientations of the tags in the group
        (assume all tags in the group have the same orientation)
        """
        orientations = []
        for i in range(self.n_tags):
            orientations.append(orientation0)
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
    
def generate_tags_yaml(group_dicts=None,n_tags_list=[6,6,2]):
    """
    @args
    group_dicts: List of group dictionaries with their global 
                 positions and flags for updating group tags
    n_tags_list: The number of tags in each group corresponding
                 to the respective group dicts.
    Generate the tags.yaml file for the tags in our test space
    """
    tag_gen_params = [group_dicts,n_tags_list]
    print("finished generating tag groups, now writing to file")
    
    with open(YAML_FILE_NAME, 'w') as f:
        # write the header
        f.write("standalone_tags: []\n")
        f.write("tag_bundles:\n  [\n    {\n      name: all_bundles',\n      layout:\n        [\n")
        
        # Loop through all group lists and save in file
        for i in range(len(tag_gen_params[0])):
            group_dict = tag_gen_params[0][i]
            n_tags = tag_gen_params[1][i]
            tag_groups_list = []        
            for id in group_dict:
                tag_group = TagGroup(id, group_dict[id],n_tags=n_tags)
                tag_groups_list.append(tag_group)

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
    
    generate_tags_yaml(
        group_dicts = [
            group_locations["6tags_left"],
            group_locations["6tags_right"],
            group_locations["2tags_left"]],
        n_tags_list = [6,6,2])
    check_tags_yaml.check_tags_yaml()