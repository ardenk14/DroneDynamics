#!/usr/bin/env python
# Process a ROS bag file with April tag detections into positional data for training

"""
Data collection process: 
1. Record flight data rosbag with topics:
    /clock
    /image_stream/camera_info
    /image_stream/image
    /sent_drone_commands

1.1. Watch the video and manually label which sections should be removed

2. Play back this rosbag but this time run the apriltags detector to get:
    /clock
    /tag_detections 
    /sent_drone_commands
(could merge setps 1 and 2)

3. Process this new rosbag with the below script, 
which converts the above two topics into a time-stamped csv file

4. Post-process the csv to remove sections of flight with no tag detections/during crashes
and interpolate values so each timepoint has both a 

Based on: http://wiki.ros.org/rosbag/Cookbook#Python

# USAGE EXAMPLE:
source devel_isolated/setup.bash  #to get apriltag message definitions
python3 process_bagfile.py bagfile_directory
python3 process_bagfile.py bagfile_directory bagfile_name1 bagfile_name2
python3 process_bagfile.py ~/Downloads
"""

import sys
import os
import csv
import rosbag
from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml

OUTLIER_POSITION_THRESHHOLD = 10.0 #[m] from origin
OUTLIER_ZSCORE_THRESHOLD = 3.0 #z-score

def get_directory_and_filenames(suffix=".bag"):
  # get bagfile directory
  directory = sys.argv[1]
  if not directory.endswith("/"):
    directory += "/"

  # create list of bagfiles to scan
  filenames = []
  if len(sys.argv) > 2:
    for filename_arg in sys.argv[2:]:
      if not filename_arg.endswith(suffix):
        filename_arg = filename_arg + suffix
      
      f = os.path.join(directory, filename_arg)
      if os.path.isfile(f) and filename_arg[-4:] == suffix:
        print(f"Found {suffix} file: {f}")    
        filenames.append(filename_arg)
  else:
    print(f"Reading all {suffix} files in {directory}")
    for filename_arg in os.listdir(directory):

      f = os.path.join(directory, filename_arg)
      if os.path.isfile(f) and filename_arg[-4:] == {suffix}:
        print(f"Found {suffix} file: {f}")    
        filenames.append(filename_arg)
  
  return directory, filenames

directory, filenames = get_directory_and_filenames(".bag")

# make directory for results
results_dir = directory + "bagfile_csvs" 
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

# look for pickle file with taggroup_dict yaml file
# This yaml file contains the taggroup_dict, which is a dictionary
# with the following structure:
# keys: list of all ids in the tag group
# values: [x,y,z,qx,qy,qz,qw] list of Global position wrt. tag with id 0
try:
  # read yaml file
  with open(directory + "taggroup_dict.yaml", 'r') as stream:
    taggroup_dict_yaml = yaml.safe_load(stream)
    taggroup_dict = {}
    for group_dict in taggroup_dict_yaml["group_list"]:
      taggroup_dict[tuple(group_dict["id_list"])] = group_dict["position_orientation"]
    print(f"Loaded taggroup_dict from {directory}taggroup_dict.yaml")
  
except:
  print(f"No taggroup_dict.yaml file found in {directory}")
  print("Loading default taggroup_dict in process_bagfile.py")
  # keys: list of all ids in the tag group
  # values: [x,y,z,qx,qy,qz,qw] list of Global position wrt. tag with id 0
  taggroup_dict = {
    (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
    97, 98, 99, 100, 101) : [0,0,0,0,0,0,1],
    (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
    97, 98, 99, 100, 101, 6, 7, 8, 9, 10, 11) : [0,0,0,0,0,0,1]
  }

# look for yaml file with the times to remove
# this yaml file contains pairs of 
# bagfile name : list of (start_time, end_time) tuples to remove from the bagfile
try:
  # read yaml file
  with open(directory + "times_to_remove.yaml", 'r') as stream:
    times_to_remove_dict = yaml.safe_load(stream)
  
  # check if all bagfiles are in the times_to_remove_dict
  for filename in filenames:
    if filename not in times_to_remove_dict:
      print(f"WARNING: times_to_remove_dict does not contain {filename}")
  
except:
  print(f"No times_to_remove.yaml file found in {directory}")
  print("Loading default times_to_remove_dict in process_bagfile.py")
  times_to_remove_dict = {}


###############################
## Process bagfiles ###########
###############################
for filename in filenames:
  bag = rosbag.Bag(directory + filename)
  csv_filepath = results_dir +"/"+filename[:-4]+'_commands_states.csv'
  print(f"Writing tag_detections and sent_drone_commands in {filename} to CSV")
  
  with open(csv_filepath, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['time','commands[0]','commands[1]','commands[2]','commands[3]',"position.x", "position.y", "position.z", "orient.x", "orient.y", "orient.z", "orient.w"])
    #TODO: what are drone_commands [0],[1],etc?
    t0 = -1
    for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections']):
      if t0 == -1:
        t0 = timestamp
      timestamp = (timestamp - t0) #convert to nanoseconds since bag start
      # print(f"topic: {topic}, time:{timestamp}")
      if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
  #       /sent_drone_commands  std_msgs/UInt8MultiArray
          d = message.data
          data_writer.writerow([timestamp, d[0], d[1], d[2], d[3], "", "", "", "", "", "", ""])
      elif topic=='/tag_detections' or topic=='tag_detections':
        # message contains an array of pose wrt each visible group
        # AprilTagDetectionArray
          # std_msgs/Header header
          # AprilTagDetection[] detections
            # int32[] id
            # float64[] size
            # geometry_msgs/PoseWithCovarianceStamped pose
            #   std_msgs/Header header
            #   geometry_msgs/PoseWithCovariance pose
            #     float64[36] covariance
            #     geometry_msgs/Pose pose
            #       geometry_msgs/Point position
            #       geometry_msgs/Quaternion orientation

          state_estimates = np.array([]) # each row is the [x,y,z,qx,qy,qx,qw] state estimate from one group
          for group in message.detections:
            # get arrays from message container
            group_global_pos = np.array(taggroup_dict[tuple(group.id)])[:3]
            group_global_orient = np.array(taggroup_dict[tuple(group.id)])[3:]
            pos = group.pose.pose.pose.position
            orient = group.pose.pose.pose.orientation

            # filter outliers
            if (pos.x**2 + pos.y**2 + pos.z**2 > OUTLIER_POSITION_THRESHHOLD**2):
              continue
            #### convert to global coordinates (use SE3 transforms)
            pos_ = np.array([pos.x, pos.y, pos.z])
            orient_ = np.array([orient.x, orient.y, orient.z, orient.w])
            rotation_gp_wrt_global = R.from_quat(group_global_orient)

            # If you don't want to flip the orientation:
            pos_drone_wrt_global = group_global_pos + pos_
            rotation_drone_wrt_gp = R.from_quat(orient_)
            rotation_drone_wrt_global = rotation_drone_wrt_gp * rotation_gp_wrt_global
            orient_drone_wrt_global = rotation_drone_wrt_global.as_quat()
            
            # # If you want to flip the orientation: 
            # pos_drone_wrt_global = group_global_pos - pos_
            # rotation_gp_wrt_drone = R.from_quat(orient_)
            # rotation_drone_wrt_gp = rotation_gp_wrt_drone.inv()
            # rotation_drone_wrt_global = rotation_drone_wrt_gp * rotation_gp_wrt_global
            # orient_drone_wrt_global = rotation_drone_wrt_global.as_quat()

            state = np.concatenate((pos_drone_wrt_global, orient_drone_wrt_global))

            if len(state_estimates) == 0:
              state_estimates = np.expand_dims(state,0)
            else:
              state_estimates = np.concatenate((state_estimates, state), axis=0)
          
          if len(state_estimates) == 0: continue
          elif len(state_estimates[:,0]) == 1:
            avg = state_estimates[0,:]
            data_writer.writerow([timestamp, "", "", "", "", avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6]])
          elif len(state_estimates[:,0]) > 1:
            # filter outliers: 
            # find the average and sd of all tag groups
            avg = np.mean(state_estimates, axis=0)
            std = np.std(state_estimates, axis=0)
            z_scores = (state_estimates - avg) / std

            # if the z-score is greater than the OUTLIER_ZSCORE_THRESHOLD, remove it
            outliers = np.where(np.abs(z_scores) > OUTLIER_ZSCORE_THRESHOLD)
            state_estimates = np.delete(state_estimates, outliers, axis=0)

            # then recalculate the average state for the remaining tag groups and write to file
            avg = np.mean(state_estimates, axis=0)
            data_writer.writerow([timestamp, "", "", "", "", avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6]])
          
  bag.close()
  print(f"Wrote {csv_filepath}")



# parameters for writerow function:
            # @param topics: list of topics or a single topic. if an empty list is given all topics will be read [optional]
          # @type  topics: list(str) or str
          # @param start_time: earliest timestamp of message to return [optional]
          # @type  start_time: U{genpy.Time}
          # @param end_time: latest timestamp of message to return [optional]
          # @type  end_time: U{genpy.Time}





