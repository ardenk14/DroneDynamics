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
from rospy import Time
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
    print(f"Looking for all {suffix} files in {directory}:")
    for filename_arg in os.listdir(directory):
      f = os.path.join(directory, filename_arg)
      if os.path.isfile(f) and filename_arg[-4:] == suffix:
        print(f"Found {suffix} file: {f}")    
        filenames.append(filename_arg)
  
  return directory, filenames


def get_bagfile_csvs_directory(directory):
  # make directory for results
  results_dir = directory + "bagfile_csvs" 
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  return results_dir

def get_taggroup_dict(directory=None):
  # look for pickle file with taggroup_dict yaml file
  # This yaml file contains the taggroup_dict, which is a dictionary
  # with the following structure:
  # keys: list of all ids in the tag group
  # values: [x,y,z,qx,qy,qz,qw] list of Global position wrt. tag with id 0
  # try:
  #   pass
    # # read yaml file
    # with open(directory + "taggroup_dict.yaml", 'r') as stream:
    #   taggroup_dict_yaml = yaml.safe_load(stream)
    #   taggroup_dict = {}
    #   for group_dict in taggroup_dict_yaml["group_list"]:
    #     taggroup_dict[tuple(group_dict["id_list"])] = group_dict["position_orientation"]
    #   print(f"Loaded taggroup_dict from {directory}taggroup_dict.yaml")
    
  # except:
    # print(f"No taggroup_dict.yaml file found in {directory}")
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
  return taggroup_dict

def get_times_to_keep_dict(directory=None):
  # look for yaml file with the times to remove
  # this yaml file contains pairs of 
  # bagfile name : list of (start_time, end_time) tuples to remove from the bagfile
  try:
    # read yaml file
    with open(directory + "times_to_keep.yaml", 'r') as stream:
      times_to_keep_dict = yaml.safe_load(stream)
    
    # check if all bagfiles are in the times_to_keep_dict
    for filename in filenames:
      if filename not in times_to_keep_dict:
        print(f"WARNING: times_to_keep_dict does not contain {filename}")
      else:
        print(f"Found: {filename} in times_to_keep_dict")
  
  except:
    print(f"No times_to_keep.yaml file found in {directory}. Processing all times")
    times_to_keep_dict = {}

  return times_to_keep_dict

def write_bagfile_to_csv(directory, filename, csv_filepath, taggroup_dict, start_time=None, end_time=None):
  bag = rosbag.Bag(directory + filename)

  with open(csv_filepath, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['time','commands[0]','commands[1]','commands[2]','commands[3]',"position.x", "position.y", "position.z", "orient.x", "orient.y", "orient.z", "orient.w", "command_state_flag"])

    if start_time: start_time = Time.from_sec(float(start_time)) # / 1.0e9)
    if end_time: end_time = Time.from_sec(float(end_time))# / 1.0e9)
    
    for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections'], start_time=start_time, end_time=end_time):
      t = timestamp.to_sec()
      if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
  #       /sent_drone_commands  std_msgs/UInt8MultiArray
          d = message.data
          data_writer.writerow([t, d[0], d[1], d[2], d[3], "", "", "", "", "", "", "",0])
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
          if len(message.detections) == 0:
            data_writer.writerow([t, "", "", "", "", "", "", "", "", "", "", "",2])
          else:
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
              data_writer.writerow([t, "", "", "", "", avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6],1])
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
              data_writer.writerow([timestamp, "", "", "", "", avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6],1])
          
  bag.close()
  print(f"Wrote {csv_filepath}")


def process_all_bagfiles(directory, filenames, results_dir, times_to_keep_dict, taggroup_dict):
  for filename in filenames:

    if filename not in times_to_keep_dict:
      csv_filepath = (results_dir +"/"+filename[:-4]+'_raw_'+'alltimes.csv')
      print(f"Writing {csv_filepath}")
      write_bagfile_to_csv(directory, filename, csv_filepath, taggroup_dict, start_time=None, end_time=None)

    else:
      for [start_time, end_time] in times_to_keep_dict[filename]:
        csv_filepath = (results_dir +"/"+filename[:-4]+'_raw_'+
              str(start_time)+'_'+str(end_time)+'.csv')
        print(f"Writing {csv_filepath}")
        write_bagfile_to_csv(directory, filename, csv_filepath, taggroup_dict, start_time=start_time, end_time=end_time)

        

if __name__ == "__main__":
  directory, filenames = get_directory_and_filenames(".bag")
  results_dir = get_bagfile_csvs_directory(directory)
  times_to_keep_dict = get_times_to_keep_dict(directory)
  taggroup_dict = get_taggroup_dict(directory)
  process_all_bagfiles(directory, filenames, results_dir, times_to_keep_dict, taggroup_dict)





