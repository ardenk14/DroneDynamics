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
from data_processing_utils import get_directory_and_filenames
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import RotationSpline, Slerp

#OUTLIER_POSITION_THRESHHOLD = 10.0 #[m] from origin
#OUTLIER_ZSCORE_THRESHOLD = 3.0 #z-score

# Bag you are processing (used to name data files)
bag_name = "Bag1"
# Adjust to augment the data by determining what time in the trajectory to grab the first state from (also used to name data files)
# bag 1 range from 0.1-
start = 0.1


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
    (102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 
     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 
     130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 
     144, 145, 146, 147) : [-1.1049, 0.6985, 0.3302, 0.0, 0.70710678, 0.0, 0.70710678],
    (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
    97, 98, 99, 100, 101, 6, 7, 8, 9, 10, 11) : [0,0,0,0,0,0,1]
  }
  return taggroup_dict

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

def plot_3d_axes(ax, rotation_matrix, origin=[0, 0, 0], axis_length=1.0):
    for i in range(len(rotation_matrix)):
      # Extract the x, y, and z unit vectors from the rotation matrix
      x_unit = rotation_matrix[i][:, 0]
      y_unit = rotation_matrix[i][:, 1]
      z_unit = rotation_matrix[i][:, 2]

      # Plot the x, y, and z axes as 3D lines from the origin
      ax.quiver(origin[0][i], origin[1][i], origin[2][i], x_unit[0], x_unit[1], x_unit[2], color='r', label='X', length=axis_length)
      ax.quiver(origin[0][i], origin[1][i], origin[2][i], y_unit[0], y_unit[1], y_unit[2], color='g', label='Y', length=axis_length)
      ax.quiver(origin[0][i], origin[1][i], origin[2][i], z_unit[0], z_unit[1], z_unit[2], color='b', label='Z', length=axis_length)

def in_range(t, times, forward_offset=0.0):
  for i in range(len(times)):
    if t >= times[i][0] and t <= times[i][-1] + forward_offset:
      return i
  return None

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
    
    br = CvBridge()
    pos_x = []
    pos_y = []
    pos_z = []
    tim = []
    images = []
    rots = []
    d_pos = []
    pos_x_inner = []
    pos_y_inner = []
    pos_z_inner = []
    time_inner = []
    img_inner = []
    rot_inner = []
    too_large = True
    in_path=False
    d_t = []
    last_pos = None
    last_rot = None
    cnt = 0

    for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections', '/tag_detections_image'], start_time=start_time, end_time=end_time):
      t = timestamp.to_sec()
      #if t >= 1681856800:
      #  break
      if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
        pass
#       /sent_drone_commands  std_msgs/UInt8MultiArray
        #d = message.data
        #print("CMD: ", d)
        #data_writer.writerow([t, d[0], d[1], d[2], d[3], "", "", "", "", "", "", "",0])
      elif topic=='/tag_detections' or topic=='tag_detections':
        print("TAGS Detected: " + str(t))
        if len(message.detections) == 0:
            data_writer.writerow([t, "", "", "", "", "", "", "", "", "", "", "",2])
        else:
          state_estimates = np.array([]) # each row is the [x,y,z,qx,qy,qx,qw] state estimate from one group
          group_cnt = 0
          for group in message.detections:
            # get arrays from message container
            group_global_pos = np.array(taggroup_dict[tuple(group.id)])[:3]
            group_global_orient = np.array(taggroup_dict[tuple(group.id)])[3:]
            pos = group.pose.pose.pose.position
            orient = group.pose.pose.pose.orientation

            pos_ = np.array([pos.z, pos.x, pos.y])
            orient_ = np.array([orient.x, orient.y, orient.z, orient.w])
            rotation_gp_wrt_global = R.from_quat(group_global_orient)

            # If you don't want to flip the orientation:
            pos_drone_wrt_global = group_global_pos + pos_
            rotation_drone_wrt_gp = R.from_quat(orient_)
            rotation_drone_wrt_global = rotation_drone_wrt_gp * rotation_gp_wrt_global
            orient_drone_wrt_global = rotation_drone_wrt_global.as_quat()
            print("Rotation: ", rotation_drone_wrt_global.as_matrix())
            rotation = rotation_drone_wrt_global.as_matrix() @ np.array([[-1, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, -1]])

            if last_pos is not None and np.linalg.norm(last_pos - pos_drone_wrt_global) < 0.25 and t - last_t <= 0.2:
              #d_pos.append(np.linalg.norm(last_pos - pos_drone_wrt_global))
              #d_t.append(t - last_t)
              #last_pos = pos_drone_wrt_global
              #last_t = t
              if too_large is True:
                pos_x_inner = []
                pos_y_inner = []
                pos_z_inner = []
                time_inner = []
                rot_inner = []
                too_large=False
                in_path = True
                pos_x_inner.append(last_pos[0])
                pos_y_inner.append(last_pos[1])
                pos_z_inner.append(last_pos[2])
                time_inner.append(last_t)
                rot_inner.append(last_rot)
              pos_x_inner.append(pos_drone_wrt_global[0])
              pos_y_inner.append(pos_drone_wrt_global[1])
              pos_z_inner.append(pos_drone_wrt_global[2])
              time_inner.append(t)
              rot_inner.append(rotation)
                #pos_x_inner.append(pos_drone_wrt_global[0])
              #pos_x.append(pos_drone_wrt_global[0])
              #pos_y.append(pos_drone_wrt_global[1])
              #pos_z.append(pos_drone_wrt_global[2])
            else:
              too_large = True
              if in_path and last_pos is not None:
                pos_x.append(pos_x_inner)
                pos_y.append(pos_y_inner)
                pos_z.append(pos_z_inner)
                tim.append(time_inner)
                rots.append(rot_inner)
                in_path = False

            #fig1 = plt.figure()
            #ax1 = fig1.add_subplot(projection='3d')
            #plot_apriltags("../../april_tags/tags_all.yaml", ax=ax1)
            #ax1.scatter(pos_x, pos_y, pos_z)
            #plt.show()

            if last_pos is None:
              last_pos = pos_drone_wrt_global
              last_t = t
              last_rot = rotation
            else:
              d_pos.append(np.linalg.norm(last_pos - pos_drone_wrt_global))
              if t - last_t < 1:
                d_t.append(t - last_t)
              last_pos = pos_drone_wrt_global
              last_t = t
              last_rot = rotation

            print("GROUP ", group_cnt, ": ", pos_)
            group_cnt += 1
            cnt += 1
      elif topic=='/tag_detections_image' or topic=='tag_detections_image':
        pass
        """print("Image: ", str(t))
        # TODO: ROS bridge to get image
        #if len(time_inner) != 0 and time_inner[-1] - t < 0.005:
        if t > 1681856294.26 and t < 1681856298.9:
          img = br.imgmsg_to_cv2(message)
          images.append(img)
          #cv2.imshow("image", img)
          #cv2.waitKey(1000)"""

  print("SECOND LOOP THROUGH")
  print("-------------------------------------------------------------------------------------------------------------")
  cmd_1 = [[] for i in range(len(pos_x))]
  cmd_2 = [[] for i in range(len(pos_x))]
  cmd_3 = [[] for i in range(len(pos_x))]
  cmd_4 = [[] for i in range(len(pos_x))]
  cmd_times = [[] for i in range(len(pos_x))]
  images = [[] for i in range(len(pos_x))]
  print("cmd list: ", cmd_1)

  for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections', '/tag_detections_image'], start_time=start_time, end_time=end_time):
      t = timestamp.to_sec()
      i = in_range(t, tim)
      j = in_range(t, tim, forward_offset=0.03)
      print("I: ", i)
      
      if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
        if i is not None:
  #       /sent_drone_commands  std_msgs/UInt8MultiArray
          d = message.data
          #print("CMD: ", d)
          cmd_1[i].append(d[0])
          cmd_2[i].append(d[1])
          cmd_3[i].append(d[2])
          cmd_4[i].append(d[3])
          cmd_times[i].append(t)
      elif topic=='/tag_detections_image' or topic=='tag_detections_image':
        if j is not None:
          #print("Image: ", str(t))
          img = br.imgmsg_to_cv2(message)
          images[j].append(img)

  # TODO: For each good path (say > 10 states), polyfit the path and sample states from 0.1 sec to the end every nth of a second
  final_state_times = [] # List showing the states for each accepted path
  firsts = [] # List showing the starting time for each accepted path
  final_states = []
  spacing = 0.2
  #start = 0.1
  for i in range(len(pos_x)):
    if len(pos_x[i]) >= 10:
      first = tim[i][0]
      firsts.append(first)
      final_state_times.append([i - first for i in tim[i]])

      # Perform linear regression using numpy's polyfit
      coefficients_x = np.polyfit(final_state_times[-1], pos_x[i], len(pos_x[i]))
      coefficients_y = np.polyfit(final_state_times[-1], pos_y[i], len(pos_y[i]))
      coefficients_z = np.polyfit(final_state_times[-1], pos_z[i], len(pos_z[i]))

      dx = np.polyder(coefficients_x, 1)
      dy = np.polyder(coefficients_y, 1)
      dz = np.polyder(coefficients_z, 1)

      
      stop = final_state_times[-1][-1]

      # TODO: Get angular position and velocity at the same times
      states_x = np.polyval(coefficients_x, np.arange(start, stop, spacing))
      states_y = np.polyval(coefficients_y, np.arange(start, stop, spacing))
      states_z = np.polyval(coefficients_z, np.arange(start, stop, spacing))
      states_dx = np.polyval(dx, np.arange(start, stop, spacing))
      states_dy = np.polyval(dy, np.arange(start, stop, spacing))
      states_dz = np.polyval(dz, np.arange(start, stop, spacing))

      orients = R.from_matrix(rots[i])
      spline = RotationSpline(final_state_times[-1], orients)
      angs = spline(np.arange(start, stop, spacing)).as_euler('XYZ')
      ang_vel = spline(np.arange(start, stop, spacing), 1)

      #print("ANG: ", angs)
      #print("ANG VEL: ", ang_vel)
      #print("STATES: ", states_x)

      final_state = np.vstack((states_x, states_y, states_z, states_dx, states_dy, states_dz, angs[:, 0], angs[:, 1], angs[:, 2], ang_vel[:, 0], ang_vel[:, 1], ang_vel[:, 2])).T
      print("STATES SHAPE: ", final_state.shape)
      print("STATES: ", final_state)
      final_states.append(final_state)
  
  # TODO: For each state you created, fit a poly to the commands leading to the next state and save coefficients as the action pair
  cnt = 0
  split_cmd_times = []
  split_cmd1 = []
  split_cmd2 = []
  split_cmd3 = []
  split_cmd4 = []
  for i in range(len(tim)):
    if tim[i][0] in firsts:
      print("SEQUENCE: ", len(tim[i]))
      stop = final_state_times[cnt][-1]
      reconstructed_times = np.arange(start, stop, spacing) + firsts[cnt]
      rt_one = reconstructed_times[0:-1].T
      rt_two = reconstructed_times[1:].T
      reconstructed_times = np.vstack((rt_one, rt_two)).T
      print("RT SHAPE: ", reconstructed_times.shape)
      last_step = -1
      all_steps = []
      all_cmd_times = []
      all_cmd1 = []
      all_cmd2 = []
      all_cmd3 = []
      all_cmd4 = []
      for j in range(len(cmd_times[i])):
        step = in_range(cmd_times[i][j], reconstructed_times)
        if step is not None:
          #print("STEP: ", step)
          if step != last_step:
            all_steps.append([])
            all_cmd_times.append([])
            all_cmd1.append([])
            all_cmd2.append([])
            all_cmd3.append([])
            all_cmd4.append([])
          all_steps[-1].append(step)#cmd_times[i][j])
          all_cmd_times[-1].append(cmd_times[i][j])
          all_cmd1[-1].append(cmd_1[i][j])
          all_cmd2[-1].append(cmd_2[i][j])
          all_cmd3[-1].append(cmd_3[i][j])
          all_cmd4[-1].append(cmd_4[i][j])
        last_step = step
      print("ALL STEPS: ", all_steps)
      split_cmd_times.append(all_cmd_times)
      split_cmd1.append(all_cmd1)
      split_cmd2.append(all_cmd2)
      split_cmd3.append(all_cmd3)
      split_cmd4.append(all_cmd4)
      cnt += 1

  actions = []
  # TODO: Now go through all the command sequences and perform least squares (Plot first to see how few variables you can reduce it to)
  for i in range(len(split_cmd_times)):
    actions.append([])
    for j in range(len(split_cmd_times[i])):
      #actions[-1].append([])
      base_t = split_cmd_times[i][j][0]
      times = [k - base_t for k in split_cmd_times[i][j]]
      #fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
      #ax1.scatter(times, split_cmd1[i][j], s=1)
      #ax2.scatter(times, split_cmd2[i][j], s=1)
      #ax3.scatter(times, split_cmd3[i][j], s=1)
      #ax4.scatter(times, split_cmd4[i][j], s=1)

      coeff_1 = np.polyfit(times, split_cmd1[i][j], 5)
      coeff_2 = np.polyfit(times, split_cmd2[i][j], 5)
      coeff_3 = np.polyfit(times, split_cmd3[i][j], 5)
      coeff_4 = np.polyfit(times, split_cmd4[i][j], 5)

      actions[-1].append(list(coeff_1) + list(coeff_2) + list(coeff_3) + list(coeff_4))

      #fitted_line_1 = np.round(np.polyval(coeff_1, times))
      #fitted_line_2 = np.round(np.polyval(coeff_2, times))
      #fitted_line_3 = np.round(np.polyval(coeff_3, times))
      #fitted_line_4 = np.round(np.polyval(coeff_4, times))

      #ax1.plot(times, fitted_line_1, 'orange', label='Fitted Line')
      #ax2.plot(times, fitted_line_2, 'orange', label='Fitted Line')
      #ax3.plot(times, fitted_line_3, 'orange', label='Fitted Line')
      #ax4.plot(times, fitted_line_4, 'orange', label='Fitted Line')

      #title = "Sequence " + str(i) + ", Step " + str(j)
      #fig1.suptitle(title)
      #plt.show()
  print("ACTIONS: ", actions)
        
      # Times for each sequence, take these and split according to times of arange above
      #cmd_times[i] 
      #firsts
  #for i in range(len(firsts)):
  #  stop = final_state_times[i][-1]
  #  reconstructed_times = np.arange(start, stop, spacing) + firsts[i]
  #  print("reconstructed: ", reconstructed_times)

  print("ACT Shape: ", np.array(actions[0]).shape)
  print("State shape: ", final_states[0].shape)

  #final_input = []
  #bag_name = "bag1"
  for i in range(len(split_cmd_times)):
    # Get rid of trajectory of thrust at the start (y intercept) is basically zero because it is either just sitting there or being carried
    if actions[i][0][5] <= 5.0:
      continue
    # Combine state and action into one long array
    traject_actions = np.array(actions[i])
    actions_set = np.zeros((traject_actions.shape[0]+1, traject_actions.shape[1]))
    actions_set[:-1, :] = traject_actions
    current_input = np.hstack((final_states[i], actions_set))#[list(final_states[i]) + actions[i]]
    print("CURRENT INPUT: ", current_input.shape)
    #current_input = [list(states_x[i]) + list(states_y[i]) + list(states_z[i]) + list(states_dx[i]) + list(states_dy[i]) + list(states_dz[i])+ actions[i]]
    # TODO: Write this line to a data file with first n=12 indices being state and the rest n=24 being action
    title = "../" + bag_name + "_" + "start" + str(start) + "_" + str(i)
    np.savez_compressed(title, data=current_input)
    #final_input.append(current_input)


  
  for i in range(len(pos_x)):
    if len(pos_x[i]) == 20:
      base_t = cmd_times[i][0]
      cmd_times[i] = [j - base_t for j in cmd_times[i]]
      fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
      ax1.scatter(cmd_times[i], cmd_1[i], s=1)
      ax2.scatter(cmd_times[i], cmd_2[i], s=1)
      ax3.scatter(cmd_times[i], cmd_3[i], s=1)
      ax4.scatter(cmd_times[i], cmd_4[i], s=1)

      coeff_1 = np.polyfit(cmd_times[i], cmd_1[i], 10)
      coeff_2 = np.polyfit(cmd_times[i], cmd_2[i], 10)
      coeff_3 = np.polyfit(cmd_times[i], cmd_3[i], 10)
      coeff_4 = np.polyfit(cmd_times[i], cmd_4[i], 10)

      print("1: ", coeff_1)
      print("2: ", coeff_2)
      print("3: ", coeff_3)
      print("4: ", coeff_4)

      fitted_line_1 = np.polyval(coeff_1, cmd_times[i])
      fitted_line_2 = np.polyval(coeff_2, cmd_times[i])
      fitted_line_3 = np.polyval(coeff_3, cmd_times[i])
      fitted_line_4 = np.polyval(coeff_4, cmd_times[i])

      ax1.plot(cmd_times[i], fitted_line_1, 'orange', label='Fitted Line')
      ax2.plot(cmd_times[i], fitted_line_2, 'orange', label='Fitted Line')
      ax3.plot(cmd_times[i], fitted_line_3, 'orange', label='Fitted Line')
      ax4.plot(cmd_times[i], fitted_line_4, 'orange', label='Fitted Line')

      """sample_rate = 10
      sub_t = [cmd_times[i][j] for j in range(0, len(cmd_times[i]), sample_rate)]
      sub_1 = [cmd_1[i][j] for j in range(0, len(cmd_1[i]), sample_rate)]
      sub_2 = [cmd_2[i][j] for j in range(0, len(cmd_1[i]), sample_rate)]
      sub_3 = [cmd_3[i][j] for j in range(0, len(cmd_1[i]), sample_rate)]
      sub_4 = [cmd_4[i][j] for j in range(0, len(cmd_1[i]), sample_rate)]
      ax1.scatter(sub_t, sub_1, s=1, c='r')
      ax2.scatter(sub_t, sub_2, s=1, c='r')
      ax3.scatter(sub_t, sub_3, s=1, c='r')
      ax4.scatter(sub_t, sub_4, s=1, c='r')"""
      plt.show()
          
  bag.close()
  cv2.destroyAllWindows()
  print("CMDS: ", [len(i) for i in cmd_1])
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  #print(np.array(pos_x).shape)
  #print(np.array(pos_y).shape)
  #print(np.array(pos_z).shape)
  for i in range(len(pos_x)):
    if len(pos_x[i]) == 20:
      print(len(pos_x[i]))
      #ax.plot(pos_x[i], pos_y[i], pos_z[i])
      plot_3d_axes(ax, rots[i], origin=[pos_x[i], pos_y[i], pos_z[i]], axis_length=0.1)
  plot_apriltags("../../april_tags/tags_all.yaml", ax=ax)
  plt.show()
  fig2, ((ax2, ax3, ax4), (ax5, ax6, ax7)) = plt.subplots(2, 3)
  #ax2, ax3, ax4 = fig2.add_subplots((1, 3))
  for i in range(len(pos_x)):
    print(len(pos_x[i]))
    if len(pos_x[i]) == 20:
      print("LENGTH IMAGES: ", len(images))
      print("LENGTH ROTATIONS: ", len(rots[i]))
      print(tim[i])
      print(pos_z[i])
      first = tim[i][0]
      tim[i] = [i - first for i in tim[i]]
      ax2.scatter(tim[i], pos_x[i])
      ax3.scatter(tim[i], pos_y[i])
      ax4.scatter(tim[i], pos_z[i])
      # Perform linear regression using numpy's polyfit
      coefficients_x = np.polyfit(tim[i], pos_x[i], len(pos_x[i])+1)
      coefficients_y = np.polyfit(tim[i], pos_y[i], len(pos_y[i])+1)
      coefficients_z = np.polyfit(tim[i], pos_z[i], len(pos_z[i])+1)

      dx = np.polyder(coefficients_x, 1)
      dy = np.polyder(coefficients_y, 1)
      dz = np.polyder(coefficients_z, 1)
      #slope, intercept = coefficients

      # Generate the fitted line
      fitted_line_x = np.polyval(coefficients_x, tim[i])
      fitted_line_y = np.polyval(coefficients_y, tim[i])
      fitted_line_z = np.polyval(coefficients_z, tim[i])

      line_dx = np.polyval(dx, tim[i])
      line_dy = np.polyval(dy, tim[i])
      line_dz = np.polyval(dz, tim[i])

      # Plot the data points and the fitted line
      ax2.plot(tim[i], fitted_line_x, 'orange', label='Fitted Line')
      ax3.plot(tim[i], fitted_line_y, 'orange', label='Fitted Line')
      ax4.plot(tim[i], fitted_line_z, 'orange', label='Fitted Line')
      ax5.plot(tim[i], line_dx, 'orange', label='Fitted Line')
      ax6.plot(tim[i], line_dy, 'orange', label='Fitted Line')
      ax7.plot(tim[i], line_dz, 'orange', label='Fitted Line')

  #ax2.boxplot(d_t)#scatter([i for i in range(len(d_pos))], d_pos, s=3)
  plt.show()
  print(f"Wrote {csv_filepath}")

  # TODO: Loop through again and grab commands and images within times of tag detection
  # TODO: Plot 3d chart with axes next to images
  # TODO: Investigate commands and frequency and change maxes

        

if __name__ == "__main__":
  taggroup_dict = get_taggroup_dict()
  write_bagfile_to_csv("/home/ardenk14/labeled_drone_bags/", "tags2_right_wall.bag", "test.csv", taggroup_dict)





