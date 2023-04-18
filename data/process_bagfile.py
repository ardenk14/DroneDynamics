#!/usr/bin/env python
# Process a ROS bag file with April tag detections into positional data for training

"""
Data collection process: 
1. Record flight data rosbag with topics:
    /clock
    /image_stream/camera_info
    /image_stream/image
    /sent_drone_commands

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

OUTLIER_POSITION_THRESHHOLD = 10.0 #[m] from origin
# get bagfile directory
directory = sys.argv[1]
if not directory.endswith("/"):
  directory += "/"

# create list of bagfiles to scan
filenames = []
if len(sys.argv) > 2:
  for filename_arg in sys.argv[2:]:
    if not filename_arg.endswith(".bag"):
      filename_arg = filename_arg + ".bag"
    
    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.bag':
      print(f"Found bagfile: {f}")    
      filenames.append(filename_arg)
else:
  print(f"Reading all .bag files in {directory}")
  for filename_arg in os.listdir(directory):

    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.bag':
      print(f"Found bagfile: {f}")    
      filenames.append(filename_arg)


# make directory for results
results_dir = directory + "bagfile_csvs" #+ filename[:-4]
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

# keys: list of all ids in the tag group
# values: [x,y,z,qx,qy,qz,qw] list of Global position wrt. tag with id 0
taggroup_dict = {
  [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
  43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
  79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
  97, 98, 99, 100, 101] : [0,0,0,0,0,0,1]
}



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

          # state_estimates = [] # each row is the [x,y,z,qx,qy,qx,qw] state estimate from one group
          # for group in message.detections:
          #   group_global_coords = taggroup_dict[group.id]
          #   pos = group.pose.pose.pose.position
          #   orient = group.pose.pose.pose.orientation

          #   # flip the vector so it represents pose wrt apriltags as opposed to apriltag pose wrt camera
          #   pos.x = -1.0*pos.x
          #   pos.y = -1.0*pos.y
          #   pos.z = -1.0*pos.z
          #   rotation = R.from_quat([orient.x, orient.y, orient.z, orient.w])
          #   rotation_inv = rotation.inv()
          #   orientation = rotation_inv.as_quat()
          #   state_estimates.append([pos.x, pos.y, pos.z, orientation[0], orientation[1], orientation[2], orientation[3]])

          # filter outliers: take the average of the three groups that are closest together

          try: 
            # assume we only have one tag group
            # print(type(message.detections[0].pose.pose.pose.position))
            pos = message.detections[0].pose.pose.pose.position 
            orient = message.detections[0].pose.pose.pose.orientation
            
            # flip the vector so it represents pose wrt apriltags as opposed to apriltag pose wrt camera
            # as_matrix was first added in scipy version 1.4.0 (specifically gh-10979). In 1.2.1 the same functionality is called as_dcm
            pos.x = -1.0*pos.x
            pos.y = -1.0*pos.y
            pos.z = -1.0*pos.z
            rotation = R.from_quat([orient.x, orient.y, orient.z, orient.w])
            rotation_inv = rotation.inv()
            orientation = rotation_inv.as_quat()

            # filter outliers:
            if (pos.x**2 + pos.y**2 + pos.z**2 < OUTLIER_POSITION_THRESHHOLD**2):
              data_writer.writerow([timestamp, "", "", "", "",pos.x, pos.y, pos.z, orientation[0], orientation[1], orientation[2], orientation[3]])
          
          except IndexError:
            # sometimes length of detections is 0
            pass
            # when there is only 1 group, is it a list?
            # pos = message.pose.pose.pose.position
            # orient = message.pose.pose.pose.orientation

          
          # data_writer.writerow([timestamp, "", "", "", "",pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w])
  bag.close()
  print(f"Wrote {csv_filepath}")



# parameters for writerow function:
            # @param topics: list of topics or a single topic. if an empty list is given all topics will be read [optional]
          # @type  topics: list(str) or str
          # @param start_time: earliest timestamp of message to return [optional]
          # @type  start_time: U{genpy.Time}
          # @param end_time: latest timestamp of message to return [optional]
          # @type  end_time: U{genpy.Time}





