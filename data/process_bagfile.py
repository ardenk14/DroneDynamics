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
python3 process_bagfile.py bagfile_directory bagfile_name
python3 process_bagfile.py ~/Downloads flight_4_12_bag2.bag
(results will show up in folder next to the bagfile)
"""

import sys
import os
import csv
import rosbag
import rospy


filename = sys.argv[2]
directory = sys.argv[1]
print("Reading the rosbag file")
if not directory.endswith("/"):
  directory += "/"
extension = ""
if not filename.endswith(".bag"):
  extension = ".bag"
bag = rosbag.Bag(directory + filename + extension)

# Create directory with name filename (without extension)
results_dir = directory + "bagfile_csvs" #+ filename[:-4]
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

print("Writing drone tag_detections and sent_drone_commands to CSV")

with open(results_dir +"/"+filename+'_commands_states.csv', mode='w') as data_file:
  data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  data_writer.writerow(['time','commands[0]','commands[1]','commands[2]','commands[3]',"position.x", "position.y", "position.z", "orient.x", "orient.y", "orient.z", "orient.w"])
 #TODO: what are drone_commands [0],[1],etc?
        # @param topics: list of topics or a single topic. if an empty list is given all topics will be read [optional]
        # @type  topics: list(str) or str
        # @param start_time: earliest timestamp of message to return [optional]
        # @type  start_time: U{genpy.Time}
        # @param end_time: latest timestamp of message to return [optional]
        # @type  end_time: U{genpy.Time}
  for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections']):
    print(f"topic: {topic}, time:{timestamp}")
    if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
#       /sent_drone_commands  std_msgs/UInt8MultiArray
        d = message.data
        data_writer.writerow([timestamp, d[0], d[1], d[2], d[3], "", "", "", "", "", "", ""])
    elif topic=='/tag_detections' or topic=='tag_detections':
      # geometry_msgs/PoseWithCovarianceStamped
      #   std_msgs/Header header
      #   geometry_msgs/PoseWithCovariance pose
      #     float64[36] covariance
      #     geometry_msgs/Pose pose
      #       geometry_msgs/Point position
      #       geometry_msgs/Quaternion orientation
        pos = message.pose.pose.position
        orient = message.pose.pose.orientation
        data_writer.writerow([timestamp, "", "", "", "",pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w])




print("Finished creating csv file!")
bag.close()





