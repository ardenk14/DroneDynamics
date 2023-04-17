#!/usr/bin/env python
"""
Process a csv file containing April tag detections and states (position/orientation) and perform the following:
1) Find the velocity by taking the difference between state values 
2) Interpolate the states to get a state for each command


# USAGE EXAMPLE:
python3 process_csv.py csv_directory
python3 process_csv.py csv_directory csv_name1 csv_name2
(results will show up in folder next to the bagfile)
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
# import quaternion as quat

LARGE_TIMEGAP_THRESHHOLD = 0.1 # [s]

# get cvs directory
directory = sys.argv[1]
if not directory.endswith("/"):
  directory += "/"

# create list of .csv's to scan
filenames = []
if len(sys.argv) > 2:
  for filename_arg in sys.argv[2:]:
    if not filename_arg.endswith(".csv"):
      filename_arg = filename_arg + ".csv"
    
    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == 'csv':
      print(f"Found file: {f}")    
      filenames.append(filename_arg)
else:
  print(f"Reading all .csv files in {directory}")
  for filename_arg in os.listdir(directory):

    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.csv':
      print(f"Found csvfile: {f}")    
      filenames.append(filename_arg)

# make directory for results
results_dir = directory + "../processed_csvs"
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

for filename in filenames:
    df = pd.read_csv(filename)
    file_to_write = results_dir +"/processed_"+filename
    print("Adding velocity and interpolating the states in {file_to_write}")

    # Add velocity columns 
    # (angular velocity still in quaternion form)
    # (change time from nanoseconds to seconds)
    df['vel.x'] = df['position.x'].diff() / (df['time'].diff() * 1e-9)
    df['vel.y'] = df['position.y'].diff() / (df['time'].diff() * 1e-9)
    df['vel.z'] = df['position.z'].diff() / (df['time'].diff() * 1e-9)
    df['ang_vel.x'] = df['orient.x'].diff() / (df['time'].diff() * 1e-9)
    df['ang_vel.y'] = df['orient.y'].diff() / (df['time'].diff() * 1e-9)
    df['ang_vel.z'] = df['orient.z'].diff() / (df['time'].diff() * 1e-9)
    df['ang_vel.w'] = df['orient.w'].diff() / (df['time'].diff() * 1e-9)

    # Add large_timegap_flag column
    df['large_timegap_flag'] = df['time'].diff() > LARGE_TIMEGAP_THRESHHOLD * 1e9

    # Interpolate the states to get a state for each command
    # (commands are at faster rate than the states)
    # At each command, fill in the states by interpolating (by time) between the previous and next state
    # set the index to time
    df = df.set_index('time')
    df['position.x'] = df['position.x'].interpolate(method='index')
    df['position.y'] = df['position.y'].interpolate(method='index')
    df['position.z'] = df['position.z'].interpolate(method='index')
    df['orient.x'] = df['orient.x'].interpolate(method='index')
    df['orient.y'] = df['orient.y'].interpolate(method='index')
    df['orient.z'] = df['orient.z'].interpolate(method='index')
    df['orient.w'] = df['orient.w'].interpolate(method='index')
    df['vel.x'] = df['vel.x'].interpolate(method='index')
    df['vel.y'] = df['vel.y'].interpolate(method='index')
    df['vel.z'] = df['vel.z'].interpolate(method='index')
    df['ang_vel.x'] = df['ang_vel.x'].interpolate(method='index')
    df['ang_vel.y'] = df['ang_vel.y'].interpolate(method='index')
    df['ang_vel.z'] = df['ang_vel.z'].interpolate(method='index')
    df['ang_vel.w'] = df['ang_vel.w'].interpolate(method='index')

    # reset the index to be the row number
    df = df.reset_index()

    # convert quaternions to euler angles for orientation and angular velocity
    euler_orient = np.zeros((df.shape[0], 3))
    q = R.from_quat([df['qw_orientation'], df['qx_orientation'], df['qy_orientation'], df['qz_orientation']])
    euler_orient = q.as_euler('xyz') # yaw (Body-Z), pitch (Body-Y), rolls (Body-X) in the air
    # for i, row in df.iterrows():
    #     q = R.from_quat(row['qw_orientation'], row['qx_orientation'], row['qy_orientation'], row['qz_orientation'])
    #     euler_orient[i] = quat.euler_angles(q, 'xyz')

    euler_ang_vel = np.zeros((df.shape[0], 3))
    q2 = R.from_quat([df['qw_ang_vel'], df['qx_ang_vel'], df['qy_ang_vel'], df['qz_ang_vel']])
    euler_ang_vel = q2.as_euler('xyz')
    # for i, row in df.iterrows():
    #     q = quat.quaternion(row['qw_ang_vel'], row['qx_ang_vel'], row['qy_ang_vel'], row['qz_ang_vel'])
    #     euler_ang_vel[i] = quat.euler_angles(q, 'xyz')
    
    # Add Euler angles to dataframe
    df[['roll', 'pitch', 'yaw']] = pd.DataFrame(euler_orient, index=df.index)
    df[['ang_vel_roll', 'ang_vel_pitch', 'ang_vel_yaw']] = pd.DataFrame(euler_ang_vel, index=df.index)

    # Save the interpolated DataFrame back to a CSV file
    df.to_csv(file_to_write, index=False)
    print(f"Wrote {file_to_write}")