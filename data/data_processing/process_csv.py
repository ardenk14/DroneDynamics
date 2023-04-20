#!/usr/bin/env python
"""
Process a csv file containing April tag detections and states (position/orientation) and perform the following:
1) Find the velocity by taking the difference between state values 
2) Interpolate the states to get a state for each command


# USAGE EXAMPLE:
python3 process_csv.py csv_directory
python3 process_csv.py csv_directory csv_name1 csv_name2
python3 process_csv.py ~/Downloads/bagfile_csvs

IDEAS:
What if we keep only the rows with states?
The high sample rate might be too fast (not enough change between samples)
What is the effect of cubic spline vs interpolation?
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

LARGE_TIMEGAP_THRESHHOLD = 0.1 # [s] (note that csvs are in nanoseconds by default)

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
    df = pd.read_csv(directory + filename)
    file_to_write = results_dir +"/processed_"+filename
    print(f"Adding velocity and interpolating the states in {file_to_write}")

    # # Add large_timegap_flag column TODO: Figure out a way to do this before interpolation
    # # make a df with only states
    # # df['large_timegap_flag'] = 0
    # df_states = df[['time', 'position.x', 'position.y', 'position.z', 'orient.x', 'orient.y', 'orient.z', 'orient.w']]
    # df_states = df_states.dropna()     # remove rows with no states
    # # add large_timegap_flag column, where flag is 1 if the time distance between current and previous state is greater than LARGE_TIMEGAP_THRESHHOLD
    # df_states['large_timegap_flag'] = (df_states['time'].diff() > LARGE_TIMEGAP_THRESHHOLD * 1e9).astype(int)
    # # join the large_timegap_flag column into the original df
    # df = df.join(df_states['large_timegap_flag'], on = 'time', how = 'left')
    # # df = df.merge(df_states[['time', 'large_timegap_flag']], on='time', how='left')
    
    # # remove every nth command (but leave the states)
    # n = 2
    # df = df[df.index % n == 0 and df['position.x'].notna()]
    
    # Interpolate the states to get a state for each command
    # (commands are at faster rate than the states)
    # At each command, fill in the states by interpolating (by time) between the previous and next state
    # set the index to time
    df = df.set_index('time')
    # df['position.x'] = df['position.x'].interpolate(method='index')
    # df['position.y'] = df['position.y'].interpolate(method='index')
    # df['position.z'] = df['position.z'].interpolate(method='index')
    # df['orient.x'] = df['orient.x'].interpolate(method='index')
    # df['orient.y'] = df['orient.y'].interpolate(method='index')
    # df['orient.z'] = df['orient.z'].interpolate(method='index')
    # df['orient.w'] = df['orient.w'].interpolate(method='index')
    # try interpolating using a cubic spline
    df['position.x'] = df['position.x'].interpolate(method='cubic')
    df['position.y'] = df['position.y'].interpolate(method='cubic')
    df['position.z'] = df['position.z'].interpolate(method='cubic')
    df['orient.x'] = df['orient.x'].interpolate(method='cubic')
    df['orient.y'] = df['orient.y'].interpolate(method='cubic')
    df['orient.z'] = df['orient.z'].interpolate(method='cubic')
    df['orient.w'] = df['orient.w'].interpolate(method='cubic')

    # # if doing this after adding vel columns:
    # df['vel.x'] = df['vel.x'].interpolate(method='index')
    # df['vel.y'] = df['vel.y'].interpolate(method='index')
    # df['vel.z'] = df['vel.z'].interpolate(method='index')
    # df['ang_vel.x'] = df['ang_vel.x'].interpolate(method='index')
    # df['ang_vel.y'] = df['ang_vel.y'].interpolate(method='index')
    # df['ang_vel.z'] = df['ang_vel.z'].interpolate(method='index')
    # df['ang_vel.w'] = df['ang_vel.w'].interpolate(method='index')
    df = df.reset_index() # reset the index to be the row number

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


    # convert quaternions to euler angles for orientation and angular velocity
    euler_orient = np.zeros((df.shape[0], 3))
    euler_ang_vel = np.zeros((df.shape[0], 3))

    # TODO: execute in batch (challenge is Nan values passed into quat)
    # q = R.from_quat([df['orient.x'], df['orient.y'], df['orient.z'], df['orient.w']])
    # q = R.from_quat(df[['orient.x', 'orient.y','orient.z','orient.w']].values)
    # # q2 = R.from_quat([df['ang_vel.x'], df['ang_vel.y'], df['ang_vel.z'], df['ang_vel.w']])
    # q2 = R.from_quat(df[['ang_vel.x', 'ang_vel.y','ang_vel.z','ang_vel.w']].values)
    # euler_orient = q.as_euler('xyz') # yaw (Body-Z), pitch (Body-Y), roll (Body-X) 
    for i, row in df.iterrows():
      if not pd.isna(row['orient.w']):
        q1 = R.from_quat([row['orient.x'], row['orient.y'], row['orient.z'], row['orient.w']])
        euler_orient[i] = q1.as_euler('xyz') # yaw (Body-Z), pitch (Body-Y), roll (Body-X) 
      if not pd.isna(row['ang_vel.w']) and not (row['ang_vel.x'] ==0 and row['ang_vel.y'] ==0 and row['ang_vel.z'] ==0 and row['ang_vel.w'] ==0):
        q2 = R.from_quat([row['ang_vel.x'], row['ang_vel.y'], row['ang_vel.z'], row['ang_vel.w']])
        euler_ang_vel[i] = q2.as_euler('xyz')# yaw (Body-Z), pitch (Body-Y), roll (Body-X)  
    
    # # Add Euler angles to dataframe
    df[['roll', 'pitch', 'yaw']] = pd.DataFrame(euler_orient, index=df.index)
    df[['ang_vel_roll', 'ang_vel_pitch', 'ang_vel_yaw']] = pd.DataFrame(euler_ang_vel, index=df.index)
    
    # # remove the columns with quaternion position and angular velocities
    # df = df.drop(columns=['orient.x', 'orient.y', 'orient.z', 'orient.w', 'ang_vel.x', 'ang_vel.y', 'ang_vel.z', 'ang_vel.w'])
    # move the flag to the last column


    # # Add a column that has a 1 if the NEXT row has a state, and 0 if it doesn't
    # # (ie. if the commands[0] is missing for the next row)
    # # This is so that when we drop the states, we'll still know what the original datapoints were
    # df['has_state'] = df['commands[0]'].notna().astype(int).shift(-1)

    # # get rid of any remaining rows containing Nans (these willl be the locations of states)
    df = df.dropna()

    # Save the interpolated DataFrame back to a CSV file
    df.to_csv(file_to_write, index=False)
    print(f"Wrote {file_to_write}")