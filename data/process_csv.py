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
# import csv

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
    df = pd.read_csv(file_to_read)
    file_to_write = results_dir +"/processed_"+filename+extension
    print("Adding velocity and interpolating the states in {file_to_write}")

    # make dataframes containing only states and only commands
    df_states = df[['time', 'position.x', 'position.y', 'position.z', 'orient.x', 'orient.y', 'orient.z', 'orient.w']]
    df_commands = df[['time', 'commands[0]', 'commands[1]', 'commands[2]', 'commands[3]']]
    
    # remove any rows that only contain a timestamp and no other data
    # (keep these for states so interpolation can be done)
    df_commands.set_index('time', inplace=True) # Set the index of the DataFrame to the 'time' column
    for index, row in df_commands.iterrows():
        if len(row.dropna()) <= 1:
            df_commands.drop(index, inplace=True)
    df_commands.reset_index(inplace=True) # Reset the index to make 'time' a column again


    # go through and fill in all velocities for df_states
    # interpolation is better done for quaternions, so keep in quaternion form
    difference = df_states.diff()
    vx_col = 
    # treat this like excel where v[i][j] = p[i][j] - p[i][j-1]/t[j]


    # interpolate the states to all commands
    df_states.set_index('time', inplace=True) # Set the index of the DataFrame to the 'time' column
    df_states.interpolate(method='time', inplace=True) # Interpolate the missing values
    df_states.reset_index(inplace=True) # Reset the index to make 'time' a column again

    # convert to euler angles

    # Save the interpolated DataFrame back to a CSV file
    df_commands.to_csv(file_to_write, index=False)
    print(f"Wrote {file_to_write}")



# with open(file_to_write, mode='w') as write_file:
#     with open(file_to_read, mode='r') as read_file:
#         writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         reader = csv.reader(read_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)






        
#         for row in reader:
#             if row[0].isalnum(): # row is header: 
#                 # row = ['time','commands[0]','commands[1]','commands[2]','commands[3]',"position.x", "position.y", "position.z", "orient.x", "orient.y", "orient.z", "orient.w"]
#                 new_row = row + ['vel.x','vel.y','vel.z','ang_vel.x','ang_vel.y', 'ang_vel.z', 'large_timegap_flag']
#                 writer.writerow(new_row)
#             else: # row is data
#                 # 
#                 t = row[0] / 1e9 #[s] (time is originally in microseconds since some arbitrary start point)

#                 if row[5] == "": #this was a command message
#                     # go back in the file to get 
#                     pass

#                 writer.writerow(row)