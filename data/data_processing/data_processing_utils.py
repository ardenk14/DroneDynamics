import sys
import os

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