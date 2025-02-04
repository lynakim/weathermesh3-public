# this script will create file ignored/profile_output.nsys-rep. Download this file to your local machine (easy to do through vscode)
# Then, install Nvidia Nsight Systems on your local machine, and open the file in it. You should start to see traces from the program.

nsys profile --trace=cuda,nvtx --stats=true --force-overwrite true -o ignored/profile_output python3  train/prof.py



