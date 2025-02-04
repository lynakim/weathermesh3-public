
set -e

# to be run from one folder up, ie bash realtime/run_backtest.sh
# this is really just a placeholder since we want this orchestrated through airflow
# airflow does exist but I haven't made the ui to trigger it easily

cycle_time="$1"
output_name="$2"

# get data
python3 -m realtime.get_ecmwf_hres "$cycle_time"
python3 -m realtime.get_operational_gfs "$cycle_time"

# run it
python3 -m realtime.run_rt_det "$1" --output "$output_name" --min-dt 6 --idempotent

# upload & process outputs
bash realtime/upload_outputs.sh --no-overwrite "$output_name"

mkdir -p "/fast/windborne/deep/ignored/evaluation/$output_name"
ln -sf "/fast/realtime/outputs/$output_name" "/fast/windborne/deep/ignored/evaluation/$output_name/outputs"
(cd ~/dlviz && python3 process_dl_output.py "$output_name")
(cd ~/dlviz && bash upload_viz_files.sh "$output_name")