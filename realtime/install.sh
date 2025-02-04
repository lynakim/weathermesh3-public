#!/bin/bash

set -e

# get models
download_file() {
    local dir=$1
    local file=$2
    local full_path=$3

    mkdir -p "$(dirname "$full_path")"

    local url="https://a.windbornesystems.com/dlnwp/$dir/$file"

    if [ ! -f "$full_path" ]; then
        echo "Downloading $full_path from $url"
        curl -f -u deploy:dcCTfKXEN7cPXC2EHTLD -o "$full_path" "$url"
    else
        echo "File $full_path already exists."
    fi
}

download_ignored_file() {
    local dir=$1
    local file=$2
    local full_path="/fast/ignored/$dir/$file"
    download_file "$dir" "$file" "$full_path"
}

download_const_file() {
    local file=$1
    local full_path="/fast/consts/$file"
    download_file "consts" "$file" "$full_path"
}

download_huge_ignored_file() {
    local dir=$1
    local file=$2
    local full_path="/fast/ignored/$dir/$file"
    download_file "/huge/$dir" "$file" "$full_path"
}

# Setup python packages
sudo apt-get install -y python3-grib
pip3 install numpy==1.26.4 scipy==1.11.1 netCDF4==1.6.4 tqdm==4.65.0 icosphere==0.1.3 matplotlib==3.7.2 pygrib google-cloud-storage==2.13.0 tensorboard==2.14.0 xarray==2023.7.0 timm==0.9.5 GPUtil==1.4.0 psutil==5.9.0 boto3 torch==2.4.1 torchvision==0.19.1

# Needed for gpu
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Folder structure required for dlnwp scripts
mkdir -p /fast/windborne/deep/ignored
ln -sT /viz/evaluation /fast/windborne/deep/ignored/evaluation
mkdir -p /fast/windborne/deep/ignored/WeatherMesh/
mkdir -p /fast/realtime
ln -sT /viz/outputs/ /fast/realtime/outputs
mkdir -p /fast/windborne/deep/ignored/WeatherMesh/
mkdir -p /fast/realtime/outputs/WeatherMesh
ln -sT /fast/realtime/outputs/WeatherMesh /fast/windborne/deep/ignored/evaluation/WeatherMesh/outputs

download_ignored_file "runs/run_Dec28-neoquadripede_20231229-015824" "model_epoch5_iter547188_loss0.088.pt"
download_ignored_file "runs/run_Jan16-neocasio_20240116-183046" "model_epoch2_iter271192_loss0.051.pt"
download_ignored_file "runs/run_Feb16-widepony-reprise_20240217-022407" "model_epoch6_iter597582_loss0.060.pt"
download_ignored_file "runs_adapt/run_Jan23-schelling_20240123-193116" "model_epoch26_iter25196_loss0.068.pt"
download_ignored_file "runs_adapt/run_Feb1-legeh-911-reprise_20240201-133247" "model_epoch56_iter54392_loss0.057.pt"
download_ignored_file "runs_hres/run_May5-multivar_20240510-144336" "model_keep_step133500_loss0.144.pt"
download_ignored_file "runs_hres/run_Oct28-coolerpacific_20241028-114757" "model_step146000_loss0.229.pt"
download_huge_ignored_file "evaluation/rtyamahabachelor5_328M/weights" "resave_20240925-100145.pt"
download_huge_ignored_file "evaluation/rtyamahabachelor5_328M/weights" "resave_20241008-121536.pt"
download_ignored_file "hres" "bay1_interps_new.pickle"
download_ignored_file "hres" "bay1_statics_new.pickle"
download_ignored_file "hres" "bay1_shapes.pickle"
download_ignored_file "hres" "bay1_pts.pickle"
download_const_file "normalization.pickle"
download_const_file "land_mask.npy"
download_const_file "soil_type.npy"
download_const_file "topography.npy"
download_const_file "normalization_delta_6h_28.pickle"
download_const_file "normalization_delta_6h_28.pickle"
download_const_file "normalization_delta_1h_28.pickle"
download_const_file "normalization_delta_3h_28.pickle"
download_const_file "normalization_delta_24h_28.pickle"
download_const_file "normalization_delta_72h_28.pickle"
download_const_file "normalization_delta_gfs2era5_28.pickle"
download_const_file "bias_gfs_hres_era5.npz"
download_const_file "bias_gfs_hres_era5.npz"

download_ignored_file "elevation" "mn75.npy"
download_ignored_file "elevation" "mn30.npy"
download_ignored_file "modis" "2020.npy"
download_ignored_file "modis" "2020_small2.npy"

for i in {1..365}
do
   download_const_file "radiation_1/${i}.npy"

  for j in {0..23}
  do
     download_const_file "neoradiation_1/${i}_${j}.npy"
     download_const_file "solarangle_1/${i}_${j}.npy"
  done
done

#sudo cp dlnwp_rt.service /etc/systemd/system
#sudo systemctl enable dlnwp_rt
#sudo systemctl start dlnwp_rt

echo "Installation complete! Reboot instance to use nvidia-smi."
