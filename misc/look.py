from utils import *
from dataloader import NeoDataConfig
from data import NeoDatasetConfig, NeoWeatherDataset
from model_latlon_3d import ForecastStepConfig, ForecastStepSwin3D
import matplotlib.pyplot as plt
import numpy as np
import torch


timesteps = [12,24]
dsc = NeoDatasetConfig(WEATHERBENCH=1)
data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                        timesteps=timesteps,
                                        requested_dates = get_dates((D(1997, 1, 1),D(2017, 12,1))),
                                        ))
model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh,patch_size=(4,8,8),hidden_dim=768,lat_compress=True,
                        timesteps=timesteps,processor_dt=12))




exit()
checkpoint_path = "ignored/runs/run_Nov21-12and24_bk/model_epoch5_iter328437_loss0.057.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)


