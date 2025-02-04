# import xarray
# import matplotlib.pyplot as plt
# import matplotlib.colors
# import numpy as np
# import math
# from typing import Optional
# from IPython.display import HTML
# import matplotlib.animation as animation

# def select(
#     data: xarray.Dataset,
#     variable: str,
#     level: Optional[int] = None,
#     max_steps: Optional[int] = None
# ) -> xarray.DataArray:
#     data = data[variable]
#     if "batch" in data.dims:
#         data = data.isel(batch=0)
#     if max_steps is not None and "prediction_timedelta" in data.sizes and max_steps < data.sizes["prediction_timedelta"]:
#         data = data.isel(prediction_timedelta=range(0, max_steps))
#     if level is not None and "level" in data.coords:
#         data = data.sel(level=level)
#     return data

# def scale(
#     data: xarray.DataArray,
#     center: Optional[float] = None,
#     robust: bool = False,
# ) -> tuple[xarray.DataArray, matplotlib.colors.Normalize, str]:
#     vmin = float(np.nanpercentile(data, (2 if robust else 0)))
#     vmax = float(np.nanpercentile(data, (98 if robust else 100)))
#     if center is not None:
#         diff = max(vmax - center, center - vmin)
#         vmin = center - diff
#         vmax = center + diff
#     return (data, matplotlib.colors.Normalize(vmin, vmax),
#             ("RdBu_r" if center is not None else "viridis"))

# def plot_data(
#     data: dict[str, tuple[xarray.DataArray, matplotlib.colors.Normalize, str]],
#     fig_title: str,
#     plot_size: float = 5,
#     robust: bool = False,
#     cols: int = 4
# ) -> HTML:
#     first_data = next(iter(data.values()))[0]
#     max_steps = first_data.sizes.get("prediction_timedelta", 1)
    
#     cols = min(cols, len(data))
#     rows = math.ceil(len(data) / cols)
#     figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
#     figure.suptitle(fig_title, fontsize=16)
#     figure.subplots_adjust(wspace=0.3, hspace=0.3)

#     images = []
#     for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
#         ax = figure.add_subplot(rows, cols, i+1)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(title)
        
#         # Handle the initial frame using prediction_timedelta
#         initial_data = plot_data.isel(prediction_timedelta=0).values
        
#         im = ax.imshow(
#             initial_data,
#             norm=norm,
#             origin="lower",
#             cmap=cmap
#         )
#         plt.colorbar(
#             mappable=im,
#             ax=ax,
#             orientation="vertical",
#             pad=0.02,
#             aspect=16,
#             shrink=0.75,
#             extend=("both" if robust else "neither")
#         )
#         images.append(im)

#     def update(frame):
#         timedelta = first_data["prediction_timedelta"].values[frame]
#         # Convert timedelta to hours for display
#         hours = timedelta / np.timedelta64(1, 'h')
#         figure.suptitle(f"{fig_title}, Forecast +{hours:.0f}h", fontsize=16)
            
#         for im, (plot_data, norm, cmap) in zip(images, data.values()):
#             frame_data = plot_data.isel(prediction_timedelta=frame).values
#             im.set_array(frame_data)

#     ani = animation.FuncAnimation(
#         fig=figure,
#         func=update,
#         frames=max_steps,
#         interval=250
#     )
#     plt.close(figure.number)
#     return HTML(ani.to_jshtml())



# def plot_vars_over_time(ds, vars):
#     """
#     Plot the evolution of a variable over time.
    
#     Args:
#         ds (xarray.Dataset): Dataset containing dry_air_mass
#         vars (list[str]): List of variables to plot
    
#     Returns:
#         tuple: (total_mass, percentage_change) as xarray.DataArrays
#     """
#     # Calculate total mass at each timestep
#     for var in vars:
#         summed_var = ds[var].sum(dim=['lat', 'lon', 'level'])
        
#         # Calculate percentage change from initial state
#         initial_summed_var = summed_var[1]
#         percentage_change = (summed_var - initial_summed_var) / initial_summed_var * 100
        
#         plt.plot(percentage_change)
#     plt.title('Percentage Changes Over Time')
#     plt.legend(vars)
#     plt.xlabel('Time')
#     plt.ylabel('Percentage Change')

#     plt.show()

import xarray
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math
from typing import Optional
from IPython.display import HTML
import matplotlib.animation as animation

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None,
    is_era: bool = False
) -> xarray.DataArray:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None:
        if is_era:
            data = data.isel(time=range(0, max_steps))
        else:
            data = data.isel(prediction_timedelta=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
    data: xarray.DataArray,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.DataArray, matplotlib.colors.Normalize, str]:
    vmin = float(np.nanpercentile(data, (2 if robust else 0)))
    vmax = float(np.nanpercentile(data, (98 if robust else 100)))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, tuple[xarray.DataArray, matplotlib.colors.Normalize, str]],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    is_era: bool = False
) -> HTML:
    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("prediction_timedelta", 1) if not is_era else first_data.sizes.get("time", 1)
    
    cols = min(int(cols), int(len(data)))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0.3, hspace=0.3)

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
        # Handle the initial frame using prediction_timedelta
        if not is_era:
            initial_data = plot_data.isel(prediction_timedelta=0).values
        else:
            initial_data = plot_data.isel(time=0).values
            if plot_data.lat[0] > plot_data.lat[-1]:
                initial_data = np.flipud(initial_data)
        
        im = ax.imshow(
            initial_data,
            norm=norm,
            origin="lower",
            cmap=cmap
        )
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            extend=("both" if robust else "neither")
        )
        images.append(im)

    def update(frame):
        if not is_era:
            timedelta = first_data["prediction_timedelta"].values[frame]
        else:
            timedelta = first_data["time"].values[frame]
        # Convert timedelta to hours for display
        if not is_era:
            hours = timedelta / np.timedelta64(1, 'h')
            figure.suptitle(f"{fig_title}, Forecast +{hours:.0f}h", fontsize=16)
 
        else:
            figure.suptitle(f"{fig_title}, Time {timedelta}:00", fontsize=16)
            
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            frame_data = plot_data.isel(prediction_timedelta=frame).values if not is_era else plot_data.isel(time=frame).values
            if is_era and plot_data.lat[0] > plot_data.lat[-1]:
                frame_data = np.flipud(frame_data)
            im.set_array(frame_data)

    ani = animation.FuncAnimation(
        fig=figure,
        func=update,
        frames=max_steps,
        interval=250
    )
    plt.close(figure.number)
    return HTML(ani.to_jshtml())



def plot_vars_over_time(ds, vars):
    """
    Plot the evolution of a variable over time.
    
    Args:
        ds (xarray.Dataset): Dataset containing dry_air_mass
        vars (list[str]): List of variables to plot
    
    Returns:
        tuple: (total_mass, percentage_change) as xarray.DataArrays
    """
    # Calculate total mass at each timestep
    for var in vars:
        summed_var = ds[var].sum(dim=['lat', 'lon', 'level'])
        
        # Calculate percentage change from initial state
        initial_summed_var = summed_var[1]
        percentage_change = (summed_var - initial_summed_var) / initial_summed_var * 100
        
        plt.plot(percentage_change)
    plt.title('Percentage Changes Over Time')
    plt.legend(vars)
    plt.xlabel('Time')
    plt.ylabel('Percentage Change')

    plt.show()

