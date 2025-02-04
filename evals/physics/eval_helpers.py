import numpy as np

def calculate_air_density(ds):
    """
    Calculate air density considering humidity.
    
    Args:
        ds (xarray.Dataset): Dataset containing temperature and specific_humidity
    
    Returns:
        xarray.DataArray: Air density in kg/m³
    """
    R_d = 287.1 # Dry gas constant (J/(kg⋅K))
    R_v = 461.5 # Water vapor gas constant (J/(kg⋅K))

    # Calculate virtual temperature effect
    virtual_temp_factor = (1 - ds['specific_humidity'] + ds['specific_humidity'] * R_d/R_v)
    
    # Calculate density (pressure in hPa, hence multiply by 100 to convert to Pa)
    return (ds.coords['level'] * 100) / (R_d * ds['temperature'] * virtual_temp_factor)

def calculate_layer_heights(ds):
    """
    Calculate the height of each pressure level layer.
    
    Args:
        ds (xarray.Dataset): Dataset containing air_density and level coordinates
    
    Returns:
        xarray.DataArray: Layer heights in meters
    """
    g=9.81

    # Calculate pressure differences between levels
    levels = ds.coords['level'].values
    delta_levels = np.diff(levels, prepend=0)
    ds.coords['delta_level'] = ('level', delta_levels)
    
    # Calculate height using hydrostatic equation, hPa -> Pa conversion
    return 100 * ds.coords['delta_level'] / (ds['air_density'] * g)

# %%
def calculate_cell_volumes(ds):
    """
    Calculate the volume of each grid cell using spherical coordinates.
    Assumes h << r and uniform grid spacing.
    
    Args:
        ds (xarray.Dataset): Dataset containing lat, lon coordinates and height
    
    Returns:
        xarray.DataArray: Cell volumes in m³
    
    Formula: V = r²(sin(lat₂)-sin(lat₁))(lon₂-lon₁)h(π/180)
    """
    r_earth = 6371000  # Earth's radius in meters
    
    # Get latitude and longitude spacings (assumes uniform grid)
    d_lat = abs(ds.coords['lat'][1] - ds.coords['lat'][0])
    d_lon = abs(ds.coords['lon'][1] - ds.coords['lon'][0])
    
    # Calculate latitudes for box edges
    lat_edges_lower = ds.coords['lat'] - d_lat/2
    lat_edges_upper = ds.coords['lat'] + d_lat/2
    
    # Calculate volume using the provided equation
    volumes = (r_earth**2 * 
              (np.sin(np.deg2rad(lat_edges_upper)) - np.sin(np.deg2rad(lat_edges_lower))) * 
              d_lon * 
              ds['height'] * 
              (np.pi/180))
    
    return volumes

def calculate_mass_flux(ds):
    """
    Calculate mass flux components and total.
    
    Args:
        ds (xarray.Dataset): Dataset containing air_density and wind components
    
    Returns:
        tuple: (mass_flux_u, mass_flux_v, mass_flux_total) as xarray.DataArrays
    """
    mass_flux_u = ds['air_density'] * ds['u_component_of_wind']
    mass_flux_v = ds['air_density'] * ds['v_component_of_wind']
    
    return mass_flux_u, mass_flux_v

def calculate_mass_flux_divergence(mass_flux_u, mass_flux_v, R=6371e3):
    """
    Calculate the divergence of mass flux with periodic boundaries on a spherical Earth.
    Uses centered differences and properly accounts for spherical geometry.
    
    Args:
        mass_flux_u (xarray.DataArray): Zonal mass flux (east-west direction)
        mass_flux_v (xarray.DataArray): Meridional mass flux (north-south direction)
        R (float): Radius of the Earth in meters
    
    Returns:
        xarray.DataArray: Mass flux divergence in kg/m³/s
    """
    # Convert latitudes to radians
    lat_radians = np.deg2rad(mass_flux_u['lat'])
    
    # Calculate centered differences with periodic boundaries
    # For zonal direction (u)
    u_right = mass_flux_u.roll(lon=-1, roll_coords=False)
    u_left = mass_flux_u.roll(lon=1, roll_coords=False)
    dlon = np.deg2rad(mass_flux_u.coords['lon'].diff(dim='lon').mean().values)
    
    # For meridional direction (v)
    v_up = mass_flux_v.roll(lat=-1, roll_coords=False)
    v_down = mass_flux_v.roll(lat=1, roll_coords=False)
    dlat = np.deg2rad(mass_flux_v.coords['lat'].diff(dim='lat').mean().values)
    
    # Calculate divergence components with proper scaling
    # Factor 1/cos(φ) * ∂/∂λ(u*cos(φ)) for zonal component
    u_div = (1 / (2 * R * dlon * np.cos(lat_radians))) * (
        u_right * np.cos(lat_radians) - u_left * np.cos(lat_radians)
    )
    
    # Factor 1/cos(φ) * ∂/∂φ(v*cos(φ)) for meridional component
    v_div = (1 / (2 * R * dlat * np.cos(lat_radians))) * (
        v_up * np.cos(lat_radians.shift(lat=-1, fill_value=np.nan)) - 
        v_down * np.cos(lat_radians.shift(lat=1, fill_value=np.nan))
    )
    
    return u_div + v_div

def calculate_density_tendency(ds, dt=6*3600, is_era=False):
    """
    Calculate the rate of change of air density using centered differences.
    
    Args:
        ds (xarray.Dataset): Dataset containing air_density
        dt (float): Time step in seconds between consecutive predictions
    
    Returns:
        xarray.DataArray: Air density tendency in kg/m³/s
    """
    # Get time differences in seconds
    # time_diffs = ds.prediction_timedelta.diff(dim='prediction_timedelta').astype('timedelta64[s]').values
    if is_era:
        density_forward = ds["air_density"].shift(time=-1)
        density_backward = ds["air_density"].shift(time=1)
    else:
        # Calculate centered differences
        density_forward = ds["air_density"].shift(prediction_timedelta=-1)
        density_backward = ds["air_density"].shift(prediction_timedelta=1)
    
    # Use actual time differences for each step
    tendency = 100 *(density_forward - density_backward) / (2 * dt)
    
    # Remove edge timesteps that don't have complete differences
    if is_era:
        tendency = tendency.isel(time=slice(1, -1))
    else:   
        tendency = tendency.isel(prediction_timedelta=slice(1, -1))
    return tendency


def analyze_mass_conservation(ds, is_era=False):
    """
    Perform complete mass conservation analysis on a dataset.
    
    Args:
        ds (xarray.Dataset): Input dataset containing required variables
        is_era (bool): Whether the dataset is ERA5
    Returns:
        xarray.Dataset: Dataset with added mass conservation variables
    """
    # Calculate basic quantities
    import time
    start = time.time()
    print("Calculating basic quantities")
    ds['air_density'] = calculate_air_density(ds)
    print(f"Time taken to calculate air density: {time.time() - start} seconds")
    start = time.time()
    ds.coords['height'] = calculate_layer_heights(ds)
    print(f"Time taken to calculate layer heights: {time.time() - start} seconds")
    start = time.time()
    ds.coords['volume'] = calculate_cell_volumes(ds)
    
    # Calculate mass
    ds['dry_air_mass'] = ds['air_density'] * ds['volume']
    print(f"Time taken to calculate dry air mass: {time.time() - start} seconds")
    start = time.time()
    # Calculate mass fluxes
    ds['mass_flux_u'], ds['mass_flux_v'] = calculate_mass_flux(ds)
    print(f"Time taken to calculate mass fluxes: {time.time() - start} seconds")
    start = time.time()
    # Calculate divergence
    ds['mass_flux_divergence'] = calculate_mass_flux_divergence(ds['mass_flux_u'], ds['mass_flux_v'])
    print(f"Time taken to calculate mass flux divergence: {time.time() - start} seconds")
    start = time.time()
    # Calculate tendency
    ds['air_density_tendency'] = calculate_density_tendency(ds, is_era)
    # Calculate continuity equation difference (where timesteps allow)
    # time_slice = slice(1, -1)  # Matching tendency calculation
    ds['continuity_error'] = abs(ds['air_density_tendency'] - ds['mass_flux_divergence'])
    print(f"Time taken to calculate continuity residual: {time.time() - start} seconds")

    return ds
