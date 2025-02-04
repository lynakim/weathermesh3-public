# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def analyze_continuity_components(ds, is_era=False):
    """Analyze components that contribute to continuity error."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate vertical sums (mass in each column)
    density_tendency_cols = ds['air_density_tendency'].sum(dim='level')
    divergence_cols = ds['mass_flux_divergence' if not is_era else 'divergence'].sum(dim='level')
    continuity_error_cols = ds['continuity_error'].sum(dim='level')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Continuity Error Analysis', fontsize=16)
    
    # Plot 1: Time series of global means
    axes[0,0].plot(density_tendency_cols.mean(dim=['lat', 'lon']), label='Density Tendency')
    axes[0,0].plot(divergence_cols.mean(dim=['lat', 'lon']), label='Mass Flux Divergence')
    axes[0,0].plot(continuity_error_cols.mean(dim=['lat', 'lon']), label='Continuity Error')
    axes[0,0].set_title('Global Mean Time Series')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].legend()
    
    # Plot 2: Level-wise analysis for a single timestep
    mid_time = ds[time_dim].size // 2
    level_means = ds['continuity_error'].isel({time_dim: mid_time}).mean(dim=['lat', 'lon'])
    axes[0,1].plot(level_means, ds['level'])
    axes[0,1].set_title(f'Vertical Profile of Continuity Error\nat time step {mid_time}')
    axes[0,1].set_xlabel('Mean Continuity Error')
    axes[0,1].set_ylabel('Pressure Level (hPa)')
    
    # Plot 3: Zonal mean analysis (fixed)
    zonal_mean = ds['continuity_error'].mean(dim=['lon', time_dim, 'level'])
    axes[1,0].plot(ds['lat'], zonal_mean)
    axes[1,0].set_title('Zonal Mean Continuity Error')
    axes[1,0].set_xlabel('Latitude')
    axes[1,0].set_ylabel('Mean Continuity Error')
    
    # Plot 4: Histogram of error magnitudes
    axes[1,1].hist(ds['continuity_error'].values.flatten(), bins=50)
    axes[1,1].set_title('Distribution of Continuity Error Values')
    axes[1,1].set_xlabel('Error Magnitude')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def analyze_mass_movement(ds, is_era=False):
    """Analyze how mass moves through the system over time."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate total mass in each column
    column_mass = ds['dry_air_mass'].sum(dim='level')
    
    # Calculate mass changes between timesteps
    mass_change = column_mass.diff(dim=time_dim)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mass Movement Analysis', fontsize=16)
    
    # Plot 1: Total mass over time
    total_mass = column_mass.sum(dim=['lat', 'lon'])
    axes[0,0].plot(total_mass)
    axes[0,0].set_title('Total System Mass Over Time')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Total Mass')
    
    # Plot 2: Zonal mean mass (fixed)
    zonal_mass = column_mass.mean(dim=['lon', time_dim])
    axes[0,1].plot(ds['lat'], zonal_mass)
    axes[0,1].set_title('Zonal Mean Column Mass')
    axes[0,1].set_xlabel('Latitude')
    axes[0,1].set_ylabel('Mean Mass')
    
    # Plot 3: Mass change distribution for middle timestep
    if mass_change.sizes[time_dim] > 0:  # Check if we have mass change data
        mid_time = mass_change.sizes[time_dim] // 2
        mass_change_snapshot = mass_change.isel({time_dim: mid_time})
        im = axes[1,0].imshow(mass_change_snapshot, 
                             cmap='RdBu_r',
                             aspect='auto',
                             origin='lower')
        axes[1,0].set_title(f'Mass Change at Time Step {mid_time}')
        plt.colorbar(im, ax=axes[1,0])
    else:
        axes[1,0].text(0.5, 0.5, 'Not enough timesteps for mass change calculation',
                      ha='center', va='center')
    
    # Plot 4: Histogram of mass changes
    if mass_change.sizes[time_dim] > 0:
        axes[1,1].hist(mass_change.values.flatten(), bins=50)
        axes[1,1].set_title('Distribution of Mass Changes')
        axes[1,1].set_xlabel('Mass Change')
        axes[1,1].set_ylabel('Frequency')
    else:
        axes[1,1].text(0.5, 0.5, 'Not enough timesteps for mass change calculation',
                      ha='center', va='center')
    
    plt.tight_layout()
    return fig

def calculate_mass_transport_metrics(ds, is_era=False):
    """Calculate key metrics about mass transport."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate column mass and its changes
    column_mass = ds['dry_air_mass'].sum(dim='level')
    
    metrics = {
        'total_mass_mean': float(column_mass.sum(dim=['lat', 'lon']).mean()),
        'total_mass_std': float(column_mass.sum(dim=['lat', 'lon']).std()),
    }
    
    # Only calculate change metrics if we have multiple timesteps
    if ds[time_dim].size > 1:
        mass_change = column_mass.diff(dim=time_dim)
        metrics.update({
            'max_local_mass_change': float(mass_change.max()),
            'min_local_mass_change': float(mass_change.min()),
            'mean_absolute_mass_change': float(np.abs(mass_change).mean()),
            'mass_change_std': float(mass_change.std()),
        })
    
    return metrics
# %%
# For GraphCast data
gc_continuity_fig = analyze_continuity_components(gc_batch_subset)
gc_mass_fig = analyze_mass_movement(gc_batch_subset)
gc_metrics = calculate_mass_transport_metrics(gc_batch_subset)

# For ERA5 data
era_continuity_fig = analyze_continuity_components(era_batch_subset, is_era=True)
era_mass_fig = analyze_mass_movement(era_batch_subset, is_era=True)
era_metrics = calculate_mass_transport_metrics(era_batch_subset, is_era=True)

# Print metrics for comparison
print("GraphCast metrics:")
for key, value in gc_metrics.items():
    print(f"{key}: {value}")

print("\nERA5 metrics:")
for key, value in era_metrics.items():
    print(f"{key}: {value}")
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import stats

def analyze_mass_centers(ds, is_era=False):
    """Track the 'center of mass' movement over time."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate column mass
    column_mass = ds['dry_air_mass'].sum(dim='level')
    
    # Create meshgrid of lat/lon coordinates
    lon_grid, lat_grid = np.meshgrid(ds['lon'], ds['lat'])
    
    # Calculate mass-weighted centers for each timestep
    times = []
    lat_centers = []
    lon_centers = []
    
    for t in range(ds[time_dim].size):
        mass_slice = column_mass.isel({time_dim: t})
        total_mass = float(mass_slice.sum())
        
        # Calculate weighted centers
        lat_center = float((mass_slice * lat_grid).sum() / total_mass)
        lon_center = float((mass_slice * lon_grid).sum() / total_mass)
        
        times.append(t)
        lat_centers.append(lat_center)
        lon_centers.append(lon_center)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectory
    ax1.plot(lon_centers, lat_centers, 'b.-')
    ax1.set_title('Mass Center Trajectory')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True)
    
    # Plot time series
    ax2.plot(times, lat_centers, label='Latitude')
    ax2.plot(times, lon_centers, label='Longitude')
    ax2.set_title('Mass Center Position Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def analyze_regional_mass_transport(ds, is_era=False):
    """Analyze mass transport patterns in different regions."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate column mass
    column_mass = ds['dry_air_mass'].sum(dim='level')
    
    # Define regions (latitude bands)
    regions = {
        'polar_north': slice(60, 90),
        'midlat_north': slice(30, 60),
        'tropical': slice(-30, 30),
        'midlat_south': slice(-60, -30),
        'polar_south': slice(-90, -60)
    }
    
    # Calculate mass in each region over time
    region_masses = {}
    for name, lat_slice in regions.items():
        region_masses[name] = column_mass.sel(lat=lat_slice).sum(dim=['lat', 'lon'])
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot absolute masses
    for name, mass in region_masses.items():
        axes[0].plot(mass, label=name)
    axes[0].set_title('Total Mass by Region')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Mass')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot mass changes
    for name, mass in region_masses.items():
        axes[1].plot(mass.diff(dim=time_dim), label=name)
    axes[1].set_title('Mass Change Rate by Region')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Mass Change')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def analyze_spectral_patterns(ds, is_era=False):
    """Analyze spectral patterns in mass movement."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate column mass
    column_mass = ds['dry_air_mass'].sum(dim='level')
    
    # Calculate global mean mass for each timestep
    global_mass = column_mass.mean(dim=['lat', 'lon'])
    
    # Convert to numpy array for FFT
    mass_values = global_mass.values
    
    # Perform FFT
    fft_values = fft.fft(mass_values)
    frequencies = fft.fftfreq(len(mass_values))
    
    # Calculate power spectrum
    power_spectrum = np.abs(fft_values)**2
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time series
    ax1.plot(global_mass)
    ax1.set_title('Global Mean Mass Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mass')
    ax1.grid(True)
    
    # Power spectrum (excluding zero frequency)
    ax2.plot(frequencies[1:len(frequencies)//2], 
             power_spectrum[1:len(frequencies)//2])
    ax2.set_title('Power Spectrum of Mass Variations')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig

def compare_models(gc_ds, era_ds):
    """Compare key statistics between GraphCast and ERA5."""
    # Calculate statistics for both models
    gc_mass = gc_ds['dry_air_mass'].sum(dim='level')
    era_mass = era_ds['dry_air_mass'].sum(dim='level')
    
    # Statistical comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution comparison
    axes[0,0].hist(gc_mass.values.flatten(), bins=50, alpha=0.5, label='GraphCast')
    axes[0,0].hist(era_mass.values.flatten(), bins=50, alpha=0.5, label='ERA5')
    axes[0,0].set_title('Distribution of Mass Values')
    axes[0,0].legend()
    
    # Q-Q plot
    stats.probplot(gc_mass.values.flatten(), dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (GraphCast)')
    
    # Spatial correlation
    gc_mean = gc_mass.mean(dim='prediction_timedelta')
    era_mean = era_mass.mean(dim='time')
    
    correlation = xr.corr(gc_mean, era_mean, dim=['lat', 'lon'])
    axes[1,0].text(0.5, 0.5, f'Spatial Correlation: {correlation.values:.3f}',
                   ha='center', va='center')
    axes[1,0].set_title('Model Correlation')
    
    # Error growth
    gc_std = gc_mass.std(dim=['lat', 'lon'])
    era_std = era_mass.std(dim=['lat', 'lon'])
    
    axes[1,1].plot(gc_std, label='GraphCast')
    axes[1,1].plot(era_std, label='ERA5')
    axes[1,1].set_title('Spatial Variability Over Time')
    axes[1,1].legend()
    
    plt.tight_layout()
    return fig
# %%
# Generate all analyses
mass_centers_gc = analyze_mass_centers(gc_batch_subset)
mass_centers_era = analyze_mass_centers(era_batch_subset, is_era=True)

regional_gc = analyze_regional_mass_transport(gc_batch_subset)
regional_era = analyze_regional_mass_transport(era_batch_subset, is_era=True)

spectral_gc = analyze_spectral_patterns(gc_batch_subset)
spectral_era = analyze_spectral_patterns(era_batch_subset, is_era=True)

model_comparison = compare_models(gc_batch_subset, era_batch_subset)
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_hydrostatic_balance(ds, is_era=False):
    """
    Analyze how well hydrostatic balance is maintained.
    dp/dz = -ρg
    """
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Convert level (hPa) to Pa for calculations
    pressure = ds['level'] * 100  # hPa to Pa
    density = ds['air_density']
    
    # Calculate vertical pressure gradient
    # Note: pressure decreases with height, so we negate the difference
    dp_dz = -(pressure.diff(dim='level'))
    
    # Calculate expected gradient from density (-ρg)
    g = 9.81  # m/s^2
    expected_gradient = -density * g
    expected_gradient = expected_gradient.isel(level=slice(1, None))  # Match dimensions with dp_dz
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hydrostatic Balance Analysis', fontsize=16)
    
    # Calculate mean profiles
    actual_profile = dp_dz.mean(dim=[time_dim, 'lat', 'lon'])
    expected_profile = expected_gradient.mean(dim=[time_dim, 'lat', 'lon'])
    
    # Plot vertical profiles
    axes[0,0].plot(actual_profile, ds['level'][1:], label='Actual')
    axes[0,0].plot(expected_profile, ds['level'][1:], label='Expected')
    axes[0,0].set_title('Mean Vertical Pressure Gradient Profile')
    axes[0,0].set_ylabel('Pressure Level (hPa)')
    axes[0,0].legend()
    axes[0,0].invert_yaxis()  # Higher pressure at bottom
    
    # Error distribution
    error = (dp_dz - expected_gradient)
    relative_error = error / expected_gradient
    axes[0,1].hist(relative_error.values.flatten(), bins=50)
    axes[0,1].set_title('Relative Hydrostatic Balance Error Distribution')
    axes[0,1].set_xlabel('Relative Error')
    
    # Time evolution of RMS error
    rms_error = np.sqrt((relative_error**2).mean(dim=['lat', 'lon', 'level']))
    axes[1,0].plot(rms_error)
    axes[1,0].set_title('RMS Relative Hydrostatic Balance Error Over Time')
    axes[1,0].set_xlabel('Time Step')
    
    # Spatial distribution of error
    mean_error = relative_error.mean(dim=[time_dim, 'level'])
    im = axes[1,1].imshow(mean_error, cmap='RdBu_r', origin='lower')
    axes[1,1].set_title('Mean Relative Hydrostatic Balance Error')
    plt.colorbar(im, ax=axes[1,1])
    
    plt.tight_layout()
    return fig

def analyze_thermal_wind(ds, is_era=False):
    """
    Analyze thermal wind relationship.
    ∂u/∂z ∝ ∂T/∂y
    
    Using pressure levels instead of height:
    ∂u/∂p ∝ ∂T/∂y
    """
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate wind shear and temperature gradient
    u_wind = ds['u_component_of_wind']
    temperature = ds['temperature']
    
    # Calculate wind shear with respect to pressure (note: flip sign since dp = -dz)
    du_dp = -u_wind.diff(dim='level') / (ds['level'].diff(dim='level') * 100)  # convert hPa to Pa
    
    # Calculate meridional temperature gradient
    # Convert lat to meters for proper scaling
    lat_meters = ds['lat'] * 111000  # rough conversion of degrees to meters
    dT_dy = temperature.diff(dim='lat') / lat_meters.diff(dim='lat')
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Thermal Wind Analysis', fontsize=16)
    
    # Calculate mean profiles
    mean_wind_shear = du_dp.mean(dim=[time_dim, 'lon', 'lat'])
    mean_temp_grad = dT_dy.mean(dim=[time_dim, 'lon', 'level'])
    
    # Plot vertical profiles
    axes[0,0].plot(mean_wind_shear, ds['level'][1:])
    axes[0,0].set_title('Mean Vertical Wind Shear Profile (du/dp)')
    axes[0,0].set_ylabel('Pressure Level (hPa)')
    axes[0,0].invert_yaxis()
    
    axes[0,1].plot(ds['lat'][1:], mean_temp_grad)
    axes[0,1].set_title('Mean Meridional Temperature Gradient (dT/dy)')
    axes[0,1].set_xlabel('Latitude')
    
    # Correlation analysis
    du_dp_mean = du_dp.mean(dim=['lon'])
    dT_dy_mean = dT_dy.mean(dim=['lon'])
    
    # Align the levels for correlation
    correlation = xr.corr(du_dp_mean, dT_dy_mean.isel(level=slice(1, None)), dim='lat')
    mean_corr = correlation.mean(dim=time_dim)
    
    axes[1,0].plot(ds['level'][1:], mean_corr)
    axes[1,0].set_title('Thermal Wind Correlation by Level')
    axes[1,0].set_xlabel('Pressure Level (hPa)')
    axes[1,0].set_ylabel('Correlation')
    axes[1,0].invert_yaxis()
    
    # Time evolution of correlation
    time_corr = correlation.mean(dim='level')
    axes[1,1].plot(time_corr)
    axes[1,1].set_title('Thermal Wind Correlation Over Time')
    axes[1,1].set_xlabel('Time Step')
    
    plt.tight_layout()
    return fig
def analyze_energy_conservation(ds, is_era=False):
    """
    Analyze conservation of energy in the system.
    """
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Calculate kinetic energy
    u = ds['u_component_of_wind']
    v = ds['v_component_of_wind']
    density = ds['air_density']
    ke = 0.5 * density * (u**2 + v**2)
    
    # Calculate potential energy (simplified)
    g = 9.81
    z = ds['geopotential'] / g  # height from geopotential
    pe = density * g * z
    
    # Calculate total energy
    total_energy = ke + pe
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Energy Conservation Analysis', fontsize=16)
    
    # Calculate mean energies
    mean_ke = ke.mean(dim=['lat', 'lon'])
    mean_pe = pe.mean(dim=['lat', 'lon'])
    mean_total = total_energy.mean(dim=['lat', 'lon'])
    
    # Time series of global mean energies
    axes[0,0].plot(mean_ke.mean(dim='level'), label='Kinetic')
    axes[0,0].plot(mean_pe.mean(dim='level'), label='Potential')
    axes[0,0].plot(mean_total.mean(dim='level'), label='Total')
    axes[0,0].set_title('Global Mean Energy Components')
    axes[0,0].legend()
    
    # Energy distribution by latitude
    lat_ke = ke.mean(dim=[time_dim, 'lon', 'level'])
    lat_pe = pe.mean(dim=[time_dim, 'lon', 'level'])
    
    axes[0,1].plot(ds['lat'], lat_ke, label='Kinetic')
    axes[0,1].plot(ds['lat'], lat_pe, label='Potential')
    axes[0,1].set_title('Mean Energy Distribution by Latitude')
    axes[0,1].legend()
    
    # Energy conservation metric
    energy_change = mean_total.diff(dim=time_dim)
    axes[1,0].hist(energy_change.values.flatten(), bins=50)
    axes[1,0].set_title('Distribution of Energy Changes')
    
    # Vertical profile of energy
    level_ke = mean_ke.mean(dim=time_dim)
    level_pe = mean_pe.mean(dim=time_dim)
    
    axes[1,1].plot(level_ke, ds['level'], label='Kinetic')
    axes[1,1].plot(level_pe, ds['level'], label='Potential')
    axes[1,1].set_title('Vertical Energy Profile')
    axes[1,1].set_ylabel('Pressure Level (hPa)')
    axes[1,1].legend()
    
    plt.tight_layout()
    return fig
def compare_physical_constraints(gc_ds, era_ds):
    """Compare how well each model satisfies physical constraints."""
    # Calculate key physical relationships
    results = {
        'gc': {},
        'era': {}
    }
    
    # Analyze hydrostatic balance errors
    for label, ds in [('gc', gc_ds), ('era', era_ds)]:
        is_era = (label == 'era')
        
        # Hydrostatic balance error
        pressure = ds['level']
        density = ds['air_density']
        dp_dz = pressure.diff(dim='level')
        expected_gradient = -density * 9.81
        hydrostatic_error = (dp_dz - expected_gradient.isel(level=slice(1, None)))
        results[label]['hydrostatic_rms'] = float(np.sqrt((hydrostatic_error**2).mean()))
        
        # Thermal wind relationship
        u_wind = ds['u']
        temperature = ds['temperature']
        du_dz = u_wind.diff(dim='level')
        dT_dy = temperature.diff(dim='lat')
        thermal_correlation = float(xr.corr(du_dz, dT_dy, dim='lat').mean())
        results[label]['thermal_wind_corr'] = thermal_correlation
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Physics Comparison', fontsize=16)
    
    # Bar plot of RMS hydrostatic errors
    axes[0,0].bar(['GraphCast', 'ERA5'], 
                 [results['gc']['hydrostatic_rms'], 
                  results['era']['hydrostatic_rms']])
    axes[0,0].set_title('RMS Hydrostatic Balance Error')
    
    # Bar plot of thermal wind correlations
    axes[0,1].bar(['GraphCast', 'ERA5'],
                 [results['gc']['thermal_wind_corr'],
                  results['era']['thermal_wind_corr']])
    axes[0,1].set_title('Thermal Wind Correlation')
    
    # Add additional comparisons as needed
    
    plt.tight_layout()
    return fig, results
# %%
# Analyze each model individually
# gc_hydrostatic = analyze_hydrostatic_balance(gc_batch_subset)
# era_hydrostatic = analyze_hydrostatic_balance(era_batch_subset, is_era=True)

gc_thermal = analyze_thermal_wind(gc_batch_subset)
era_thermal = analyze_thermal_wind(era_batch_subset, is_era=True)

gc_energy = analyze_energy_conservation(gc_batch_subset)
era_energy = analyze_energy_conservation(era_batch_subset, is_era=True)

# Direct comparison
comparison_fig, comparison_results = compare_physical_constraints(gc_batch_subset, era_batch_subset)
# %%
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_thermal_wind_by_region(ds, is_era=False, min_points=2):
    """Analyze thermal wind relationship in different latitude bands with robust error handling."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    # Define regions
    regions = {
        'tropics': slice(-30, 30),
        'northern_midlat': slice(30, 60),
        'southern_midlat': slice(-60, -30),
        'northern_polar': slice(60, 90),
        'southern_polar': slice(-90, -60)
    }
    
    # Calculate wind shear and temperature gradient
    u_wind = ds['u_component_of_wind']
    temperature = ds['temperature']
    
    # Calculate wind shear with respect to pressure
    du_dz = u_wind.diff(dim='level')
    dp = ds['level'].diff('level') * 100  # convert hPa to Pa
    du_dp = -du_dz / dp
    
    # Calculate meridional temperature gradient
    dy = (ds['lat'].diff('lat') * 111000)  # rough conversion of degrees to meters
    dT_dy = temperature.diff(dim='lat') / dy
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Regional Thermal Wind Analysis', fontsize=16)
    
    # Analyze by region
    correlations = {}
    rms_errors = {}
    
    for region_name, lat_slice in regions.items():
        try:
            # Select region
            region_du_dp = du_dp.sel(lat=lat_slice)
            region_dT_dy = dT_dy.sel(lat=lat_slice)
            
            # Ensure we have enough data points
            if len(region_du_dp.lat) < min_points:
                print(f"Warning: Insufficient data points in {region_name}")
                continue
                
            # Calculate mean profiles
            du_dp_mean = region_du_dp.mean(dim=['lon'])
            dT_dy_mean = region_dT_dy.mean(dim=['lon'])
            
            # Calculate correlation only where both fields have valid data
            valid_mask = ~(np.isnan(du_dp_mean) | np.isnan(dT_dy_mean))
            if valid_mask.sum() > min_points:
                correlation = xr.corr(
                    du_dp_mean.where(valid_mask),
                    dT_dy_mean.isel(level=slice(1, None)).where(valid_mask),
                    dim='lat'
                )
                correlations[region_name] = correlation.mean(dim=time_dim)
            
            # Calculate RMS errors
            error = (region_du_dp - region_dT_dy.isel(level=slice(1, None))).mean(dim=['lon', 'lat'])
            rms = np.sqrt((error**2).mean().where(~np.isnan(error), 0))
            rms_errors[region_name] = float(rms)
            
        except Exception as e:
            print(f"Warning: Error processing {region_name}: {str(e)}")
            continue
    
    # Plotting
    try:
        # Plot regional correlations by level
        for name, corr in correlations.items():
            if not np.all(np.isnan(corr)):
                axes[0,0].plot(corr, ds['level'][1:], label=name)
        axes[0,0].set_title('Thermal Wind Correlation by Region')
        axes[0,0].set_ylabel('Pressure Level (hPa)')
        axes[0,0].set_xlabel('Correlation')
        axes[0,0].legend()
        axes[0,0].invert_yaxis()
        
        # Plot mean vertical profiles by region
        for name in regions.keys():
            if name in correlations:
                mean_du_dp = du_dp.sel(lat=regions[name]).mean(dim=['lat', 'lon', time_dim])
                if not np.all(np.isnan(mean_du_dp)):
                    axes[0,1].plot(mean_du_dp, ds['level'][1:], label=name)
        axes[0,1].set_title('Mean Wind Shear Profiles by Region')
        axes[0,1].set_xlabel('du/dp')
        axes[0,1].set_ylabel('Pressure Level (hPa)')
        axes[0,1].legend()
        axes[0,1].invert_yaxis()
        
        # Plot valid RMS errors
        valid_regions = [k for k, v in rms_errors.items() if not np.isnan(v)]
        valid_values = [rms_errors[k] for k in valid_regions]
        if valid_regions:
            axes[1,0].bar(valid_regions, valid_values)
            axes[1,0].set_title('RMS Errors by Region')
            plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot zonal mean wind shear
        mean_du_dp = du_dp.mean(dim=['lon', time_dim])
        for level in [1000, 850, 500, 200]:  # Example pressure levels
            try:
                level_data = mean_du_dp.sel(level=level, method='nearest')
                if not np.all(np.isnan(level_data)):
                    axes[1,1].plot(ds['lat'], level_data, label=f'{level} hPa')
            except Exception as e:
                print(f"Warning: Could not plot level {level}: {str(e)}")
        axes[1,1].set_title('Zonal Mean Wind Shear at Different Levels')
        axes[1,1].set_xlabel('Latitude')
        axes[1,1].set_ylabel('du/dp')
        axes[1,1].legend()
        
    except Exception as e:
        print(f"Warning: Error in plotting: {str(e)}")
    
    plt.tight_layout()
    return fig, correlations, rms_errors

def analyze_energy_conservation_detail(ds, is_era=False):
    """Detailed analysis of energy conservation with robust error handling."""
    time_dim = 'time' if is_era else 'prediction_timedelta'
    
    try:
        # Calculate energies
        ke = 0.5 * ds['air_density'] * (ds['u_component_of_wind']**2 + ds['v_component_of_wind']**2)
        g = 9.81
        pe = ds['air_density'] * g * (ds['geopotential'] / g)
        total_energy = ke + pe
        
        # Calculate energy changes safely
        energy_change = total_energy.diff(dim=time_dim)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Energy Conservation Analysis', fontsize=16)
        
        # Plot energy change distribution by pressure level
        level_changes = energy_change.std(dim=[time_dim, 'lat', 'lon'])
        axes[0,0].plot(level_changes, ds['level'])
        axes[0,0].set_title('Energy Variability by Level')
        axes[0,0].set_xlabel('Standard Deviation of Energy Changes')
        axes[0,0].set_ylabel('Pressure Level (hPa)')
        axes[0,0].invert_yaxis()
        
        # Plot zonal mean energy
        zonal_energy = total_energy.mean(dim=['lon', time_dim])
        for level in [1000, 850, 500, 200]:
            try:
                level_data = zonal_energy.sel(level=level, method='nearest')
                if not np.all(np.isnan(level_data)):
                    axes[0,1].plot(ds['lat'], level_data, label=f'{level} hPa')
            except Exception as e:
                print(f"Warning: Could not plot level {level}: {str(e)}")
        axes[0,1].set_title('Zonal Mean Energy at Different Levels')
        axes[0,1].set_xlabel('Latitude')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].legend()
        
        # Calculate and plot conservation metric over time
        conservation_metric = np.abs(energy_change).mean(dim=['lat', 'lon', 'level'])
        axes[1,0].plot(conservation_metric)
        axes[1,0].set_title('Energy Conservation Metric Over Time')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Mean Absolute Energy Change')
        
        # Plot relative energy changes with careful handling of zeros
        valid_energy = total_energy.where(total_energy != 0)
        relative_change = (energy_change / valid_energy).where(valid_energy != 0)
        finite_changes = relative_change.values[np.isfinite(relative_change.values)]
        if len(finite_changes) > 0:
            axes[1,1].hist(finite_changes, bins=50, range=np.percentile(finite_changes, [1, 99]))
            axes[1,1].set_title('Distribution of Relative Energy Changes')
            axes[1,1].set_xlabel('Relative Change')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Calculate summary statistics carefully
        stats = {
            'mean_absolute_change': float(np.nanmean(np.abs(energy_change))),
            'std_change': float(np.nanstd(energy_change)),
            'max_abs_change': float(np.nanmax(np.abs(energy_change))),
            'relative_change_std': float(np.nanstd(relative_change))
        }
        
        return fig, stats
        
    except Exception as e:
        print(f"Error in energy conservation analysis: {str(e)}")
        return None, None
# %%
# Regional thermal wind analysis
gc_thermal_regional, gc_correlations, gc_rms = analyze_thermal_wind_by_region(gc_batch_subset)
era_thermal_regional, era_correlations, era_rms = analyze_thermal_wind_by_region(era_batch_subset, is_era=True)

# Detailed energy conservation analysis
gc_energy_detail, gc_energy_stats = analyze_energy_conservation_detail(gc_batch_subset)
era_energy_detail, era_energy_stats = analyze_energy_conservation_detail(era_batch_subset, is_era=True)

# Print comparison statistics
print("\nThermal Wind RMS Errors:")
for region in gc_rms.keys():
    print(f"{region}:")
    print(f"  GraphCast: {gc_rms[region]:.6f}")
    print(f"  ERA5: {era_rms[region]:.6f}")
    print(f"  Difference: {((gc_rms[region] - era_rms[region])/era_rms[region]*100):.2f}%")

print("\nEnergy Conservation Metrics:")
for metric in gc_energy_stats.keys():
    print(f"{metric}:")
    print(f"  GraphCast: {gc_energy_stats[metric]:.6f}")
    print(f"  ERA5: {era_energy_stats[metric]:.6f}")
    print(f"  Difference: {((gc_energy_stats[metric] - era_energy_stats[metric])/era_energy_stats[metric]*100):.2f}%")
# %%
