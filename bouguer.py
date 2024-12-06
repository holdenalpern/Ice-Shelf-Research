import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import harmonica as hm
from sklearn.preprocessing import QuantileTransformer
import gstatsim
import skgstat as skg
import xarray as xr
import xrft
import verde as vd

import warnings
warnings.filterwarnings("ignore")

from prisms import PrismGen
from utilities import xy_into_grid, lowpass_filter_domain

def nte_correction_sgs(ds, grav, density):
    density_dict = {
        'ice': 917,
        'water': 1027,
        'rock': density
    }
    pgen = PrismGen(density_dict)
    
    # Crop the dataset to the area of interest
    x_min, x_max = grav['x'].min(), grav['x'].max()
    y_min, y_max = grav['y'].min(), grav['y'].max()
    buffer = 50_000  # Adjust buffer size as needed

    ds_cropped = ds.sel(x=slice(x_min - buffer, x_max + buffer), y=slice(y_min - buffer, y_max + buffer))
    grav_cropped = grav[(grav['x'] >= x_min - buffer) & (grav['x'] <= x_max + buffer) &
                        (grav['y'] >= y_min - buffer) & (grav['y'] <= y_max + buffer)]

    # Ensure 'inv_msk' is defined in 'ds_cropped'
    if 'inv_msk' not in ds_cropped:
        ds_cropped['inv_msk'] = ds_cropped['mask'] == 3  # Adjust based on your data

    # Generate prisms only where needed
    prisms, densities = pgen.make_prisms(ds_cropped, ds_cropped.bed.values, msk='inv')

    pred_coords = (grav_cropped.x.values, grav_cropped.y.values, grav_cropped.height.values)

    # Compute gravity in chunks to reduce memory usage
    def compute_gravity_in_chunks(pred_coords, prisms, densities, chunk_size=500):
        g_z_total = np.zeros(len(pred_coords[0]))
        num_chunks = len(prisms) // chunk_size + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(prisms))
            g_z_chunk = hm.prism_gravity(
                coordinates=pred_coords,
                prisms=prisms[start:end],
                density=densities[start:end],
                field='g_z',
                parallel=True
            )
            g_z_total += g_z_chunk
        return g_z_total

    g_z = compute_gravity_in_chunks(pred_coords, prisms, densities, chunk_size=500)

    residual = grav_cropped.faa.values - g_z

    # Use only the points outside the inversion mask
    mask_outside_inv = grav_cropped['inv_msk'] == False
    X = np.stack([grav_cropped.x.values[mask_outside_inv], grav_cropped.y.values[mask_outside_inv]]).T
    y = residual[mask_outside_inv]
    XX = np.stack([grav_cropped.x.values, grav_cropped.y.values]).T

    # Normal score transformation
    data = y.reshape(-1, 1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    NormZ = nst_trans.transform(data).squeeze()

    # Compute experimental (isotropic) variogram
    coords = X
    values = NormZ

    maxlag = 100_000  # Adjust as needed
    n_lags = 40       # Reduced from 70

    V3 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags,
                       maxlag=maxlag, normalize=False)
    V3.model = 'spherical'

    # Set variogram parameters
    azimuth = 0
    nugget = V3.parameters[2]
    major_range = V3.parameters[0]
    minor_range = V3.parameters[0]
    sill = V3.parameters[1]
    vtype = 'spherical'

    # Save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 16         # Reduced from 64
    rad = 50_000   # Reduced from 100,000 meters

    # Prepare DataFrame for SGS interpolation
    df_grid = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'NormZ': NormZ})

    sim = gstatsim.Interpolation.okrige_sgs(XX, df_grid, 'X', 'Y', 'NormZ', k, vario, rad, quiet=True)
    sim_trans = nst_trans.inverse_transform(sim.reshape(-1, 1))

    # Map the results back to the original gravity dataset
    grav_full = grav.copy()
    grav_full['sim'] = np.nan
    grav_full.loc[grav_cropped.index, 'sim'] = sim_trans.squeeze()

    return grav_full.faa.values - grav_full['sim'].values




from scipy.interpolate import RegularGridInterpolator

def filter_boug(ds, grav, target, cutoff, pad):
    """
    Implement your actual Bouguer filtering logic here.
    This is a placeholder function and should be replaced with your real implementation.
    """
    # Placeholder: Return the target unfiltered
    # Replace this with your actual filtering logic
    return target

def sgs_filt(ds, grav, density, cutoff, pad, sim1):
    """
    Filters the Bouguer gravity data based on SGS simulation.

    Parameters:
    - ds: xarray Dataset containing grid information.
    - grav: pandas DataFrame containing gravity observations.
    - density: Density value used in calculations.
    - cutoff: Cutoff frequency for filtering.
    - pad: Padding parameter.
    - sim1: SGS simulation data as a 2D NumPy array.

    Returns:
    - new_target: The result of subtracting the interpolated Bouguer filter from the observed gravity data.
                  Shape: (n_observations,)
    """
    # Apply the filter_boug function to the simulation grid
    boug_filt = filter_boug(ds, grav, sim1, cutoff, pad)
    print("boug_filt shape after filter_boug:", boug_filt.shape)
    
    # If boug_filt is 1D, reshape to (ny, nx)
    if boug_filt.ndim == 1:
        ny = len(ds['y'])
        nx = len(ds['x'])
        if boug_filt.size != ny * nx:
            raise ValueError(f"Expected boug_filt size {ny * nx}, but got {boug_filt.size}")
        boug_filt = boug_filt.reshape((ny, nx))
        print("Reshaped boug_filt to:", boug_filt.shape)
    
    # Prepare grid coordinates
    x_grid = ds['x'].values  # Shape: (nx,)
    y_grid = ds['y'].values  # Shape: (ny,)
    
    # Ensure grid coordinates are sorted in ascending order
    x_sorted_indices = np.argsort(x_grid)
    y_sorted_indices = np.argsort(y_grid)
    x_grid_sorted = x_grid[x_sorted_indices]
    y_grid_sorted = y_grid[y_sorted_indices]
    
    # Sort boug_filt accordingly
    boug_filt_sorted = boug_filt[y_sorted_indices, :][:, x_sorted_indices]
    print("boug_filt_sorted shape:", boug_filt_sorted.shape)
    
    # Create interpolator for boug_filt
    interpolator_boug_filt = RegularGridInterpolator(
        (y_grid_sorted, x_grid_sorted),
        boug_filt_sorted,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Prepare observation points
    x_obs = grav['x'].values  # Shape: (n_observations,)
    y_obs = grav['y'].values  # Shape: (n_observations,)
    points = np.column_stack((y_obs, x_obs))  # Shape: (n_observations, 2)
    
    print("Interpolate boug_filt to observation points...")
    
    # Interpolate boug_filt to observation points
    boug_filt_at_obs = interpolator_boug_filt(points)  # Shape: (n_observations,)
    print("boug_filt_at_obs shape:", boug_filt_at_obs.shape)
    
    # Check if boug_filt_at_obs has the same shape as grav.faa.values
    if boug_filt_at_obs.shape != grav.faa.values.shape:
        raise ValueError(f"After interpolation, boug_filt_at_obs shape {boug_filt_at_obs.shape} does not match grav.faa.values shape {grav.faa.values.shape}")
    
    # Compute new_target
    new_target = grav.faa.values - boug_filt_at_obs  # Shape: (n_observations,)
    print("new_target shape:", new_target.shape)
    
    return new_target
