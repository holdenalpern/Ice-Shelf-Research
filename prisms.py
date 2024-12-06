import numpy as np
import xarray as xr
import boule as bl
    
class PrismGen:
    def __init__(self, density_dict):
        self.ice_dens = density_dict['ice']
        self.water_dens = density_dict['water']
        self.rock_dens = density_dict['rock']

    # prisms.py

import numpy as np

class PrismGen:
    def make_prisms(self, ds, bed, msk, ice):
        """
        Create prisms based on mask and bed data.

        Parameters:
        - ds: xarray Dataset containing grid information.
        - bed: 2D NumPy array representing bed elevation.
        - msk: Mask type, e.g., 'inv'.
        - ice: Boolean indicating if ice prisms should be created.

        Returns:
        - prisms: List of prism definitions.
        - densities: List of densities corresponding to each prism.
        """
        # Use 'inv_mask' instead of 'dist_msk'
        inv_mask = ds.inv_mask.data  # Shape: (ny, nx)

        # Water prisms mask
        water_msk = np.where(((ds.mask == 0) ^ (ds.mask == 3)) & (inv_mask == True), True, False)

        # Initialize lists to store prisms and densities
        prisms = []
        densities = []

        # Define densities
        water_density = 1027  # kg/m^3
        ice_density = 917      # kg/m^3
        rock_density = 2670    # kg/m^3

        # Process water prisms
        if np.any(water_msk):
            water_bed = bed[water_msk]
            prisms.extend(water_bed.flatten())
            densities.extend([water_density] * water_bed.size)

        # Ice prisms mask (if applicable)
        if ice:
            ice_msk = np.where(((ds.mask == 3) ^ (ds.mask == 2)) & (inv_mask == True), True, False)
            if np.any(ice_msk):
                ice_bed = bed[ice_msk]
                prisms.extend(ice_bed.flatten())
                densities.extend([ice_density] * ice_bed.size)
        
        # Rock prisms mask (assuming mask == 2 indicates rock)
        rock_msk = np.where(ds.mask == 2, True, False)
        if np.any(rock_msk):
            rock_bed = bed[rock_msk]
            prisms.extend(rock_bed.flatten())
            densities.extend([rock_density] * rock_bed.size)

        return prisms, densities


    def split_prisms(self, prisms):
        '''
        Function to split prisms above and below the ellipsoid.
        Rerturns combined prisms and an index of which ones are above the ellipsoid.
        '''
        prisms_pos = prisms[prisms[:,5] >= 0, :]
        prisms_neg = prisms[prisms[:,4] < 0, :]
        prisms_pos[prisms_pos[:,4] < 0, 4] = 0.0
        prisms_neg[prisms_neg[:,5] > 0, 5] = 0.0
        prisms = np.vstack([prisms_pos, prisms_neg])
        idx_pos = np.full(prisms.shape[0], False)
        idx_pos[:prisms_pos.shape[0]] = True
        return prisms, idx_pos