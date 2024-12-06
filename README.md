# Ice-Shelf-Research
This repository contains the code from a research project which uses SGS and MCMC to model bathymetry under the Getz Ice Shelf.

The goal is to enhance our understanding of sub-ice geological structures, which are crucial for assessing ice dynamics and stability in polar regions, to understand patterns like grounding line retreat, which affect sea level rise, with acccurate uncertainty.

The workflow begins with **data ingestion**, where gravity measurements and BedMachine topography are imported and preprocessed. **Gravity corrections** are then applied to account for various geophysical effects, ensuring the data accurately reflects the underlying bedrock features. To improve signal quality and reduce noise, **upward continuation** techniques are employed on the gravity data.

The core of the project involves **geostatistical inversion** methods, utilizing Markov Chain Monte Carlo (MCMC) techniques to estimate the bedrock topography from the corrected gravity data. This process is guided by a sequence of inversion parameters that control the resolution, range, and amplitude of model updates, as well as conditioning weights that ensure smooth transitions and prevent artifacts near the boundaries of the inversion domain.

The repository includes several Jupyter notebooks and Python scripts that cover each stage of the process:

- **Jupyter Notebooks:**
  - `01_data_ingest.ipynb`: Imports and preprocesses raw datasets.
  - `02_gravity_intro.ipynb`: Introduces the significance of gravity data in bedrock modeling.
  - `03_gravity_corrections.ipynb`: Applies necessary corrections to the gravity measurements.
  - `04_geostatistics.ipynb`: Implements geostatistical methods for inversion.
  - `MCMC_linear_regression.ipynb`: Explores MCMC techniques for linear regression on gravity data.
  - `single_inv.ipynb`: Conducts a single-stage inversion for bedrock estimation.
  - `upward_continuation.ipynb`: Performs upward continuation to enhance gravity data quality.

- **Python Scripts:**
  - `block_update.py`: Manages block-based updates during the inversion process.
  - `bouguer.py`: Contains functions for Bouguer anomaly calculations.
  - `diagnostics.py`: Provides tools for evaluating inversion results.
  - `prisms.py`: Generates prisms used in gravity modeling.
  - `rfgen.py`: Handles random field generation for geostatistical modeling.
  - `sgs_inversions.py`: Implements stochastic geostatistical inversion techniques.
  - `rbf_mcmc.py`: Integrates Radial Basis Functions with MCMC for inversion.
  - `utilities.py`: Includes utility functions used across various scripts.

Additional data files such as `final_gravity_dataset_with_modeled_gravity.csv` and `final_bedmachine_dataset_polar_stereo.nc` are provided for use in the inversion process.

The project aims to produce accurate and plausible models of ice shelf bedrock, with adequate levels of uncertainty, via SGS and MCMC, contributing valuable insights to glaciological research and beyond.
