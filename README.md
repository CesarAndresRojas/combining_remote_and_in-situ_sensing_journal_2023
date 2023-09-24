# combining_remote_and_in-situ_sensing_journal_2023

This repository contains Python scripts that combines remote sensing data from Sentinel-2 (S2) and Landsat 8-9 (L8-9) with in-situ data collected in Biscayne Bay to create a 3-D Satellite Derived Underwater Environment (SDUE). The goal is to incorporate the SDUE as an extra sensor in a Kalman filter for underwater robot localization.

This code is part of the accompanying source code for the publication titled "Combining Multi-Satellite Remote and In-situ Sensing for
Autonomous Underwater Vehicle Localization and Navigation" published to the Elsevier Ocean Engineering.

## Prerequisites

- Python 3.x installed on your system
- Required Python packages (Install with `pip install -r requirements.txt`): datetime, json, math, netCDF4, shapely, scikit-learn, rasterio, rioxarray, xarray, geopandas, matplotlib, numpy, pandas,
  
## Note for Windows users:
Installing GDAL packages on Windows is complicated, for higher chances of success please do the following:

1. Download and install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
2. Download precompiled GDAL Wheel:
   - https://github.com/cgohlke/geospatial-wheels/
   - https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal (No Longer Available)
   
3. Install GDAL wheel with PIP: `pip install package-name.whl`


## Getting Started

1. Clone this repository to your local machine.

2. Install the required Python packages by running `pip install -r requirements.txt`.

3. Run '20221007_estimation_<PARAMTER_NAME>.py' to create surface water quality estimators
   
5. Create surface water quality estimators for bathymetry, chlorophyll, dissolved oxygen, and tubidity.

6. Run '20221007_SDUE_knn.py' to create the Satellite Derived Underwater Enviornment (SDUE) using KNN regressors that extend the surface water quality estimates to lower levels.

7. Run '20221007_SDUE_plot.py' to draw a 3-D contour plot depicting the water quality estimates for each parameter.

8. Run '20221007_SDUE_plot2.py' to draw a 3-D surface plot depicting the water quality estimates for each parameter.

9. The jupyter notebook 'Example_execution.ipynb' shows how to use the SDUE in a kalman filter.

## Usage

All scripts do the following:

- Reads datasets in the 'data' directory.

- Datasets are organized by 'level' according to the amount of processing.

- Outputs Charts to the 'data' directory, typically in the level 4 subdirectory.
  
## License

This project is licensed under the [MIT License](LICENSE).
