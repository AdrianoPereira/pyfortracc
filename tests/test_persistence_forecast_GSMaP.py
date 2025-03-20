"""
PyForTraCC Persistence Forecast Test Script

This script tests the persistence forecast functionality by generating forecasts
for a set of radar data and visualizing the results.

Author: Adriano P. Almeida (Forecast module)
Date: March 14, 2025
"""
# Add package to path - consider using proper package installation instead
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import gzip
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pyfortracc
from pyfortracc.forecast import persistence_forecast
from dotenv import load_dotenv
import matplotlib
matplotlib.use('TkAgg')

# Load environment variables
load_dotenv()

import gzip
def read_function(path):
    ams = gzip.open(path,mode='rb')
    data = np.frombuffer(ams.read(), dtype=np.float32).reshape(1200, 3600)
    data = np.roll(data, shift=1800, axis=1)[::-1]
    return data


def read_gsmap_data(path):
        """Read radar data from NetCDF file.
        
        Parameters:
        -----------
        path : str
            Path to the NetCDF file
            
        Returns:
        --------
        numpy.ndarray
            2D array of radar data
        """
        variable = "DBZc"
        z_level = 6  # Elevation level 2.5 km
        
        try:
            with gzip.open(path) as gz:
                with netCDF4.Dataset("dummy", mode="r", memory=gz.read()) as nc:
                    data = nc.variables[variable][:].data[0, z_level, :, :][::-1, :]
                    data[data == -9999] = np.nan
            return data
        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            raise


if __name__ == '__main__':
    """Run a demonstration of the persistence forecast."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('persistence_forecast_demo')
    
    # Path configuration - adjust to your environment
    input_path = os.getenv('INPUT_PATH')
    output_path = os.getenv('OUTPUT_PATH')
    
    # Configure forecast parameters
    name_list = {
        # Input/output paths
        'input_path': input_path,
        'output_path': output_path,
        
        # Segmentation parameters
        'thresholds': [0.1],
        'min_cluster_size': [6],
        'operator': '>=',
        'timestamp_pattern': 'gsmap_mvk.%Y%m%d.%H%M.v7.0001.0.dat.gz', 
        'delta_time': 60,
        
        # Geographic boundaries
        'lon_min': -179.95,
        'lon_max': 179.95,
        'lat_min': -59.95,
        'lat_max': 59.95,
        
        # Forecast parameters
        'last_timestamp': '2014-10-06 15:00',
        'horizon_forecast': 10,
        'track_forecast': False,
        'wrap_grid': True,
        'edges': True,
        'variable_unit': 'dBZ',
        'save_forecast': True
    }
    
    logger.info("Starting persistence forecast demo")
    
    try:
        # Generate forecasts
        # pyfortracc.track(name_list, read_function)
        forecast_frames = persistence_forecast(name_list, read_function)
        logger.info(f"Generated {len(forecast_frames)} forecast frames")
        
        # Visualize results
        logger.info("Creating visualization")
        
        # Define color maps for different forecast horizons
        cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys', 
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot each forecast horizon with a different color
        for i, frame in enumerate(forecast_frames):
            # Create a binary mask for each forecast
            mask = np.zeros_like(frame)
            mask[frame > 0] = 1
            
            # Use contour to show the outlines
            contour = ax.contour(
                mask, 
                levels=[0.5], 
                colors=[plt.cm.get_cmap(cmaps[i % len(cmaps)])(0.7)],
                linewidths=2
            )
        
        # Add geographic grid if coordinates are available
        if all(k in name_list for k in ['lon_min', 'lon_max', 'lat_min', 'lat_max']):
            # Create longitude and latitude arrays
            lons = np.linspace(name_list['lon_min'], name_list['lon_max'], frame.shape[1])
            lats = np.linspace(name_list['lat_min'], name_list['lat_max'], frame.shape[0])
            
            # Set axis labels
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            # Set axis ticks
            lon_ticks = np.linspace(name_list['lon_min'], name_list['lon_max'], 5)
            lat_ticks = np.linspace(name_list['lat_min'], name_list['lat_max'], 5)
            
            ax.set_xticks(np.linspace(0, frame.shape[1]-1, 5))
            ax.set_yticks(np.linspace(0, frame.shape[0]-1, 5))
            ax.set_xticklabels([f"{lon:.2f}°" for lon in lon_ticks])
            ax.set_yticklabels([f"{lat:.2f}°" for lat in lat_ticks])
        
        # Add title and legend
        plt.title(f"Persistence Forecast from {name_list['last_timestamp']}")
        plt.legend(loc='upper right')
        
        # Add grid
        plt.grid(alpha=0.3)
        
        # Save figure
        output_file = os.path.join(output_path, 'persistence_forecast_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
        
        # Show figure
        plt.show()
        
        logger.info("Demo completed successfully")
    
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
