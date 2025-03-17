import numpy as np
import duckdb
import datetime
from tqdm import tqdm
import xarray as xr
import os
import logging
from typing import List, Dict, Any


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with specified name, log file, and level.
    
    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : str, optional
        Path to log file, if None logs will only show in console
    level : int, optional
        Logging level, default is INFO
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def persistence_forecast(name_list: Dict[str, Any], read_function) -> np.ndarray:
    """
    Generate forecast frames by extrapolating the movement of clusters from the last tracked frame.
    
    Parameters:
    -----------
    name_list : dict
        Dictionary containing configuration forecast parameters:
        - 'horizon_forecast': Number of time steps to forecast ahead
        - 'last_timestamp': Timestamp of the last tracked frame (format: 'YYYY-MM-DD HH:MM')
        - 'output_path': Path to the tracking data
        - 'wrap_grid': Boolean indicating whether to wrap coordinates around grid boundaries
        - 'save_forecast': Boolean indicating whether to save forecast frames
    
    read_function : function
        Function to read a frame from a file path
    
    Returns:
    --------
    np.ndarray
        Array of forecast frames with shape (horizon_forecast, height, width)
    """
    # Set up logger
    log_dir = os.path.join(name_list['output_path'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"persistence_forecast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger('persistence_forecast', log_file)
    
    logger.info("Starting persistence forecast generation")
    
    # Extract forecast parameters from configuration
    horizon_forecast = name_list['horizon_forecast']
    logger.info(f"Forecast horizon set to {horizon_forecast} steps")
    
    last_timestamp = datetime.datetime.strptime(name_list['last_timestamp'], '%Y-%m-%d %H:%M')
    logger.info(f"Last timestamp: {last_timestamp}")
    
    save_forecast = name_list.get('save_forecast', True)
    logger.info(f"Save forecast option: {save_forecast}")
    
    delta_time = name_list['delta_time']
    logger.info(f"Delta time between steps: {delta_time} minutes")
    
    variable_unit = name_list.get('variable_unit', None)
    logger.info(f"Variable unit: {variable_unit}")
    
    # Log coordinate system information
    lon_min = name_list.get('lon_min', None)
    lon_max = name_list.get('lon_max', None)
    lat_min = name_list.get('lat_min', None)
    lat_max = name_list.get('lat_max', None)
    
    if all(x is not None for x in [lon_min, lon_max, lat_min, lat_max]):
        logger.info(f"Domain boundaries: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")
    else:
        logger.info("Using array indices as coordinates (geographic coordinates not provided)")
    
    # Build path to the last tracking table
    last_tracking_table_filename = (f"{name_list['output_path']}track/trackingtable/"
                                   f"{last_timestamp.strftime('%Y%m%d_%H%M')}.parquet")
    logger.info(f"Reading tracking table from: {last_tracking_table_filename}")
    
    try:
        # Read tracking table using DuckDB
        query = f"SELECT * FROM read_parquet('{last_tracking_table_filename}')"
        last_tracking_table = duckdb.query(query).df()
        logger.info(f"Successfully read tracking table with {len(last_tracking_table)} clusters")
        
        # Read the last frame using the provided read function
        frame_file = last_tracking_table['file'].iloc[0]
        logger.info(f"Reading last frame from: {frame_file}")
        last_frame = read_function(frame_file)
        logger.info(f"Last frame shape: {last_frame.shape}")
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        raise
    
    # Create coordinate grids
    if all(x is not None for x in [lon_min, lon_max, lat_min, lat_max]):
        lons = np.linspace(lon_min, lon_max, last_frame.shape[1])
        lats = np.linspace(lat_min, lat_max, last_frame.shape[0])
        logger.info("Created geographic coordinate grids")
    else:
        lons = np.arange(last_frame.shape[1])
        lats = np.arange(last_frame.shape[0])
        logger.info("Created index-based coordinate grids")

    pixel_width = lons[1] - lons[0]
    pixel_height = lats[1] - lats[0]
    logger.info(f"Pixel dimensions: width={pixel_width:.4f}, height={pixel_height:.4f}")
    
    # Initialize list to store forecast frames
    forecast_frames = []
    logger.info("Initialized forecast frames list")
    
    # Create forecast directory if saving is enabled
    if save_forecast:
        forecast_dir = f"{name_list['output_path']}forecast/persistence/"   
        os.makedirs(forecast_dir, exist_ok=True)
        logger.info(f"Created forecast directory: {forecast_dir}")
    
    # Count active clusters (non-NEW status)
    active_clusters = sum(1 for cluster in last_tracking_table.to_dict('records') if 'NEW' not in cluster['status'])
    logger.info(f"Found {active_clusters} active clusters to be extrapolated")
    
    # Generate forecasts for each horizon step with progress bar
    for horizon in tqdm(range(1, horizon_forecast + 1), desc="Generating forecasts", unit="horizon"):
        logger.info(f"Processing forecast horizon {horizon}/{horizon_forecast}")
        forecast_frame = np.zeros_like(last_frame)
        
        # Get tracking data as a list of dictionaries for iteration
        clusters = duckdb.query(query).to_df().to_dict('records')
        processed_clusters = 0
        
        # Process each cluster in the tracking table
        for cluster in clusters:
            # Skip newly detected clusters that don't have trajectory information
            if 'NEW' in cluster['status']:
                continue
            
            processed_clusters += 1
            
            # Extract cluster data
            uid = cluster['uid']  # Unique identifier for the cluster
            array_x = np.array(cluster['array_x'])  # X coordinates of cluster points
            array_y = np.array(cluster['array_y'])  # Y coordinates of cluster points
            u = cluster['u_']  # X velocity component
            v = cluster['v_']  # Y velocity component
            
            # Log cluster details at DEBUG level
            logger.debug(f"Extrapolating cluster {uid} with velocity u={u:.2f}, v={v:.2f}")
            
            # Calculate new positions based on velocity vectors
            if name_list['edges']:
                # Wrap coordinates around grid boundaries if specified
                new_array_y = np.ceil(array_y + ((v/pixel_height) * horizon)).astype(int) % last_frame.shape[0]
                new_array_x = np.ceil(array_x + ((u/pixel_width) * horizon)).astype(int) % last_frame.shape[1]
                logger.debug(f"Wrapped coordinates used for cluster {uid}")
            else:
                # Otherwise, calculate new positions and clip to frame boundaries
                new_array_y = np.ceil(array_y + ((v/pixel_height) * horizon)).astype(int)
                new_array_x = np.ceil(array_x + ((u/pixel_width) * horizon)).astype(int)
                
                # Ensure new positions stay within frame boundaries
                new_array_y = np.clip(new_array_y, 0, last_frame.shape[0] - 1)
                new_array_x = np.clip(new_array_x, 0, last_frame.shape[1] - 1)
                logger.debug(f"Clipped coordinates used for cluster {uid}")
            
            # Update the forecast frame by transferring intensity values from original to forecast positions            
            forecast_frame[new_array_y, new_array_x] = last_frame[array_y, array_x]
        
        logger.info(f"Processed {processed_clusters} clusters for horizon {horizon}")
        
        # Add completed forecast frame to the list
        forecast_frames.append(forecast_frame)
        logger.debug(f"Added forecast frame for horizon {horizon} to results list")
        
        # Calculate stats for logging
        non_zero_points = np.count_nonzero(forecast_frame)
        min_val = forecast_frame[forecast_frame > 0].min() if non_zero_points > 0 else 0
        max_val = forecast_frame.max()
        logger.info(f"Forecast stats - Non-zero points: {non_zero_points}, Min: {min_val:.2f}, Max: {max_val:.2f}")
        
        # Save forecast as NetCDF if enabled
        if save_forecast:
            try:
                # Calculate forecast timestamp
                forecast_time = last_timestamp + datetime.timedelta(minutes=delta_time * horizon)
                
                # Create filename in the required format
                filename = (f"pyfortracc_forecast_{last_timestamp.strftime('%Y%m%d_%H%M')}_"
                            f"{forecast_time.strftime('%Y%m%d_%H%M')}.nc")
                filepath = os.path.join(forecast_dir, filename)
                logger.info(f"Saving forecast to: {filepath}")
                
                # Hours since reference date
                reference_date = datetime.datetime(1970, 1, 1)
                hours_since_reference = (forecast_time - reference_date).total_seconds() / 3600.0
                
                # Create xarray Dataset
                ds = xr.Dataset(
                    data_vars={
                        "forecast": (["time", "lat", "lon"], forecast_frame[np.newaxis, :, :])
                    },
                    coords={
                        "time": ([hours_since_reference]),
                        "lat": lats,
                        "lon": lons
                    },
                    attrs={
                        "title": "Forecast based on persistence",
                        "source": "PyForTraCC",
                        "reference_time": last_timestamp.strftime('%Y-%m-%d %H:%M'),
                        "forecast_time": forecast_time.strftime('%Y-%m-%d %H:%M'),
                        "forecast_horizon": horizon,
                        "delta_time_minutes": delta_time
                    }
                )
                
                # Set variable attributes
                ds["forecast"].attrs = {
                    "long_name": "Forecast based on persistence",
                    "units": f"{variable_unit}",
                    "missing_value": -9999.0
                }
                
                # Set coordinate attributes
                ds["time"].attrs = {
                    "long_name": "time",
                    "units": f"hours since 1970-01-01 00:00:00"
                }
                
                ds["lat"].attrs = {
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y"
                }
                
                ds["lon"].attrs = {
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X"
                }
                
                # Save as NetCDF with encoding options
                encoding = {
                    "forecast": {
                        "zlib": True,
                        "complevel": 4,
                        "_FillValue": -9999.0
                    },
                    "time": {"dtype": "float64"},
                    "lat": {"dtype": "float64"},
                    "lon": {"dtype": "float64"}
                }
                
                ds.to_netcdf(
                    filepath,
                    format="NETCDF4_CLASSIC",
                    encoding=encoding
                )
                logger.info(f"Successfully saved forecast for horizon {horizon}")
                
            except Exception as e:
                logger.error(f"Error saving forecast for horizon {horizon}: {str(e)}")
    
    # Convert list of frames to numpy array
    forecast_frames = np.array(forecast_frames)
    logger.info(f"Completed forecasting. Generated {len(forecast_frames)} forecast frames")
    logger.info(f"Final forecast array shape: {forecast_frames.shape}")
    
    return forecast_frames