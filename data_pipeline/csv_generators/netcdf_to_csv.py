import xarray as xr
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class NetCDFConverter:
    """Converts NetCDF files to CSV time series data files"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.dataset = None
        self.output_dir = self.config.get('output_dir', 'outputs/generated_data')
        
    def load_netcdf(self, file_path):
        """Load NetCDF file and return xarray Dataset"""
        try:
            logger.info(f"Loading NetCDF file: {file_path}")
            self.dataset = xr.open_dataset(file_path)
            logger.info(f"Loaded dataset with variables: {list(self.dataset.data_vars)}")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading NetCDF file: {str(e)}")
            raise
            
    def list_variables(self):
        """List all variables in the loaded dataset"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_netcdf first.")
        
        variables = []
        for var_name in self.dataset.data_vars:
            var = self.dataset[var_name]
            dims = ", ".join([f"{dim}: {var.sizes[dim]}" for dim in var.dims])
            variables.append({
                'name': var_name,
                'dimensions': var.dims,
                'shape': var.shape,
                'description': var.attrs.get('long_name', 'No description'),
                'units': var.attrs.get('units', 'No units'),
                'dims_detail': dims
            })
        return variables
    
    def extract_locations(self, variable_name):
        """Extract all lat/lon locations for a variable"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_netcdf first.")
            
        if variable_name not in self.dataset:
            raise ValueError(f"Variable {variable_name} not found in dataset")
            
        var = self.dataset[variable_name]
        
        # Check if lat and lon are in the dimensions
        locations = []
        if 'lat' in var.dims and 'lon' in var.dims:
            for lat in var.lat.values:
                for lon in var.lon.values:
                    locations.append((lat, lon))
        
        return locations
    
    def create_csv_for_location(self, variable_name, lat, lon, output_dir=None, filename=None):
        """Create a CSV file for a specific variable and location"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_netcdf first.")
            
        try:
            # Select data for this location
            var_data = self.dataset[variable_name].sel(lat=lat, lon=lon)
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': var_data.time.values,
                variable_name: var_data.values
            })
            
            # Set date as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Apply processing if needed
            if self.config.get('fill_missing', True):
                df = df.fillna(method=self.config.get('fill_method', 'ffill'))
            
            if self.config.get('resample_freq'):
                df = df.resample(self.config.get('resample_freq')).mean()
                df = df.fillna(method=self.config.get('fill_method', 'ffill'))
            
            # Create output directory if needed
            if output_dir is None:
                output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename if not provided
            if filename is None:
                lat_str = f"{lat:.2f}".replace('.', 'p').replace('-', 'neg')
                lon_str = f"{lon:.2f}".replace('.', 'p').replace('-', 'neg')
                filename = f"{variable_name}_lat_{lat_str}_lon_{lon_str}.csv"
            
            # Save to CSV
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path)
            
            logger.info(f"Created CSV file at {output_path} with {len(df)} rows")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating CSV for location (lat={lat}, lon={lon}): {str(e)}")
            raise
    
    def process_all_locations(self, variable_name, output_dir=None, folder_by_variable=False, date_filtered=None):
        """Process all locations for a specific variable"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_netcdf first.")
            
        try:
            # Set output directory
            if output_dir is None:
                output_dir = self.output_dir
            
            # Create variable subdirectory if needed
            if folder_by_variable:
                output_dir = os.path.join(output_dir, variable_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Filter by date if specified
            if date_filtered:
                start_date = date_filtered.get('start')
                end_date = date_filtered.get('end')
                
                if start_date and end_date:
                    logger.info(f"Filtering data from {start_date} to {end_date}")
                    self.dataset = self.dataset.sel(time=slice(start_date, end_date))
            
            # Get all locations
            locations = self.extract_locations(variable_name)
            total_locations = len(locations)
            
            logger.info(f"Processing {total_locations} locations for variable {variable_name}")
            
            # Create CSV for each location
            created_files = []
            for i, (lat, lon) in enumerate(locations):
                try:
                    file_path = self.create_csv_for_location(
                        variable_name, lat, lon, output_dir
                    )
                    created_files.append(file_path)
                    
                    # Log progress
                    if (i+1) % 10 == 0 or (i+1) == total_locations:
                        logger.info(f"Processed {i+1}/{total_locations} locations")
                        
                except Exception as e:
                    logger.warning(f"Error processing location {i+1}/{total_locations}: {str(e)}")
                    continue
            
            logger.info(f"Created {len(created_files)} CSV files for variable {variable_name}")
            return created_files
            
        except Exception as e:
            logger.error(f"Error processing all locations: {str(e)}")
            raise
    
    def process_by_date_folder(self, variable_name, output_dir=None, date_ranges=None):
        """Process a variable by date ranges, creating separate folders"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_netcdf first.")
            
        if not date_ranges:
            logger.warning("No date ranges provided, using entire dataset")
            return self.process_all_locations(variable_name, output_dir)
            
        try:
            # Set output directory
            if output_dir is None:
                output_dir = self.output_dir
                
            created_files = {}
            
            # Process each date range
            for date_range in date_ranges:
                range_name = date_range.get('name', 'unknown_range')
                start_date = date_range.get('start')
                end_date = date_range.get('end')
                
                logger.info(f"Processing range: {range_name} ({start_date} to {end_date})")
                
                # Create a folder for this date range
                range_dir = os.path.join(output_dir, range_name)
                
                # Process with date filtering
                files = self.process_all_locations(
                    variable_name,
                    output_dir=range_dir,
                    date_filtered={'start': start_date, 'end': end_date}
                )
                
                created_files[range_name] = files
                
            return created_files
            
        except Exception as e:
            logger.error(f"Error processing by date folder: {str(e)}")
            raise

# Command-line interface if run directly
if __name__ == "__main__":
    import argparse
    import yaml
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert NetCDF files to CSV time series')
    parser.add_argument('--input', '-i', required=True, help='Input NetCDF file')
    parser.add_argument('--output', '-o', default='outputs/generated_data', help='Output directory')
    parser.add_argument('--variable', '-v', help='Variable to process')
    parser.add_argument('--list', '-l', action='store_true', help='List variables and exit')
    parser.add_argument('--config', '-c', help='Config YAML file')
    parser.add_argument('--by-date', '-d', action='store_true', help='Process by date ranges in config')
    
    args = parser.parse_args()
    
    # Load config if specified
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
    
    # Set output directory
    config['output_dir'] = args.output
    
    # Create converter and load file
    converter = NetCDFConverter(config)
    converter.load_netcdf(args.input)
    
    # List variables if requested
    if args.list:
        variables = converter.list_variables()
        print("\nVariables in the NetCDF file:")
        for i, var in enumerate(variables, 1):
            print(f"{i}. {var['name']} ({var['dims_detail']})")
            print(f"   Description: {var['description']}")
            print(f"   Units: {var['units']}")
        exit(0)
    
    # Process the variable
    if args.variable:
        if args.by_date and 'date_ranges' in config:
            converter.process_by_date_folder(
                args.variable,
                date_ranges=config['date_ranges']
            )
        else:
            converter.process_all_locations(args.variable)
    else:
        logger.error("No variable specified. Use --variable or --list to see available variables.")
