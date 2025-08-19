import os
import sys
import logging
import yaml
import argparse

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_pipeline.csv_generators.netcdf_to_csv import NetCDFConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeolocationsGenerator")

def generate_geolocation_csvs(netcdf_file, config_file=None, output_dir=None, 
                             variables=None, by_date=False):
    """
    Generate CSV files for geolocation time series from NetCDF file
    
    Args:
        netcdf_file: Path to NetCDF file
        config_file: Path to configuration file
        output_dir: Output directory (overrides config)
        variables: List of variables to process (overrides config)
        by_date: Whether to process by date ranges in config
    
    Returns:
        Dictionary of generated file paths
    """
    try:
        # Load configuration
        config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        
        # Override with function parameters
        if output_dir:
            config['output_dir'] = output_dir
        
        # Create converter and load file
        converter = NetCDFConverter(config)
        converter.load_netcdf(netcdf_file)
        
        created_files = {}
        
        # Determine which variables to process
        vars_to_process = variables or config.get('variables', [])
        if not vars_to_process:
            # If no variables specified, get available ones
            all_vars = converter.list_variables()
            vars_to_process = [var['name'] for var in all_vars]
            
        # Process each variable
        for var in vars_to_process:
            logger.info(f"Processing variable: {var}")
            
            if by_date and 'date_ranges' in config:
                var_files = converter.process_by_date_folder(
                    var,
                    date_ranges=config['date_ranges']
                )
            else:
                var_files = converter.process_all_locations(var)
                
            created_files[var] = var_files
            
        return created_files
        
    except Exception as e:
        logger.error(f"Error generating geolocation CSVs: {str(e)}")
        raise

def main():
    """Main function when run as script"""
    parser = argparse.ArgumentParser(description="Generate CSV files for geolocation time series")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--config", "-c", default="config/netcdf_processing.yaml", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output directory (overrides config)")
    parser.add_argument("--variables", "-v", nargs="+", help="Variables to process (overrides config)")
    parser.add_argument("--by-date", "-d", action="store_true", help="Process by date ranges in config")
    
    args = parser.parse_args()
    
    try:
        generate_geolocation_csvs(
            args.input,
            config_file=args.config,
            output_dir=args.output,
            variables=args.variables,
            by_date=args.by_date
        )
        logger.info("CSV generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
