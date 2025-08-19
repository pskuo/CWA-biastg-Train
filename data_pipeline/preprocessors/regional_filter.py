import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.basemap import Basemap

logger = logging.getLogger(__name__)

class RegionalFilter:
    """
    Filters oceanographic data by geographic regions.
    
    This class provides tools to filter CSV files or datasets based on 
    latitude and longitude coordinates, selecting data from specified regions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the regional filter.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.regions = self.config.get('regions', [])
        self.output_dir = self.config.get('output_dir', 'outputs/generated_data/filtered_regions')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_region(self, name, lat_min, lat_max, lon_min, lon_max):
        """
        Add a region definition.
        
        Args:
            name (str): Name of the region
            lat_min (float): Minimum latitude
            lat_max (float): Maximum latitude
            lon_min (float): Minimum longitude
            lon_max (float): Maximum longitude
            
        Returns:
            self: Returns instance for method chaining
        """
        region = {
            'name': name,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        self.regions.append(region)
        logger.info(f"Added region: {name} (Lat: {lat_min} to {lat_max}, Lon: {lon_min} to {lon_max})")
        
        return self
    
    def get_region_by_name(self, region_name):
        """
        Get a region definition by name.
        
        Args:
            region_name (str): Name of the region to get
            
        Returns:
            dict: Region definition or None if not found
        """
        for region in self.regions:
            if region['name'] == region_name:
                return region
        
        logger.warning(f"Region not found: {region_name}")
        return None
    
    def filter_csv_files(self, csv_dir, region_name=None, output_subdir=None):
        """
        Filter multiple CSV files based on latitude and longitude in filenames.
        
        Args:
            csv_dir (str): Directory containing CSV files with lat/lon in filenames
            region_name (str): Name of the region to filter by
            output_subdir (str, optional): Subdirectory for output files
            
        Returns:
            list: Paths to filtered CSV files
        """
        # Get the region definition
        if region_name:
            region = self.get_region_by_name(region_name)
            if not region:
                raise ValueError(f"Region not found: {region_name}")
        else:
            if not self.regions:
                raise ValueError("No regions defined")
            region = self.regions[0]
            region_name = region['name']
        
        # Create output directory
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
        else:
            output_dir = os.path.join(self.output_dir, region_name)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all CSV files
        csv_files = []
        for file in os.listdir(csv_dir):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(csv_dir, file))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {csv_dir}")
            return []
        
        # Process each file
        filtered_files = []
        
        for file_path in csv_files:
            try:
                # Extract lat/lon from filename
                filename = os.path.basename(file_path)
                
                # Assume format like variable_lat_X.XX_lon_Y.YY.csv
                lat_lon = self._extract_lat_lon_from_filename(filename)
                
                if not lat_lon:
                    logger.warning(f"Could not extract lat/lon from filename: {filename}")
                    continue
                    
                lat, lon = lat_lon
                
                # Check if location is in the region
                if (region['lat_min'] <= lat <= region['lat_max'] and 
                    region['lon_min'] <= lon <= region['lon_max']):
                    
                    # Copy file to output directory
                    output_path = os.path.join(output_dir, filename)
                    
                    # Read and write the file (this allows for future processing if needed)
                    df = pd.read_csv(file_path)
                    df.to_csv(output_path, index=False)
                    
                    filtered_files.append(output_path)
                    logger.debug(f"Filtered file: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Filtered {len(filtered_files)} files for region {region_name}")
        return filtered_files
    
    def filter_dataframe(self, df, region_name=None, lat_col='lat', lon_col='lon'):
        """
        Filter a DataFrame based on lat/lon columns.
        
        Args:
            df (pd.DataFrame): DataFrame containing lat/lon columns
            region_name (str): Name of the region to filter by
            lat_col (str): Name of latitude column
            lon_col (str): Name of longitude column
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Get the region definition
        if region_name:
            region = self.get_region_by_name(region_name)
            if not region:
                raise ValueError(f"Region not found: {region_name}")
        else:
            if not self.regions:
                raise ValueError("No regions defined")
            region = self.regions[0]
        
        # Check if columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"DataFrame must contain {lat_col} and {lon_col} columns")
        
        # Apply filter
        filtered_df = df[
            (df[lat_col] >= region['lat_min']) & 
            (df[lat_col] <= region['lat_max']) &
            (df[lon_col] >= region['lon_min']) & 
            (df[lon_col] <= region['lon_max'])
        ]
        
        logger.info(f"Filtered DataFrame from {len(df)} to {len(filtered_df)} rows for region {region['name']}")
        return filtered_df
    
    def plot_regions(self, output_path=None, figsize=(12, 8)):
        """
        Plot all defined regions on a map.
        
        Args:
            output_path (str, optional): Path to save the plot
            figsize (tuple): Figure size
            
        Returns:
            str: Path to the saved plot
        """
        if not self.regions:
            logger.warning("No regions defined to plot")
            return None
        
        plt.figure(figsize=figsize)
        
        # Create a global map
        m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, 
                   llcrnrlon=-180, urcrnrlon=180, resolution='c')
        
        # Draw map features
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='coral', lake_color='aqua')
        m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1])
        
        # Plot each region
        colors = plt.cm.tab10.colors
        for i, region in enumerate(self.regions):
            color = colors[i % len(colors)]
            
            # Convert lat/lon to map coordinates
            x1, y1 = m(region['lon_min'], region['lat_min'])
            x2, y2 = m(region['lon_max'], region['lat_max'])
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1, region['name'], color=color, fontweight='bold',
                    backgroundcolor='white', fontsize=10)
        
        plt.title('Defined Geographic Regions')
        
        # Save plot if path provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'regions_map.png')
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Regions map saved to {output_path}")
        return output_path
    
    def _extract_lat_lon_from_filename(self, filename):
        """
        Extract latitude and longitude from a filename.
        
        Args:
            filename (str): Filename to parse
            
        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        try:
            # Handle different filename formats
            if 'lat_' in filename and 'lon_' in filename:
                # Format: variable_lat_X.XX_lon_Y.YY.csv
                parts = filename.split('_')
                
                lat_idx = parts.index('lat') + 1
                lon_idx = parts.index('lon') + 1
                
                # Handle replacements like 'p' for '.' and 'neg' for '-'
                lat_str = parts[lat_idx].replace('p', '.').replace('neg', '-')
                lon_str = parts[lon_idx].split('.')[0].replace('p', '.').replace('neg', '-')
                
                return float(lat_str), float(lon_str)
                
            # Add more format handlers as needed
                
        except Exception as e:
            logger.debug(f"Error extracting lat/lon from {filename}: {str(e)}")
            
        return None
