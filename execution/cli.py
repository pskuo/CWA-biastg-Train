#!/usr/bin/env python
import os
import sys
import click
import yaml
import logging
import pandas as pd
import json
import numpy as np
from datetime import datetime
import importlib
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TimeSeriesCLI')

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Add file handler to save logs
file_handler = logging.FileHandler(f'logs/timeseries_cli_{datetime.now().strftime("%Y%m%d")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def load_config(config_path):
    """Load a configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {str(e)}")
        raise

def get_available_models():
    """Get list of available models"""
    return [
        'arima', 'sarimax', 'lstm', 'rnn', 
        'time_series_transformer_i', 'time_series_transformer_ii',
        'time_series_transformer_iii', 'time_series_imaging', 'bilstm'
    ]

def get_model_adapter(model_name, config=None):
    """Dynamically import and initialize the appropriate model adapter"""
    try:
        module_path = f"models.core.{model_name}.adapter"
        adapter_module = importlib.import_module(module_path)
        
        # Convert model_name to CamelCase for class name
        class_name_parts = [part.capitalize() for part in model_name.split('_')]
        class_name = ''.join(class_name_parts) + 'Adapter'
        
        adapter_class = getattr(adapter_module, class_name)
        adapter_instance = adapter_class(config)
        
        logger.info(f"Loaded adapter for model: {model_name}")
        return adapter_instance
    except Exception as e:
        logger.error(f"Error loading model adapter: {str(e)}")
        raise ValueError(f"Failed to load model adapter for {model_name}: {str(e)}")

def plot_forecast_comparison(historical_data, forecast_dict, output_path):
    """Create plot comparing multiple model forecasts"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data.values, 
                label='Historical Data', color='black', linewidth=2)
        
        # Plot each forecast with a different color
        colors = plt.cm.tab10.colors
        for i, (model_name, forecast) in enumerate(forecast_dict.items()):
            color = colors[i % len(colors)]
            plt.plot(forecast.index, forecast.values, 
                    label=f'{model_name} Forecast', 
                    color=color, linestyle='--')
            
        # Add labels and formatting
        plt.title('Forecast Model Comparison')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Error creating comparison plot: {str(e)}")
        raise

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Time Series Forecasting CLI for oceanographic data.
    
    This tool provides functionality for:
    1. Converting NetCDF files to CSV time series data
    2. Training and evaluating forecasting models
    3. Generating forecasts for future time periods
    4. Applying bias correction to improve forecast accuracy
    """
    pass

@cli.command(name="extract-data")
@click.option("--input", "-i", required=True, help="Input NetCDF file path")
@click.option("--output", "-o", default="outputs/generated_data", help="Output directory")
@click.option("--config", "-c", default="config/netcdf_processing.yaml", help="Configuration file")
@click.option("--variable", "-v", multiple=True, help="Variables to process (can be used multiple times)")
@click.option("--by-date/--no-by-date", default=False, help="Process by date ranges in config")
@click.option("--list-vars", "-l", is_flag=True, help="List variables in the NetCDF file and exit")
@click.option("--region", "-r", help="Region to filter (must be defined in config)")
def extract_data(input, output, config, variable, by_date, list_vars, region):
    """Convert NetCDF files to CSV time series files."""
    try:
        # Load configuration
        cfg = load_config(config)
        
        # Set output directory if provided
        if output:
            cfg['output_dir'] = output
        
        # Import here to avoid circular imports
        from data_pipeline.csv_generators.netcdf_to_csv import NetCDFConverter
        
        # Create converter and load file
        converter = NetCDFConverter(cfg)
        converter.load_netcdf(input)
        
        # List variables if requested
        if list_vars:
            variables = converter.list_variables()
            click.echo("\nVariables in the NetCDF file:")
            for i, var in enumerate(variables, 1):
                click.echo(f"{i}. {var['name']} ({var['dims_detail']})")
                click.echo(f"   Description: {var['description']}")
                click.echo(f"   Units: {var['units']}")
            return
        
        # Process region filter if specified
        if region:
            if 'regions' not in cfg or not any(r.get('name') == region for r in cfg.get('regions', [])):
                click.echo(f"Error: Region '{region}' not defined in config", err=True)
                return
            
            # Find the region config
            region_config = next(r for r in cfg.get('regions', []) if r.get('name') == region)
            cfg['current_region'] = region_config
            
            # Create region directory
            region_dir = os.path.join(cfg['output_dir'], region)
            os.makedirs(region_dir, exist_ok=True)
            cfg['output_dir'] = region_dir
            
            click.echo(f"Filtering data for region: {region}")
        
        # Process specified variables
        if variable:
            for var in variable:
                click.echo(f"Processing variable: {var}")
                if by_date and 'date_ranges' in cfg:
                    converter.process_by_date_folder(var, date_ranges=cfg.get('date_ranges'))
                else:
                    converter.process_all_locations(var)
        else:
            # Process variables from config
            for var in cfg.get('variables', []):
                click.echo(f"Processing variable: {var}")
                if by_date and 'date_ranges' in cfg:
                    converter.process_by_date_folder(var, date_ranges=cfg.get('date_ranges'))
                else:
                    converter.process_all_locations(var)
                
        click.echo("Data extraction completed successfully")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Data extraction error: {str(e)}", exc_info=True)

@cli.command(name="train-model")
@click.option("--model", "-m", required=True, help="Model to train (e.g., arima, lstm)")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file or directory")
@click.option("--config", "-c", default=None, help="Model configuration file")
@click.option("--output", "-o", default=None, help="Output directory for model results")
@click.option("--target-column", "-t", default=None, help="Target column for forecasting")
@click.option("--horizon", "-h", default=None, type=int, help="Forecast horizon (steps ahead)")
@click.option("--eval-metric", "-e", default="rmse", help="Evaluation metric (rmse, mae, mape)")
def train_model(model, data_path, config, output, target_column, horizon, eval_metric):
    """Train a time series forecasting model on the provided data."""
    try:
        # Validate model choice
        available_models = get_available_models()
        if model not in available_models:
            click.echo(f"Error: Model '{model}' not available. Choose from: {', '.join(available_models)}", err=True)
            return
        
        # Load model config file if not provided
        if config is None:
            config = f"config/model_configs/{model}.yaml"
        
        # Load configuration
        try:
            model_config = load_config(config)
        except Exception as e:
            click.echo(f"Error loading model config: {str(e)}", err=True)
            return
        
        # Update configuration with CLI parameters
        if target_column:
            model_config['target_column'] = target_column
        
        if horizon:
            model_config['forecast_horizon'] = horizon
        
        if output:
            model_config['output_dir'] = output
            model_config['results_dir'] = os.path.join(output, 'results')
        
        # Create output directories
        for dir_key in ['output_dir', 'results_dir', 'model_dir']:
            if dir_key in model_config:
                os.makedirs(model_config[dir_key], exist_ok=True)
        
        # Get the model adapter
        model_adapter = get_model_adapter(model, model_config)
        
        # Check if input is a directory or file
        if os.path.isdir(data_path):
            # Find all CSV files in the directory
            csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                        if f.endswith('.csv')]
            
            if not csv_files:
                click.echo(f"No CSV files found in {data_path}", err=True)
                return
            
            # Train on first file by default
            data_file = csv_files[0]
            click.echo(f"Training on {data_file} (first file in directory)")
        else:
            data_file = data_path
        
        # Run the model
        click.echo(f"Training model {model} on data from {data_file}")
        results = model_adapter.run_pipeline(data_file)
        
        # Save results summary
        summary_path = os.path.join(model_config.get('results_dir', 'outputs/model_results'), 'summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump({
                'model': model,
                'data_file': data_file,
                'config_file': config,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'rmse': float(results.attrs.get('rmse', 0)),
                    'mae': float(results.attrs.get('mae', 0)), 
                    'mape': float(results.attrs.get('mape', 0))
                },
                'forecast_horizon': model_config.get('forecast_horizon', 10)
            }, f, indent=2)
        
        # Print evaluation metrics
        click.echo("\nModel Training Results:")
        click.echo(f"RMSE: {results.attrs.get('rmse', 'N/A')}")
        click.echo(f"MAE: {results.attrs.get('mae', 'N/A')}")
        click.echo(f"MAPE: {results.attrs.get('mape', 'N/A')}%")
        click.echo(f"\nResults saved to {model_config.get('results_dir', 'outputs/model_results')}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Model training error: {str(e)}", exc_info=True)

@cli.command(name="forecast")
@click.option("--model", "-m", required=True, help="Model to use for forecasting")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file")
@click.option("--config", "-c", default=None, help="Model configuration file")
@click.option("--output", "-o", default=None, help="Output directory for forecast results")
@click.option("--horizon", "-h", default=10, type=int, help="Forecast horizon (steps ahead)")
@click.option("--start-date", "-s", default=None, help="Start date for forecast (YYYY-MM-DD)")
@click.option("--frequency", "-f", default='D', help="Forecast frequency (D=daily, W=weekly, etc.)")
@click.option("--plot/--no-plot", default=True, help="Generate forecast plot")
@click.option("--apply-bias-correction/--no-bias-correction", default=False, help="Apply bias correction")
@click.option("--correction-method", default="linear_scaling", 
              type=click.Choice(['linear_scaling', 'quantile_mapping', 'distribution_mapping', 
                               'delta_change', 'ratio_adjustment', 'regression']),
              help="Bias correction method to use")
@click.option("--reference-data", default=None, help="Reference data for bias correction")
def generate_forecast(model, data_path, config, output, horizon, start_date, frequency, plot, 
                     apply_bias_correction, correction_method, reference_data):
    """Generate forecasts using a trained model."""
    try:
        # Validate model choice
        available_models = get_available_models()
        if model not in available_models:
            click.echo(f"Error: Model '{model}' not available. Choose from: {', '.join(available_models)}", err=True)
            return
        
        # Load model config file if not provided
        if config is None:
            config = f"config/model_configs/{model}.yaml"
        
        # Load configuration
        model_config = load_config(config)
        
        # Update configuration with CLI parameters
        model_config['forecast_horizon'] = horizon
        
        if start_date:
            model_config['forecast_start_date'] = start_date
        
        model_config['forecast_frequency'] = frequency
        
        if output:
            model_config['output_dir'] = output
            model_config['results_dir'] = os.path.join(output, 'results')
        
        # Create output directories
        for dir_key in ['output_dir', 'results_dir']:
            if dir_key in model_config:
                os.makedirs(model_config[dir_key], exist_ok=True)
        
        # Get the model adapter
        model_adapter = get_model_adapter(model, model_config)
        
        # Run the forecasting
        click.echo(f"Generating {horizon}-step forecast using {model}")
        results = model_adapter.run_pipeline(data_path)
        
        # Get forecast output path
        output_dir = model_config.get('results_dir', 'outputs/model_results')
        forecast_path = os.path.join(output_dir, f'{model}_forecast.csv')
        
        # Save forecast to CSV
        results.to_csv(forecast_path)
        click.echo(f"Forecast generated and saved to {forecast_path}")
        
        # Apply bias correction if requested
        if apply_bias_correction:
            if not reference_data:
                click.echo("Warning: No reference data provided for bias correction. Skipping bias correction.")
            else:
                from data_pipeline.preprocessors.bias_correction import BiasCorrection
                
                click.echo(f"Applying {correction_method} bias correction...")
                
                # Load reference data
                try:
                    ref_data = pd.read_csv(reference_data, parse_dates=['date'], index_col='date')
                except Exception as e:
                    click.echo(f"Error loading reference data: {str(e)}", err=True)
                    return
                
                # Setup bias correction
                bias_config = {
                    'method': correction_method,
                    'output_dir': os.path.join(output_dir, 'bias_corrected')
                }
                
                # Create bias corrector
                bias_corrector = BiasCorrection(bias_config)
                
                # Get the target column
                target_col = model_config.get('target_column')
                if not target_col and 'forecast' in results.columns:
                    target_col = 'forecast'
                
                # Extract forecast data
                forecast_data = results[target_col] if target_col in results.columns else results
                
                # Extract reference data
                ref_series = ref_data[target_col] if target_col in ref_data.columns else ref_data.iloc[:, 0]
                
                # Fit and apply correction
                bias_corrector.fit(ref_series, forecast_data)
                corrected_data = bias_corrector.transform(forecast_data)
                
                # Save corrected forecast
                corrected_path = os.path.join(output_dir, f'{model}_forecast_corrected.csv')
                corrected_df = pd.DataFrame({
                    'forecast': forecast_data,
                    'forecast_corrected': corrected_data
                })
                corrected_df.to_csv(corrected_path)
                click.echo(f"Bias-corrected forecast saved to {corrected_path}")
                
                # Plot with corrected data if requested
                if plot:
                    # Load original data for comparison
                    original_data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
                    target_column = model_config.get('target_column')
                    
                    if target_column not in original_data.columns:
                        target_column = original_data.columns[0]
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 30 points)
                    ax.plot(original_data.index[-30:], original_data[target_column].iloc[-30:], 
                        label='Historical Data', color='blue')
                    
                    # Plot forecasts
                    ax.plot(results.index, results[target_col] if target_col in results.columns else results, 
                        label='Original Forecast', color='red', linestyle='--')
                        
                    ax.plot(corrected_data.index, corrected_data, 
                        label='Corrected Forecast', color='green', linestyle='-.')
                    
                    ax.set_title(f'{model.upper()} Forecast with Bias Correction (Horizon: {horizon})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(target_column)
                    ax.legend()
                    ax.grid(True)
                    
                    # Save plot
                    plot_path = os.path.join(output_dir, f'{model}_forecast_corrected_plot.png')
                    plt.savefig(plot_path)
                    plt.close()
                    
                    click.echo(f"Corrected forecast plot saved to {plot_path}")
        
        # Generate plot if requested and no bias correction was applied
        elif plot:
            # Load original data for comparison
            original_data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            target_column = model_config.get('target_column')
            
            if target_column not in original_data.columns:
                target_column = original_data.columns[0]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data (last 30 points)
            ax.plot(original_data.index[-30:], original_data[target_column].iloc[-30:], 
                   label='Historical Data', color='blue')
            
            # Plot forecast
            target_col = model_config.get('target_column', 'forecast')
            forecast_data = results[target_col] if target_col in results.columns else results
            
            ax.plot(forecast_data.index, forecast_data, 
                   label='Forecast', color='red', linestyle='--')
            
            ax.set_title(f'{model.upper()} Forecast (Horizon: {horizon})')
            ax.set_xlabel('Date')
            ax.set_ylabel(target_column)
            ax.legend()
            ax.grid(True)
            
            # Save plot
            plot_path = os.path.join(output_dir, f'{model}_forecast_plot.png')
            plt.savefig(plot_path)
            plt.close()
            
            click.echo(f"Forecast plot saved to {plot_path}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Forecast generation error: {str(e)}", exc_info=True)

@cli.command(name="compare-models")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file")
@click.option("--models", "-m", required=True, multiple=True, help="Models to compare")
@click.option("--output", "-o", default="outputs/model_results/comparison", help="Output directory")
@click.option("--horizon", "-h", default=10, type=int, help="Forecast horizon for all models")
@click.option("--metric", default="rmse", help="Primary metric for comparison (rmse, mae, mape)")
@click.option("--apply-bias-correction/--no-bias-correction", default=False, 
              help="Apply bias correction to all models")
@click.option("--correction-method", default="linear_scaling", 
              help="Bias correction method to use if correction enabled")
@click.option("--reference-data", default=None, help="Reference data for bias correction")
def compare_models(data_path, models, output, horizon, metric, apply_bias_correction, 
                  correction_method, reference_data):
    """Compare performance of multiple forecasting models."""
    try:
        # Validate model choices
        available_models = get_available_models()
        for model in models:
            if model not in available_models:
                click.echo(f"Error: Model '{model}' not available. Choose from: {', '.join(available_models)}", err=True)
                return
        
        # Create output directory
        os.makedirs(output, exist_ok=True)
        
        # Initialize results collection
        results = {
            'models': [],
            'rmse': [],
            'mae': [],
            'mape': [],
            'forecast_data': {},
            'corrected_forecast_data': {},
            'training_time': []
        }
        
        # Load reference data for bias correction if needed
        if apply_bias_correction and reference_data:
            try:
                ref_data = pd.read_csv(reference_data, parse_dates=['date'], index_col='date')
                click.echo(f"Loaded reference data for bias correction: {reference_data}")
            except Exception as e:
                click.echo(f"Error loading reference data: {str(e)}. Bias correction will be skipped.", err=True)
                apply_bias_correction = False
        else:
            apply_bias_correction = False
        
        # Train and evaluate each model
        for model_name in models:
            click.echo(f"\nEvaluating model: {model_name}")
            
            # Load model config
            config_path = f"config/model_configs/{model_name}.yaml"
            try:
                model_config = load_config(config_path)
            except Exception as e:
                click.echo(f"Error loading config for {model_name}: {str(e)}", err=True)
                continue
            
            # Update config for comparison
            model_config['forecast_horizon'] = horizon
            model_config['output_dir'] = os.path.join(output, model_name)
            model_config['results_dir'] = os.path.join(output, model_name, 'results')
            
            # Create model-specific output directories
            for dir_key in ['output_dir', 'results_dir']:
                if dir_key in model_config:
                    os.makedirs(model_config[dir_key], exist_ok=True)
            
            # Get the model adapter
            model_adapter = get_model_adapter(model_name, model_config)
            
            # Track training time
            start_time = datetime.now()
            
            # Run the model
            model_results = model_adapter.run_pipeline(data_path)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            target_col = model_config.get('target_column', 'forecast')
            forecast_data = model_results[target_col] if target_col in model_results.columns else model_results
            
            # Apply bias correction if requested
            if apply_bias_correction:
                try:
                    from data_pipeline.preprocessors.bias_correction import BiasCorrection
                    
                    # Setup bias correction
                    bias_config = {
                        'method': correction_method,
                        'output_dir': os.path.join(output, model_name, 'bias_corrected')
                    }
                    
                    # Create bias corrector
                    bias_corrector = BiasCorrection(bias_config)
                    
                    # Get reference column
                    ref_col = target_col if target_col in ref_data.columns else ref_data.columns[0]
                    
                    # Fit and apply correction
                    bias_corrector.fit(ref_data[ref_col], forecast_data)
                    corrected_data = bias_corrector.transform(forecast_data)
                    
                    # Calculate metrics for corrected data
                    corrected_rmse = np.sqrt(((corrected_data - ref_data[ref_col]) ** 2).mean())
                    corrected_mae = (corrected_data - ref_data[ref_col]).abs().mean()
                    corrected_mape = 100 * ((corrected_data - ref_data[ref_col]).abs() / ref_data[ref_col]).mean()
                    
                    # Store corrected forecast data
                    results['corrected_forecast_data'][model_name] = corrected_data
                    
                    # Use corrected metrics
                    model_rmse = corrected_rmse
                    model_mae = corrected_mae
                    model_mape = corrected_mape
                    
                    # Save corrected forecast
                    corrected_path = os.path.join(model_config['results_dir'], f'forecast_corrected.csv')
                    corrected_df = pd.DataFrame({
                        'forecast': forecast_data,
                        'forecast_corrected': corrected_data
                    })
                    corrected_df.to_csv(corrected_path)
                    
                    click.echo(f"Applied bias correction to {model_name}")
                    
                except Exception as e:
                    click.echo(f"Error applying bias correction to {model_name}: {str(e)}", err=True)
                    # Fall back to uncorrected metrics
                    model_rmse = float(model_results.attrs.get('rmse', float('inf')))
                    model_mae = float(model_results.attrs.get('mae', float('inf')))
                    model_mape = float(model_results.attrs.get('mape', float('inf')))
            else:
                # Use original metrics
                model_rmse = float(model_results.attrs.get('rmse', float('inf')))
                model_mae = float(model_results.attrs.get('mae', float('inf')))
                model_mape = float(model_results.attrs.get('mape', float('inf')))
            
            # Store results
            results['models'].append(model_name)
            results['rmse'].append(model_rmse)
            results['mae'].append(model_mae)
            results['mape'].append(model_mape)
            results['training_time'].append(training_time)
            results['forecast_data'][model_name] = forecast_data
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'model': results['models'],
            'rmse': results['rmse'],
            'mae': results['mae'],
            'mape': results['mape'],
            'training_time': results['training_time']
        })
        
        # Sort by the chosen metric
        comparison_df = comparison_df.sort_values(metric)
        
        # Save comparison to CSV
        comparison_path = os.path.join(output, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        # Generate comparison plot
        try:
            # Load original data for plotting
            original_data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            target_col = original_data.columns[0]  # Default to first column
            
            # Generate plot path
            plot_comparison_path = os.path.join(output, 'forecast_comparison.png')
            
            # Plot original forecasts
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data (last 30 points)
            ax.plot(original_data.index[-30:], original_data[target_col].iloc[-30:], 
                  label='Historical Data', color='black', linewidth=2)
            
            # Plot each forecast with a different color
            colors = plt.cm.tab10.colors
            for i, model_name in enumerate(results['models']):
                color = colors[i % len(colors)]
                forecast = results['forecast_data'][model_name]
                ax.plot(forecast.index, forecast, 
                       label=f'{model_name} Forecast', 
                       color=color, linestyle='--')
            
            ax.set_title('Forecast Model Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col)
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(plot_comparison_path, dpi=300)
            plt.close()
            
            click.echo(f"Comparison plot saved to {plot_comparison_path}")
            
            # Generate corrected comparison plot if bias correction was applied
            if apply_bias_correction and results['corrected_forecast_data']:
                corrected_plot_path = os.path.join(output, 'forecast_comparison_corrected.png')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                ax.plot(original_data.index[-30:], original_data[target_col].iloc[-30:], 
                      label='Historical Data', color='black', linewidth=2)
                
                # Plot each corrected forecast with a different color
                for i, model_name in enumerate(results['models']):
                    if model_name in results['corrected_forecast_data']:
                        color = colors[i % len(colors)]
                        forecast = results['corrected_forecast_data'][model_name]
                        ax.plot(forecast.index, forecast, 
                               label=f'{model_name} Corrected', 
                               color=color, linestyle='-')
                
                ax.set_title('Bias-Corrected Forecast Comparison')
                ax.set_xlabel('Date')
                ax.set_ylabel(target_col)
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(corrected_plot_path, dpi=300)
                plt.close()
                
                click.echo(f"Corrected comparison plot saved to {corrected_plot_path}")
                
        except Exception as e:
            click.echo(f"Error generating comparison plot: {str(e)}", err=True)
        
        # Print comparison table
        click.echo("\nModel Comparison Results:")
        click.echo(comparison_df.to_string(index=False))
        click.echo(f"\nBest model based on {metric}: {comparison_df.iloc[0]['model']}")
        click.echo(f"\nComparison saved to {comparison_path}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Model comparison error: {str(e)}", exc_info=True)

@cli.command(name="apply-bias-correction")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file or directory")
@click.option("--reference-path", "-r", required=True, help="Path to reference/observed data CSV")
@click.option("--method", "-m", default="linear_scaling", 
              type=click.Choice(['linear_scaling', 'quantile_mapping', 'distribution_mapping', 
                               'delta_change', 'ratio_adjustment', 'regression']),
              help="Bias correction method to use")
@click.option("--output", "-o", default=None, help="Output directory for corrected data")
@click.option("--target-column", "-t", default=None, help="Target column to correct")
@click.option("--plot/--no-plot", default=True, help="Generate comparison plot")
@click.option("--evaluate/--no-evaluate", default=True, help="Evaluate correction performance")
def apply_bias_correction(data_path, reference_path, method, output, target_column, plot, evaluate):
    """Apply bias correction to forecasted or modeled data."""
    try:
        from data_pipeline.preprocessors.bias_correction import BiasCorrection
        
        # Load reference data
        try:
            reference_data = pd.read_csv(reference_path, parse_dates=['date'], index_col='date')
        except Exception as e:
            click.echo(f"Error loading reference data: {str(e)}", err=True)
            return
        
        # Create output directory
        if output:
            output_dir = output
        else:
            output_dir = f"outputs/generated_data/bias_corrected/{method}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create bias correction config
        config = {
            'method': method,
            'target_variable': target_column,
            'output_dir': output_dir
        }
        
        # Initialize bias correction
        bias_corrector = BiasCorrection(config)
        
        # Check if input is directory or file
        if os.path.isdir(data_path):
            # Find CSV files
            csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                         if f.endswith('.csv')]
            
            if not csv_files:
                click.echo(f"No CSV files found in {data_path}", err=True)
                return
                
            # Process first file
            data_file = csv_files[0]
            click.echo(f"Processing first file: {data_file}")
        else:
            data_file = data_path
        
        # Load model data
        try:
            model_data = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
        except Exception as e:
            click.echo(f"Error loading model data: {str(e)}", err=True)
            return
        
        # Set target column if not specified
        if target_column is None:
            if len(model_data.columns) > 0:
                target_column = model_data.columns[0]
                click.echo(f"Using first column as target: {target_column}")
            else:
                click.echo("No columns found in model data", err=True)
                return
        
        # Extract target columns
        if target_column in model_data.columns:
            model_series = model_data[target_column]
        else:
            click.echo(f"Target column {target_column} not found in model data", err=True)
            return
            
        if target_column in reference_data.columns:
            reference_series = reference_data[target_column]
        else:
            click.echo(f"Target column {target_column} not found in reference data", err=True)
            return
        
        # Fit bias correction
        click.echo(f"Fitting bias correction using {method}...")
        bias_corrector.fit(reference_series, model_series)
        
        # Apply correction
        click.echo("Applying bias correction...")
        corrected_data = bias_corrector.transform(model_series)
        
        # Save to CSV
        output_filename = f"bias_corrected_{target_column}_{method}.csv"
        output_path = bias_corrector.save_to_csv(
            corrected_data, 
            original_data=model_series,
            filename=output_filename
        )
        
        click.echo(f"Corrected data saved to: {output_path}")
        
        # Plot comparison if requested
        if plot:
            plot_path = bias_corrector.plot_correction(
                model_series, 
                corrected_data,
                reference_series
            )
            click.echo(f"Comparison plot saved to: {plot_path}")
        
        # Evaluate if requested
        if evaluate:
            click.echo("\nEvaluation Results:")
            eval_results = bias_corrector.evaluate_correction(
                reference_series,
                model_series,
                corrected_data
            )
            
            # Display key metrics
            for metric in ['rmse_improvement_pct', 'mae_improvement_pct', 'bias_reduction_pct']:
                if metric in eval_results:
                    click.echo(f"{metric.replace('_', ' ').title()}: {eval_results[metric]:.2f}%")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Bias correction error: {str(e)}", exc_info=True)

@cli.command(name="filter-region")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file or directory")
@click.option("--region", "-r", required=True, help="Name of region to filter")
@click.option("--config", "-c", default="config/preprocessors/regional_filter.yaml", 
              help="Regional filter configuration file")
@click.option("--output", "-o", default=None, help="Output directory for filtered data")
@click.option("--plot-region/--no-plot", default=True, help="Generate region plot")
def filter_region(data_path, region, config, output, plot_region):
    """Filter data based on geographic regions."""
    try:
        from data_pipeline.preprocessors.regional_filter import RegionalFilter
        
        # Load configuration
        filter_config = load_config(config)
        
        # Set output directory if provided
        if output:
            filter_config['output_dir'] = output
        
        # Create regional filter
        region_filter = RegionalFilter(filter_config)
        
        # Check if region exists
        region_def = region_filter.get_region_by_name(region)
        if not region_def:
            click.echo(f"Region '{region}' not defined in configuration", err=True)
            return
            
        click.echo(f"Filtering data for region: {region}")
        click.echo(f"Boundaries: Lat {region_def['lat_min']} to {region_def['lat_max']}, " +
                   f"Lon {region_def['lon_min']} to {region_def['lon_max']}")
        
        # Filter the data
        filtered_files = region_filter.filter_csv_files(data_path, region)
        
        click.echo(f"Filtered {len(filtered_files)} files for region {region}")
        
        # Generate region plot if requested
        if plot_region:
            try:
                plot_path = region_filter.plot_regions()
                click.echo(f"Region plot saved to: {plot_path}")
            except Exception as e:
                click.echo(f"Error generating region plot: {str(e)}", err=True)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Regional filtering error: {str(e)}", exc_info=True)

@cli.command(name="list-models")
def list_models():
    """List all available forecasting models."""
    models = get_available_models()
    
    click.echo("\nAvailable Time Series Forecasting Models:")
    for i, model in enumerate(models, 1):
        # Get model description if available
        try:
            config_path = f"config/model_configs/{model}.yaml"
            config = load_config(config_path)
            description = config.get('description', 'No description available')
        except:
            description = 'No description available'
        
        click.echo(f"{i}. {model} - {description}")

@cli.command(name="validate-data")
@click.option("--data-path", "-d", required=True, help="Path to input CSV file or directory")
@click.option("--config", "-c", default="config/data_validator.yaml", help="Data validation config")
@click.option("--report", "-r", default=None, help="Path to save validation report")
@click.option("--fix/--no-fix", default=False, help="Attempt to fix data issues")
def validate_data(data_path, config, report, fix):
    """Validate time series data for quality and consistency."""
    try:
        from data_pipeline.data_validator import DataValidator
        
        # Load configuration
        validation_config = load_config(config)
        
        # Initialize validator
        validator = DataValidator(validation_config)
        
        # Process file or directory
        if os.path.isdir(data_path):
            # Find all CSV files
            csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                        if f.endswith('.csv')]
            
            if not csv_files:
                click.echo(f"No CSV files found in {data_path}", err=True)
                return
            
            # Validate each file
            results = {}
            for csv_file in csv_files:
                click.echo(f"Validating {os.path.basename(csv_file)}...")
                file_result = validator.validate_file(csv_file, fix=fix)
                results[os.path.basename(csv_file)] = file_result
                
                # Print summary of issues
                issues_count = sum(len(issues) for issues in file_result['issues'].values())
                if issues_count > 0:
                    click.echo(f"  Found {issues_count} issues")
                else:
                    click.echo("  No issues found")
                    
            # Generate summary report
            if report:
                report_path = report
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"outputs/validation_reports/validation_{timestamp}.json"
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'config': validation_config,
                    'results': results
                }, f, indent=2)
                
            click.echo(f"Validation report saved to {report_path}")
            
        else:
            # Validate single file
            click.echo(f"Validating {os.path.basename(data_path)}...")
            result = validator.validate_file(data_path, fix=fix)
            
            # Print issues
            issues_count = sum(len(issues) for issues in result['issues'].values())
            if issues_count > 0:
                click.echo(f"Found {issues_count} issues:")
                for check_name, issues in result['issues'].items():
                    if issues:
                        click.echo(f"  {check_name}: {len(issues)} issues")
                        for issue in issues[:5]:  # Show first 5 issues only
                            click.echo(f"    - {issue}")
                        
                        if len(issues) > 5:
                            click.echo(f"    ... and {len(issues) - 5} more")
            else:
                click.echo("No issues found")
            
            # Generate report
            if report:
                report_path = report
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.splitext(os.path.basename(data_path))[0]
                report_path = f"outputs/validation_reports/validation_{filename}_{timestamp}.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'file': data_path,
                    'config': validation_config,
                    'result': result
                }, f, indent=2)
                
            click.echo(f"Validation report saved to {report_path}")
            
            # If fixes were applied
            if fix and result['fixes_applied']:
                click.echo(f"\nApplied {len(result['fixes_applied'])} fixes:")
                for fix_info in result['fixes_applied'][:5]:  # Show first 5 fixes
                    click.echo(f"  - {fix_info}")
                    
                if len(result['fixes_applied']) > 5:
                    click.echo(f"  ... and {len(result['fixes_applied']) - 5} more")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Data validation error: {str(e)}", exc_info=True)

@cli.command(name="batch-forecast")
@click.option("--model", "-m", required=True, help="Model to use for forecasting")
@click.option("--data-dir", "-d", required=True, help="Directory with multiple CSV time series")
@click.option("--config", "-c", default=None, help="Model configuration file")
@click.option("--output", "-o", default=None, help="Output directory for forecast results")
@click.option("--horizon", "-h", default=10, type=int, help="Forecast horizon (steps ahead)")
@click.option("--limit", "-l", default=None, type=int, help="Max number of files to process")
@click.option("--parallel/--no-parallel", default=False, help="Use parallel processing")
@click.option("--threads", "-t", default=4, type=int, help="Number of threads for parallel processing")
def batch_forecast(model, data_dir, config, output, horizon, limit, parallel, threads):
    """Generate forecasts for multiple time series files in batch."""
    try:
        # Validate model choice
        available_models = get_available_models()
        if model not in available_models:
            click.echo(f"Error: Model '{model}' not available. Choose from: {', '.join(available_models)}", err=True)
            return
        
        # Load model config file if not provided
        if config is None:
            config = f"config/model_configs/{model}.yaml"
        
        # Load configuration
        model_config = load_config(config)
        
        # Update configuration with CLI parameters
        model_config['forecast_horizon'] = horizon
        
        if output:
            model_config['output_dir'] = output
        else:
            model_config['output_dir'] = f"outputs/batch_forecasts/{model}"
        
        # Create output directory
        os.makedirs(model_config['output_dir'], exist_ok=True)
        
        # Find all CSV files
        csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                    if f.endswith('.csv')]
        
        if not csv_files:
            click.echo(f"No CSV files found in {data_dir}", err=True)
            return
        
        # Apply limit if specified
        if limit and limit < len(csv_files):
            click.echo(f"Limiting to {limit} of {len(csv_files)} files")
            csv_files = csv_files[:limit]
        
        click.echo(f"Found {len(csv_files)} CSV files to process")
        
        # Process files
        if parallel and len(csv_files) > 1:
            try:
                import concurrent.futures
                
                click.echo(f"Using parallel processing with {threads} threads")
                
                # Define processing function
                def process_file(file_path):
                    try:
                        # Get file-specific config
                        file_config = model_config.copy()
                        
                        # Create file-specific output directory
                        file_basename = os.path.splitext(os.path.basename(file_path))[0]
                        file_output_dir = os.path.join(model_config['output_dir'], file_basename)
                        file_config['output_dir'] = file_output_dir
                        file_config['results_dir'] = os.path.join(file_output_dir, 'results')
                        
                        # Create output directories
                        os.makedirs(file_config['output_dir'], exist_ok=True)
                        os.makedirs(file_config['results_dir'], exist_ok=True)
                        
                        # Get model adapter
                        file_adapter = get_model_adapter(model, file_config)
                        
                        # Run forecast
                        results = file_adapter.run_pipeline(file_path)
                        
                        # Save forecast
                        forecast_path = os.path.join(file_config['results_dir'], f'forecast.csv')
                        results.to_csv(forecast_path)
                        
                        return {
                            'file': file_path,
                            'output': forecast_path,
                            'rmse': float(results.attrs.get('rmse', 0)),
                            'success': True,
                            'error': None
                        }
                    except Exception as e:
                        return {
                            'file': file_path,
                            'success': False,
                            'error': str(e)
                        }
                
                # Process files in parallel
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    futures = {executor.submit(process_file, file_path): file_path for file_path in csv_files}
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        file_path = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            # Print progress
                            if result['success']:
                                click.echo(f"[{i}/{len(csv_files)}] Successfully processed: {os.path.basename(file_path)}")
                            else:
                                click.echo(f"[{i}/{len(csv_files)}] Failed to process: {os.path.basename(file_path)} - {result['error']}")
                                
                        except Exception as e:
                            click.echo(f"[{i}/{len(csv_files)}] Error processing {os.path.basename(file_path)}: {str(e)}", err=True)
                            results.append({
                                'file': file_path,
                                'success': False,
                                'error': str(e)
                            })
                
            except ImportError:
                click.echo("Parallel processing not available, falling back to sequential processing", err=True)
                parallel = False
        
        # Sequential processing
        if not parallel:
            results = []
            for i, file_path in enumerate(csv_files, 1):
                try:
                    click.echo(f"[{i}/{len(csv_files)}] Processing: {os.path.basename(file_path)}")
                    
                    # Get file-specific config
                    file_config = model_config.copy()
                    
                    # Create file-specific output directory
                    file_basename = os.path.splitext(os.path.basename(file_path))[0]
                    file_output_dir = os.path.join(model_config['output_dir'], file_basename)
                    file_config['output_dir'] = file_output_dir
                    file_config['results_dir'] = os.path.join(file_output_dir, 'results')
                    
                    # Create output directories
                    os.makedirs(file_config['output_dir'], exist_ok=True)
                    os.makedirs(file_config['results_dir'], exist_ok=True)
                    
                    # Get model adapter
                    file_adapter = get_model_adapter(model, file_config)
                    
                    # Run forecast
                    model_results = file_adapter.run_pipeline(file_path)
                    
                    # Save forecast
                    forecast_path = os.path.join(file_config['results_dir'], f'forecast.csv')
                    model_results.to_csv(forecast_path)
                    
                    results.append({
                        'file': file_path,
                        'output': forecast_path,
                        'rmse': float(model_results.attrs.get('rmse', 0)),
                        'success': True,
                        'error': None
                    })
                    
                    click.echo(f"  Successfully processed: {os.path.basename(file_path)}")
                    
                except Exception as e:
                    click.echo(f"  Failed to process: {os.path.basename(file_path)} - {str(e)}", err=True)
                    results.append({
                        'file': file_path,
                        'success': False,
                        'error': str(e)
                    })
        
        # Generate summary report
        summary_path = os.path.join(model_config['output_dir'], 'batch_summary.json')
        
        summary = {
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'config_file': config,
            'total_files': len(csv_files),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', False)),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        click.echo("\nBatch Forecast Summary:")
        click.echo(f"Total files processed: {len(csv_files)}")
        click.echo(f"Successful: {summary['successful']}")
        click.echo(f"Failed: {summary['failed']}")
        click.echo(f"Summary report saved to: {summary_path}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        logger.error(f"Batch forecast error: {str(e)}", exc_info=True)

if __name__ == '__main__':
    # Set up exception handling for the CLI
    try:
        cli()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        click.echo(f"Critical error: {str(e)}", err=True)
        sys.exit(1)
