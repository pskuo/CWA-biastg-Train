import os
import time
import logging
from datetime import datetime
import traceback
import json
from .parameter_handler import ParameterHandler

class PipelineOrchestrator:
    """
    Orchestrates the execution of time series forecasting pipelines.
    
    This class manages the workflow of different pipeline stages, handles errors,
    tracks performance, and coordinates the overall execution flow.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config (dict, optional): Configuration dictionary
            config_path (str, optional): Path to configuration file
        """
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameter handler
        self.params = ParameterHandler(config, config_path)
        
        # Track pipeline execution
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'stages': {},
            'errors': [],
            'current_stage': None
        }
        
        # Pipeline components
        self.components = {}
        
        self.logger.info("Pipeline orchestrator initialized")
    
    def register_component(self, name, component):
        """
        Register a component for use in the pipeline.
        
        Args:
            name (str): Component name
            component: Component instance
            
        Returns:
            bool: True if successful
        """
        try:
            self.components[name] = component
            self.logger.debug(f"Registered component: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register component {name}: {str(e)}")
            return False
    
    def get_component(self, name):
        """
        Get a registered component.
        
        Args:
            name (str): Component name
            
        Returns:
            Component instance or None
        """
        if name in self.components:
            return self.components[name]
        
        self.logger.warning(f"Component not found: {name}")
        return None
    
    def _start_stage(self, stage_name, stage_params=None):
        """
        Start tracking a pipeline stage.
        
        Args:
            stage_name (str): Name of the stage
            stage_params (dict, optional): Parameters for this stage
        """
        self.execution_stats['current_stage'] = stage_name
        
        stage_stats = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None,
            'success': None,
            'params': stage_params or {}
        }
        
        self.execution_stats['stages'][stage_name] = stage_stats
        self.logger.info(f"Starting pipeline stage: {stage_name}")
    
    def _end_stage(self, stage_name, success=True, output=None):
        """
        End tracking a pipeline stage.
        
        Args:
            stage_name (str): Name of the stage
            success (bool): Whether the stage succeeded
            output (dict, optional): Output data from the stage
        """
        if stage_name not in self.execution_stats['stages']:
            self.logger.warning(f"Ending untracked stage: {stage_name}")
            return
            
        stage = self.execution_stats['stages'][stage_name]
        end_time = datetime.now()
        
        if 'start_time' in stage:
            # Convert string back to datetime if needed
            start_time = stage['start_time']
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            
            duration = (end_time - start_time).total_seconds()
        else:
            duration = 0
        
        stage.update({
            'end_time': end_time.isoformat(),
            'duration': duration,
            'success': success
        })
        
        if output:
            stage['output'] = output
        
        self.execution_stats['current_stage'] = None
        
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Pipeline stage {stage_name} {status} in {duration:.2f}s")
    
    def execute_stage(self, stage_name, callable_fn, params=None, error_handler=None):
        """
        Execute a pipeline stage with tracking.
        
        Args:
            stage_name (str): Name of the stage
            callable_fn: Function to call for this stage
            params (dict, optional): Parameters to pass to the function
            error_handler: Optional function to handle errors
            
        Returns:
            tuple: (success, result)
        """
        self._start_stage(stage_name, params)
        
        try:
            # Execute the stage function
            if params:
                result = callable_fn(**params)
            else:
                result = callable_fn()
                
            self._end_stage(stage_name, success=True, output={'result': str(type(result))})
            return True, result
            
        except Exception as e:
            error_info = {
                'stage': stage_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.execution_stats['errors'].append(error_info)
            self._end_stage(stage_name, success=False, output=error_info)
            
            self.logger.error(f"Error in stage {stage_name}: {str(e)}")
            self.logger.debug(f"Detailed error: {traceback.format_exc()}")
            
            # Call error handler if provided
            if error_handler:
                try:
                    error_handler(stage_name, e, error_info)
                except Exception as handler_error:
                    self.logger.error(f"Error in error handler: {str(handler_error)}")
            
            return False, None
    
    def execute_pipeline(self, stages, stop_on_error=True):
        """
        Execute a series of pipeline stages in sequence.
        
        Args:
            stages (list): List of stage configurations
            stop_on_error (bool): Whether to stop on first error
            
        Returns:
            tuple: (success, results)
        """
        # Start timing the pipeline
        self.execution_stats['start_time'] = datetime.now().isoformat()
        
        pipeline_results = {}
        pipeline_success = True
        
        self.logger.info(f"Beginning pipeline execution with {len(stages)} stages")
        
        # Execute each stage
        for stage in stages:
            stage_name = stage['name']
            stage_fn = stage['function']
            stage_params = stage.get('params', {})
            
            success, result = self.execute_stage(
                stage_name, stage_fn, stage_params, stage.get('error_handler')
            )
            
            pipeline_results[stage_name] = result
            
            if not success:
                pipeline_success = False
                if stop_on_error:
                    self.logger.warning("Pipeline stopped due to error")
                    break
        
        # End timing the pipeline
        self.execution_stats['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.execution_stats['start_time'])
        end_time = datetime.fromisoformat(self.execution_stats['end_time'])
        self.execution_stats['duration'] = (end_time - start_time).total_seconds()
        
        # Log completion
        if pipeline_success:
            self.logger.info(f"Pipeline executed successfully in {self.execution_stats['duration']:.2f}s")
        else:
            self.logger.warning(f"Pipeline execution completed with errors in {self.execution_stats['duration']:.2f}s")
        
        return pipeline_success, pipeline_results
    
    def save_execution_report(self, output_path=None):
        """
        Save the execution statistics to a file.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report
        """
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.params.get('results_dir', 'outputs/results'),
                f"pipeline_report_{timestamp}.json"
            )
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create report
            report = {
                'execution_stats': self.execution_stats,
                'configuration': self.params.get_all(),
                'timestamp': datetime.now().isoformat(),
            }
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Execution report saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving execution report: {str(e)}")
            return None
