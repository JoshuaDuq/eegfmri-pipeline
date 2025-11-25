import sys
import logging
from pathlib import Path
from typing import Tuple

from ..config.loader import load_settings, ConfigDict
from ..io.general import get_logger


###################################################################
# Path Setup
###################################################################

def setup_script_path(script_file: str) -> Path:
    """
    Setup project root path and add to sys.path.
    
    Args:
        script_file: __file__ from the calling script
        
    Returns:
        Path to project root
    """
    project_root = Path(script_file).parent.parent.parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


###################################################################
# Environment Setup
###################################################################

def setup_script_environment(script_file: str, logger_name: str | None = None) -> Tuple[ConfigDict, logging.Logger]:
    """
    Setup script environment: path, config, and logger.
    
    Args:
        script_file: __file__ from the calling script
        logger_name: Optional logger name (defaults to script module name)
        
    Returns:
        Tuple of (config, logger)
    """
    setup_script_path(script_file)
    config = load_settings()
    
    if logger_name is None:
        logger_name = Path(script_file).stem
    
    logger = get_logger(logger_name)
    
    return config, logger

