"""
WSGI config for production deployment
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.app import app, load_models

logger.info("Loading models...")
if load_models():
    logger.info("Models loaded successfully!")
else:
    logger.error("Failed to load models!")
