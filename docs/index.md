# IPL Score Prediction - Documentation

## Overview

This document provides detailed information about the IPL Score Prediction system.

## Architecture

### Data Flow
1. Raw CSV data → Data Preprocessing → Feature Engineering
2. Preprocessed features → Model Training → Serialized models
3. User input → API → Model prediction → Response

### Components

#### Data Preprocessing (`src/data_preprocessing.py`)
- Loads IPL match data from CSV
- Cleans and normalizes data
- Creates innings summaries
- Prepares training/testing datasets

#### Models (`src/models.py`)
- Linear Regression: Basic gradient descent
- Ridge Regression: L2 regularization
- Lasso Regression: L1 regularization
- Decision Tree: Variance reduction splits

#### Evaluation (`src/evaluation.py`)
- MSE, RMSE, R² metrics
- Model comparison utilities

#### API Server (`src/app.py`)
- Flask REST API
- Prediction endpoints
- Model management

## Configuration

Edit `config/settings.py` to customize:
- Model paths
- Dataset configuration
- API settings

## Development

```bash
# Run tests
python -m pytest tests/

# Train models
python src/train.py

# Start server
python run.py
```

## API Reference

See README.md for complete API documentation.
