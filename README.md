# IPL Score Prediction Model

A professional machine learning system for predicting IPL cricket match scores. Built with pure NumPy (no sklearn), Flask backend, and a modern interactive frontend.

## Features

### Machine Learning Models
- **Linear Regression** - Gradient descent implementation from scratch
- **Ridge Regression** - L2 regularization for better generalization
- **Lasso Regression** - L1 regularization with feature selection
- **Decision Tree Regressor** - Custom implementation with variance reduction

### Performance Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### Analysis Tools
- Side-by-side model comparison
- Residual analysis
- Confidence range prediction
- Accuracy by score ranges

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python src/train.py
```

### 3. Start the Server
```bash
python run.py
```

### 4. Open in Browser
```
http://localhost:5000
```

## Project Structure

```
ipl-score/
├── config/                      # Configuration files
│   ├── __init__.py
│   └── settings.py
│
├── src/                        # Source code package
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading & feature engineering
│   ├── models.py               # ML models implementation
│   ├── evaluation.py           # Performance metrics
│   ├── train.py                # Training script
│   └── app.py                  # Flask API server
│
├── templates/                   # HTML templates
│   └── index.html              # Main frontend interface
│
├── static/                      # Static assets (CSS, JS)
│
├── models/                      # Trained model files (generated)
│   ├── trained_models.pkl      # Serialized models
│   ├── preprocessing_params.pkl # Encoders & normalizers
│   └── evaluation_results.json  # Model metrics
│
├── data/                        # Data directory
│   └── IPL.csv                 # Training dataset
│
├── reports/                     # Generated reports and analysis
├── logs/                        # Application logs
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
├── tests/                       # Unit tests
│
├── run.py                       # Application entry point
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # Documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict match score |
| `/api/models` | GET | List available models |
| `/api/comparison` | GET | Model performance comparison |
| `/api/teams` | GET | List of teams |
| `/api/cities` | GET | List of cities |
| `/api/health` | GET | Health check |

## Prediction Input

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `batting_team` | string | Batting team name | "Mumbai Indians" |
| `bowling_team` | string | Bowling team name | "Chennai Super Kings" |
| `city` | string | Match city | "Mumbai" |
| `runs` | integer | Current score | 85 |
| `overs` | float | Overs bowled (0-20) | 10.0 |
| `wickets` | integer | Wickets lost (0-10) | 3 |

## API Usage Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "city": "Mumbai",
    "runs": 85,
    "overs": 10.0,
    "wickets": 3
  }'
```

```python
# Python example
from src.train import predict_score

predicted = predict_score(
    batting_team='Mumbai Indians',
    bowling_team='Chennai Super Kings',
    city='Mumbai',
    runs=85,
    overs=10.0,
    wickets=3,
    model_name='Decision Tree'
)

print(f"Predicted Final Score: {predicted:.0f}")
```

## Model Performance

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| **Decision Tree** | 4.55 | 2.13 | **0.9949** |
| Linear Regression | 103.32 | 10.16 | 0.8837 |
| Ridge Regression | 103.44 | 10.17 | 0.8836 |
| Lasso Regression | 103.86 | 10.19 | 0.8831 |

**Best Model:** Decision Tree with R² = 0.9949 (RMSE = 2.13 runs)

## Model Implementation Details

All ML models are implemented **from scratch using only NumPy**:

### Linear Regression
- Batch gradient descent optimization
- Configurable learning rate and iterations
- Automatic intercept handling

### Ridge Regression
- L2 penalty term for regularization
- Reduces overfitting
- Better for correlated features

### Lasso Regression
- L1 penalty with soft thresholding
- Automatic feature selection
- Sparse coefficient solutions

### Decision Tree Regressor
- Variance reduction split criterion
- Configurable depth and leaf constraints
- Recursive tree building

## Dataset

The model is trained on historical IPL ball-by-ball data:
- Match metadata (teams, venue, toss, date)
- Ball-by-ball events (runs, wickets, extras)
- Player information (batters, bowlers, fielders)
- Match outcomes and results

## Technology Stack

- **Backend:** Python, NumPy, Flask, Flask-CORS
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **ML:** Custom implementations (no sklearn)

## License

MIT License

## Author

IPL Prediction Team

---

Built with ❤️ using NumPy and Flask
