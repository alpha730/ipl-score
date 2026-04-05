"""
IPL Data Preprocessing Module
Uses core parameters: runs, overs, wickets
Predicts FINAL score from MID-INNINGS state
"""

import numpy as np
import csv


def load_data(filepath):
    """Load IPL data from CSV file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def clean_data(data):
    """Clean and filter the data"""
    cleaned = []
    for row in data:
        if not row.get('batting_team') or not row.get('bowling_team'):
            continue

        try:
            row['over'] = int(float(row['over'])) if row.get('over') else 0
            row['team_runs'] = int(row['team_runs']) if row.get('team_runs') else 0
            row['team_balls'] = int(row['team_balls']) if row.get('team_balls') else 0
            row['team_wicket'] = int(row['team_wicket']) if row.get('team_wicket') else 0
            cleaned.append(row)
        except (ValueError, TypeError):
            continue

    return cleaned


def create_innings_summary(data):
    """
    Create innings-level summary using core parameters:
    - team_runs (runs scored)
    - team_balls (balls faced)
    - team_wicket (wickets lost)
    """
    innings_data = {}

    for row in data:
        match_id = row['match_id']
        innings = row.get('innings', '1')

        if innings != '1':
            continue

        key = f"{match_id}_{innings}"

        if key not in innings_data:
            innings_data[key] = {
                'match_id': match_id,
                'innings': innings,
                'batting_team': row['batting_team'],
                'bowling_team': row['bowling_team'],
                'venue': row.get('venue', ''),
                'city': row.get('city', ''),
                'season': row.get('season', ''),
                'runs': 0,
                'balls': 0,
                'wickets': 0,
                'overs': 0.0,
                'run_rate': 0.0
            }

        try:
            runs = int(row.get('team_runs', 0) or 0)
            balls = int(row.get('team_balls', 0) or 0)
            wickets = int(row.get('team_wicket', 0) or 0)

            if runs > innings_data[key]['runs']:
                innings_data[key]['runs'] = runs
            if balls > innings_data[key]['balls']:
                innings_data[key]['balls'] = balls
            if wickets > innings_data[key]['wickets']:
                innings_data[key]['wickets'] = wickets

        except (ValueError, TypeError):
            pass

    # Calculate derived parameters
    for key in innings_data:
        inn = innings_data[key]
        inn['overs'] = inn['balls'] / 6.0
        if inn['overs'] > 0:
            inn['run_rate'] = inn['runs'] / inn['overs']

    return list(innings_data.values())


def create_mid_innings_samples(innings_data):
    """
    Create multiple training samples from each innings at different stages.
    This allows the model to learn how to predict FINAL score from MID-INNINGS state.
    """
    samples = []

    for inn in innings_data:
        # Only use complete innings (20 overs = 120 balls)
        if inn['balls'] < 120:
            continue

        final_score = inn['runs']

        # Create samples at different over thresholds
        # Sample at 6 overs (powerplay)
        if inn['balls'] >= 36:  # 6 overs
            samples.append({
                'batting_team': inn['batting_team'],
                'bowling_team': inn['bowling_team'],
                'city': inn['city'],
                'runs': int(inn['runs'] * 36 / inn['balls']) if inn['balls'] > 0 else 0,
                'overs': 6.0,
                'wickets': max(0, int(inn['wickets'] * 36 / inn['balls'])),
                'final_score': final_score
            })

        # Sample at 10 overs
        if inn['balls'] >= 60:  # 10 overs
            samples.append({
                'batting_team': inn['batting_team'],
                'bowling_team': inn['bowling_team'],
                'city': inn['city'],
                'runs': int(inn['runs'] * 60 / inn['balls']) if inn['balls'] > 0 else 0,
                'overs': 10.0,
                'wickets': max(0, int(inn['wickets'] * 60 / inn['balls'])),
                'final_score': final_score
            })

        # Sample at 15 overs
        if inn['balls'] >= 90:  # 15 overs
            samples.append({
                'batting_team': inn['batting_team'],
                'bowling_team': inn['bowling_team'],
                'city': inn['city'],
                'runs': int(inn['runs'] * 90 / inn['balls']) if inn['balls'] > 0 else 0,
                'overs': 15.0,
                'wickets': max(0, int(inn['wickets'] * 90 / inn['balls'])),
                'final_score': final_score
            })

        # Sample at 18 overs
        if inn['balls'] >= 108:  # 18 overs
            samples.append({
                'batting_team': inn['batting_team'],
                'bowling_team': inn['bowling_team'],
                'city': inn['city'],
                'runs': int(inn['runs'] * 108 / inn['balls']) if inn['balls'] > 0 else 0,
                'overs': 18.0,
                'wickets': max(0, int(inn['wickets'] * 108 / inn['balls'])),
                'final_score': final_score
            })

        # Add final innings state as well
        samples.append({
            'batting_team': inn['batting_team'],
            'bowling_team': inn['bowling_team'],
            'city': inn['city'],
            'runs': inn['runs'],
            'overs': 20.0,
            'wickets': inn['wickets'],
            'final_score': final_score
        })

    return samples


def prepare_training_data(innings_data, encoders=None):
    """
    Prepare features using core parameters:
    - batting_team (categorical)
    - bowling_team (categorical)
    - runs (current score)
    - overs (overs bowled)
    - wickets (wickets lost)

    Target: final_score (score at end of 20 overs)
    """
    categorical_cols = ['batting_team', 'bowling_team', 'city']

    # Create mid-innings samples
    samples = create_mid_innings_samples(innings_data)

    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            unique_vals = set()
            for sample in samples:
                if col in sample:
                    unique_vals.add(sample[col])
            encoders[col] = {v: i for i, v in enumerate(sorted(unique_vals))}

    X = []
    y = []

    for sample in samples:
        features = []

        # Categorical: team encodings
        bat_team = sample.get('batting_team', '')
        bowl_team = sample.get('bowling_team', '')
        city = sample.get('city', '')

        features.append(encoders['batting_team'].get(bat_team, 0))
        features.append(encoders['bowling_team'].get(bowl_team, 0))
        features.append(encoders['city'].get(city, 0))

        # Core numeric parameters
        features.append(sample['runs'])      # Current runs scored
        features.append(sample['overs'])     # Overs bowled
        features.append(sample['wickets'])   # Wickets lost

        # Target: final score
        target = sample['final_score']
        if target > 0:
            X.append(features)
            y.append(target)

    return np.array(X), np.array(y), encoders


def normalize_features(X, mean=None, std=None, fit=True):
    """Normalize features using z-score normalization"""
    if fit:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1

    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def get_feature_names():
    """Return list of feature names"""
    return [
        'batting_team_encoded',
        'bowling_team_encoded',
        'city_encoded',
        'runs',           # Current score
        'overs',          # Overs bowled
        'wickets'         # Wickets lost
    ]


def get_target_name():
    """Return target variable name"""
    return 'final_score'


if __name__ == "__main__":
    print("Loading IPL data...")
    data = load_data("IPL.csv")
    print(f"Loaded {len(data)} rows")

    print("\nCleaning data...")
    cleaned = clean_data(data)
    print(f"Cleaned data: {len(cleaned)} rows")

    print("\nCreating innings summary...")
    innings = create_innings_summary(cleaned)
    print(f"Created {len(innings)} innings summaries")

    print("\nPreparing training data (mid-innings samples)...")
    X, y, encoders = prepare_training_data(innings)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nFeature names:", get_feature_names())
    print("Target name:", get_target_name())

    print("\n--- Data Statistics ---")
    print(f"Final Score range: {np.min(y):.0f} - {np.max(y):.0f}")
    if X.size > 0:
        print(f"Current Runs range: {np.min(X[:, 3]):.0f} - {np.max(X[:, 3]):.0f}")
        print(f"Overs range: {np.min(X[:, 4]):.1f} - {np.max(X[:, 4]):.1f}")
        print(f"Wickets range: {np.min(X[:, 5]):.0f} - {np.max(X[:, 5]):.0f}")
