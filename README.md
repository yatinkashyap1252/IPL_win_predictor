# IPL Win Predictor

This project is a machine learning model to predict the outcome of Indian Premier League (IPL) cricket matches. Using historical match and delivery data, the model predicts whether the batting team will win or lose based on the current match situation.

## Dataset

The project uses two datasets:
- `matches.csv`: Contains match-level information such as teams, city, winner, and total runs.
- `deliveries.csv`: Contains ball-by-ball delivery data including runs scored, wickets, overs, and other match events.

## Features and Preprocessing

Key features used for prediction include:
- Batting team
- Bowling team
- City where the match is played
- Runs left to win
- Balls left in the innings
- Current run rate
- Required run rate
- Wickets remaining
- Target score
- Pressure index (calculated based on required run rate and wickets remaining)
- Over progress (percentage of innings completed)
- Run rate difference (current run rate minus required run rate)

Categorical features are one-hot encoded, and numeric features are used as-is.

## Model

A Random Forest Classifier is trained on the processed data to predict the match outcome (win or lose for the batting team). Two versions of the model are saved:
- `ipl_win_predictor_v1.joblib`
- `ipl_win_predictor_v2.joblib` (improved version with additional features)

The model achieves good accuracy on the test set (around 80-85%).

## Usage

### Running the Notebook

The main notebook (`Untitled.ipynb`) contains the full data processing, model training, evaluation, and example predictions. You can run it in a Jupyter environment to reproduce the results.

### Using the Saved Model

You can load the saved model and make predictions by preparing input data in the following format:

```python
user_input = {
    'batting_team': 'Chennai Super Kings',
    'bowling_team': 'Punjab Kings',
    'city': 'Chandigarh',
    'target': 220,
    'current_score': 160,
    'overs_completed': 17.0,
    'wickets_fallen': 5
}
```

Use the provided `prepare_input` function to convert this input into the required feature format:

```python
import pandas as pd

def prepare_input(user_input):
    runs_left = user_input['target'] - user_input['current_score']
    balls_left = 120 - int(user_input['overs_completed'] * 6)
    current_run_rate = (user_input['current_score'] / user_input['overs_completed']) if user_input['overs_completed'] != 0 else 0
    required_run_rate = (runs_left * 6 / balls_left) if balls_left != 0 else 0
    wickets_remaining = 10 - user_input['wickets_fallen']
    pressure = round(required_run_rate / (wickets_remaining + 1), 2) if wickets_remaining > 0 else required_run_rate
    over_progress = round((120 - balls_left) / 120, 2)
    run_rate_diff = current_run_rate - required_run_rate

    test_input = pd.DataFrame([{
        'batting_team': user_input['batting_team'],
        'bowling_team': user_input['bowling_team'],
        'city': user_input['city'],
        'runs_left': runs_left,
        'balls_left': balls_left,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate,
        'wickets_remaining': wickets_remaining,
        'target': user_input['target'],
        'pressure': pressure,
        'over_progress': over_progress,
        'run_rate_diff': run_rate_diff
    }])
    return test_input
```

Load the model and predict:

```python
import joblib

model = joblib.load('ipl_win_predictor_v2.joblib')
test_input = prepare_input(user_input)
prediction = model.predict(test_input)[0]
probability = model.predict_proba(test_input)[0][1]

print("Predicted Result:", "Win" if prediction == 1 else "Lose")
print(f"Win Probability: {probability:.2%}")
```

## Installation

Make sure you have the following Python packages installed:

- numpy
- pandas
- scikit-learn
- joblib

You can install them using pip:

```bash
pip install numpy pandas scikit-learn joblib
```

## License

This project is provided as-is for educational purposes.

## Author

Created by IPL Win Predictor project contributor.
