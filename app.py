
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model_file_final.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate():
    try:
        data = request.get_json()

        # Extract input data
        team_A = data['team_A']
        team_B = data['team_B']
        team_batting_first = data['team_batting_first']
        runs_scored = data['runs_scored']
        wickets_rem = 10- data['wickets_fallen']
        overs_completed = data['overs_completed']
        runs_still_required = data['runs_still_required']
        match_city = data['match_city']

        # Calculate additional features
        balls_left_t = (20 - overs_completed) * 6
        crr = runs_scored / overs_completed
        rrr = (runs_still_required * 6) / balls_left_t
        target = runs_scored + runs_still_required

        # Determine team_1 and team_2
        team_1 = team_batting_first
        team_2 = team_A if team_batting_first == team_B else team_B

        # Create DataFrame for prediction
        df = pd.DataFrame({
            'batting_team': [team_2],
            'bowling_team': [team_1],
            'city': [match_city],
            'runs_left': [runs_still_required],
            'balls_left': [balls_left_t],
            'wickets': [wickets_rem],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict winning probability
        probability = model.predict_proba(df)[0]
          
        team_A_prob =  round(probability[0] * 100)
        team_B_prob =  round(probability[1] * 100)

        if team_2 ==team_B:
             return jsonify({
            'team_A': team_A,
            'team_B': team_B,
            'team_A_prob': team_A_prob,
            'team_B_prob': team_B_prob
        })
        else:
            
            return jsonify({
            'team_A': team_A,
            'team_B': team_B,
            'team_A_prob': team_B_prob,
            'team_B_prob': team_A_prob
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during processing'}), 500

if __name__ == '__main__':
    app.run(debug=True,port=8080)
