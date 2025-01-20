# # # # from flask import Flask, request, jsonify, render_template
# # # # import joblib
# # # # import pandas as pd

# # # # app = Flask(__name__)

# # # # # Load the models
# # # # xgb_model = joblib.load('xgb_model_final.pkl')
# # # # lr_model = joblib.load('lr_model_final.pkl')

# # # # # Function to predict the winning probability
# # # # def predict_winning_probability(model, team_A, team_B, team_batting_first, runs_scored, wickets_fallen, overs_completed, runs_still_required):
# # # #     team_1_encoded = 0 if team_batting_first == team_A else 1
# # # #     team_2_encoded = 1 if team_batting_first == team_A else 0
    
# # # #     overs_left = 20 - overs_completed
# # # #     required_run_rate = runs_still_required / overs_left if overs_left > 0 else 0
# # # #     wickets_to_run_ratio = wickets_fallen / runs_still_required if runs_still_required > 0 else 0
    
# # # #     input_data = pd.DataFrame({
# # # #         'team_1_encoded': [team_1_encoded],
# # # #         'team_2_encoded': [team_2_encoded],
# # # #         'team_2_score': [runs_scored],
# # # #         'team_2_wickets_till_over_20': [wickets_fallen],
# # # #         'runs_still_required_over_20': [runs_still_required],
# # # #         'required_run_rate_over_20': [required_run_rate],
# # # #         'wickets_to_run_ratio_over_1': [wickets_to_run_ratio]
# # # #     })
# # # #     probability = model.predict_proba(input_data)
# # # #     return probability[0]

# # # # @app.route('/')
# # # # def home():
# # # #     return render_template('index.html')

# # # # @app.route('/estimate', methods=['POST'])
# # # # def estimate():
# # # #     try:
# # # #         data = request.json
# # # #         team_A = data['team_A']
# # # #         team_B = data['team_B']
# # # #         team_batting_first = data['team_batting_first']
# # # #         runs_scored = data['runs_scored']
# # # #         wickets_fallen = data['wickets_fallen']
# # # #         overs_completed = data['overs_completed']
# # # #         runs_still_required = data['runs_still_required']
        
# # # #         probability_xgb = predict_winning_probability(xgb_model, team_A, team_B, team_batting_first, runs_scored, wickets_fallen, overs_completed, runs_still_required)
# # # #         probability_lr = predict_winning_probability(lr_model, team_A, team_B, team_batting_first, runs_scored, wickets_fallen, overs_completed, runs_still_required)
# # # #         print("shruti")
# # # #         print(probability_xgb)
# # # #         print("shruti2")
# # # #         return jsonify({
# # # #             'xgb': {
# # # #                 'team_A': probability_xgb[1] * 100,
# # # #                 'team_B': probability_xgb[0] * 100
# # # #             },
# # # #             'lr': {
# # # #                 'team_A': probability_lr[1] * 100,
# # # #                 'team_B': probability_lr[0] * 100
# # # #             }
# # # #         })
# # # #     except Exception as e:
# # # #         print(f"Error: {e}")
# # # #         return jsonify({'error': 'An error occurred during processing'}), 500

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)


# # # import pandas as pd
# # # import joblib
# # # from flask import Flask, request, jsonify

# # # # Load model
# # # xgb_model = joblib.load('xgb_model1.pkl')

# # # # Load mappings
# # # team_1_mapping_df = pd.read_csv('team_1_mapping.csv')
# # # team_2_mapping_df = pd.read_csv('team_2_mapping.csv')
# # # winner_mapping_df = pd.read_csv('winner_mapping.csv')

# # # # Convert mappings to dictionaries
# # # team_1_mapping = dict(zip(team_1_mapping_df['Team_Name'], team_1_mapping_df['Encoded']))
# # # team_2_mapping = dict(zip(team_2_mapping_df['Team_Name'], team_2_mapping_df['Encoded']))

# # # # Reverse mappings for decoding
# # # reverse_team_1_mapping = {v: k for k, v in team_1_mapping.items()}
# # # reverse_team_2_mapping = {v: k for k, v in team_2_mapping.items()}

# # # app = Flask(__name__)

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     data = request.get_json()

# # #     # Extract input data
# # #     team_A = data['team_A']
# # #     team_B = data['team_B']
# # #     team_batting_first = data['team_batting_first']
# # #     runs_scored = data['runs_scored']
# # #     wickets_fallen = data['wickets_fallen']
# # #     overs_completed = data['overs_completed']
# # #     runs_still_required = data['runs_still_required']

# # #     # Determine team_1 and team_2
# # #     team_1 = team_batting_first
# # #     team_2 = team_A if team_batting_first == team_B else team_B

# # #     # Encode team names
# # #     team_1_encoded = team_1_mapping[team_1]
# # #     team_2_encoded = team_2_mapping[team_2]

# # #     # Calculate derived features
# # #     overs_left = 20 - overs_completed
# # #     required_run_rate = runs_still_required / overs_left if overs_left > 0 else 0
# # #     wickets_to_run_ratio = wickets_fallen / runs_still_required if runs_still_required > 0 else 0

# # #     # Prepare input for model
# # #     input_data = pd.DataFrame({
# # #         'team_1_encoded': [team_1_encoded],
# # #         'team_2_encoded': [team_2_encoded],
# # #         'team_2_score': [runs_scored],
# # #         'team_2_wickets_till_over_20': [wickets_fallen],
# # #         'runs_still_required_over_20': [runs_still_required],
# # #         'required_run_rate_over_20': [required_run_rate],
# # #         'wickets_to_run_ratio_over_1': [wickets_to_run_ratio]
# # #     })

# # #     # Predict probabilities
# # #     probabilities = xgb_model.predict_proba(input_data)[0]
# # #     response = {
# # #         'team_1': team_1,
# # #         'team_2': team_2,
# # #         'team_1_prob': probabilities[team_1_encoded],
# # #         'team_2_prob': probabilities[team_2_encoded]
# # #     }

# # #     return jsonify(response)

# # # if __name__ == '__main__':
# # #     app.run(debug=True)



# # #ayya niche sacho che


# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify, render_template

# # Load model
# xgb_model = joblib.load('logreg_model1.pkl')

# # Load mappings
# team_1_mapping_df = pd.read_csv('team_1_mapping_lg.csv')
# team_2_mapping_df = pd.read_csv('team_2_mapping_lg.csv')
# winner_mapping_df = pd.read_csv('winner_mapping_lg.csv')

# # Convert mappings to dictionaries
# team_1_mapping = dict(zip(team_1_mapping_df['Team_Name'], team_1_mapping_df['Encoded']))
# team_2_mapping = dict(zip(team_2_mapping_df['Team_Name'], team_2_mapping_df['Encoded']))

# # Reverse mappings for decoding
# reverse_team_1_mapping = {v: k for k, v in team_1_mapping.items()}
# reverse_team_2_mapping = {v: k for k, v in team_2_mapping.items()}

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/estimate', methods=['POST'])
# def estimate():
#     data = request.get_json()

#     # Extract input data
#     team_A = data['team_A']
#     team_B = data['team_B']
#     team_batting_first = data['team_batting_first']
#     runs_scored = data['runs_scored']
#     wickets_fallen = data['wickets_fallen']
#     overs_completed = data['overs_completed']
#     runs_still_required = data['runs_still_required']

#     # Determine team_1 and team_2
#     team_1 = team_batting_first
#     team_2 = team_A if team_batting_first == team_B else team_B

#     # Encode team names
#     team_1_encoded = team_1_mapping.get(team_1)
#     team_2_encoded = team_2_mapping.get(team_2)

#     if team_1_encoded is None or team_2_encoded is None:
#         return jsonify({"error": "Invalid team names"}), 400

#     # Calculate derived features
#     overs_left = 20 - overs_completed
#     required_run_rate = runs_still_required / overs_left if overs_left > 0 else 0
#     wickets_to_run_ratio = wickets_fallen / runs_still_required if runs_still_required > 0 else 0

#     # Prepare input for model
#     input_data = pd.DataFrame({
#         'team_1_encoded': [team_1_encoded],
#         'team_2_encoded': [team_2_encoded],
#         'team_2_score': [runs_scored],
#         'team_2_wickets_till_over_20': [wickets_fallen],
#         'runs_still_required_over_20': [runs_still_required],
#         'required_run_rate_over_20': [required_run_rate],
#         'wickets_to_run_ratio_over_1': [wickets_to_run_ratio]
#     })

#     # Predict probabilities
#     probabilities = xgb_model.predict_proba(input_data)[0]

#     response = {
#         "team_A": team_A,
#         "team_B": team_B,
#         "team_A_prob": 100-(probabilities[team_1_encoded] * 100),
#         "team_B_prob": (probabilities[team_1_encoded] * 100)
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)




#niche pipe vadu


from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('pipe.pkl')

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
        wickets_fallen = 10- data['wickets_fallen']
        overs_completed = data['overs_completed']
        runs_still_required = data['runs_still_required']

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
            'city': ['Mumbai'],
            'runs_left': [runs_still_required],
            'balls_left': [balls_left_t],
            'wickets': [wickets_fallen],
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
    app.run(debug=True, port=8050, host='0.0.0.0')
