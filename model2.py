# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# import xgboost as xgb
# import joblib
# import json

# # Load the data
# df = pd.read_csv('matches9.csv')

# # Ensure team names and winner are categorical
# df['team_1'] = df['team_1'].astype('category')
# df['team_2'] = df['team_2'].astype('category')
# df['winner'] = df['winner'].astype('category')

# # Encode the winner column
# df['winner_encoded'] = df['winner'].cat.codes

# # Encode team names
# df['team_1_encoded'] = df['team_1'].cat.codes
# df['team_2_encoded'] = df['team_2'].cat.codes

# # Save mappings for later use
# team_1_mapping = dict(enumerate(df['team_1'].cat.categories))
# team_2_mapping = dict(enumerate(df['team_2'].cat.categories))
# winner_mapping = dict(enumerate(df['winner'].cat.categories))

# # Save mappings to JSON
# with open('team_encodings.json', 'w') as f:
#     json.dump({
#         'team_1_mapping': team_1_mapping,
#         'team_2_mapping': team_2_mapping,
#         'winner_mapping': winner_mapping
#     }, f, indent=4)

# print("Team mappings saved to team_encodings.json")

# # Define features and labels
# features = df[['team_1_encoded', 'team_2_encoded', 'team_2_score', 'team_2_wickets_till_over_20', 
#                'runs_still_required_over_20', 'required_run_rate_over_20', 'wickets_to_run_ratio_over_1']]
# labels = df['winner_encoded']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Train the XGBoost model
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# xgb_model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = xgb_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Save the trained model
# joblib.dump(xgb_model, 'xgb_model1.pkl')
# print("Model saved as xgb_model1.pkl")


# # Save mappings as CSV files
# team_1_mapping_df = pd.DataFrame(list(team_1_mapping.items()), columns=['Encoded', 'Team_Name'])
# team_1_mapping_df.to_csv('team_1_mapping.csv', index=False)

# team_2_mapping_df = pd.DataFrame(list(team_2_mapping.items()), columns=['Encoded', 'Team_Name'])
# team_2_mapping_df.to_csv('team_2_mapping.csv', index=False)

# winner_mapping_df = pd.DataFrame(list(winner_mapping.items()), columns=['Encoded', 'Winner'])
# winner_mapping_df.to_csv('winner_mapping.csv', index=False)

# print("Mappings saved as CSV files: team_1_mapping.csv, team_2_mapping.csv, winner_mapping.csv")




import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import json

# Load the data
df = pd.read_csv('matches9.csv')

# Ensure team names and winner are categorical
df['team_1'] = df['team_1'].astype('category')
df['team_2'] = df['team_2'].astype('category')
df['winner'] = df['winner'].astype('category')

# Encode the winner column
df['winner_encoded'] = df['winner'].cat.codes

# Encode team names
df['team_1_encoded'] = df['team_1'].cat.codes
df['team_2_encoded'] = df['team_2'].cat.codes

# Save mappings for later use
team_1_mapping = dict(enumerate(df['team_1'].cat.categories))
team_2_mapping = dict(enumerate(df['team_2'].cat.categories))
winner_mapping = dict(enumerate(df['winner'].cat.categories))

# Save mappings to JSON
with open('team_encodings.json', 'w') as f:
    json.dump({
        'team_1_mapping': team_1_mapping,
        'team_2_mapping': team_2_mapping,
        'winner_mapping': winner_mapping
    }, f, indent=4)

print("Team mappings saved to team_encodings.json")

# Define features and labels
features = df[['team_1_encoded', 'team_2_encoded', 'team_2_score', 'team_2_wickets_till_over_20', 
               'runs_still_required_over_20', 'required_run_rate_over_20', 'wickets_to_run_ratio_over_1']]
labels = df['winner_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga','lbfgs']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Best hyperparameters found by GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(best_model, 'logreg_model1.pkl')
print("Model saved as logreg_model1.pkl")

# Save mappings as CSV files
team_1_mapping_df = pd.DataFrame(list(team_1_mapping.items()), columns=['Encoded', 'Team_Name'])
team_1_mapping_df.to_csv('team_1_mapping_lg.csv', index=False)

team_2_mapping_df = pd.DataFrame(list(team_2_mapping.items()), columns=['Encoded', 'Team_Name'])
team_2_mapping_df.to_csv('team_2_mapping_lg.csv', index=False)

winner_mapping_df = pd.DataFrame(list(winner_mapping.items()), columns=['Encoded', 'Winner'])
winner_mapping_df.to_csv('winner_mapping_lg.csv', index=False)

print("Mappings saved as CSV files: team_1_mapping_lg.csv, team_2_mapping_lg.csv, winner_mapping_lg.csv")
