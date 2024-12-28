from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import random
from dataframes import df, team_names, team_data
from LogRegModel import model, evaluate_model
from split_function import split_func
from match import match_between, tournament
import pandas as pd

app = Flask(__name__)
Swagger(app)

# Глобальні змінні
X_train, X_test, y_train, y_test = split_func(df)
# Отримання значень колонки 'id' у вигляді списку
team_id_list = team_data['match|summaryStats|team_combined|TeamId|$numberLong'].tolist()
print(team_id_list)


@app.route('/match_between', methods=['POST'])
def match_between_endpoint():
    """
    Predict the winner between two teams
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            id_team_1:
              type: integer
            id_team_2:
              type: integer
    responses:
      200:
        description: Prediction completed
    """
    data = request.json
    id_team_1 = data.get('id_team_1')
    id_team_2 = data.get('id_team_2')

    if id_team_1 is None or id_team_2 is None:
        return jsonify({'error': 'Both team IDs must be provided'}), 400

    result = match_between(id_team_1, id_team_2)
    if 'error' in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """
    Get metrics of the trained model
    ---
    responses:
      200:
        description: Metrics retrieved successfully
    """
    if not model:
        return jsonify({'error': 'Model not trained'}), 400

    accuracy = evaluate_model(model, X_test, y_test)
    y_pred = (model(X_test) > 0.5).float().numpy()
    report = classification_report(y_test.numpy(), y_pred, output_dict=True)

    return jsonify({'accuracy': accuracy, 'classification_report': report})


@app.route('/simulate_tournament', methods=['POST'])
def simulate_tournament():
    """
    Simulate a tournament with 8 teams
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            teams:
              type: array
              items:
                type: integer
              minItems: 8
              maxItems: 8
    responses:
      200:
        description: Tournament simulated successfully
    """
    data = request.json
    teams = data.get('teams')

    if not teams or len(teams) != 8:
        return jsonify({'error': 'You must provide exactly 8 team IDs.'}), 400

    try:
        result = tournament(teams)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/simulate_random_tournament', methods=['GET'])
def simulate_random_tournament():
    """
    Simulate a random tournament with 8 teams selected from the predefined team_id_list
    ---
    responses:
      200:
        description: Random tournament simulated successfully
    """
    if len(team_id_list) < 8:
        return jsonify({'error': 'Not enough teams in the predefined list to simulate a tournament.'}), 400

    random_teams = random.sample(team_id_list, 8)

    try:
        result = tournament(random_teams)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
