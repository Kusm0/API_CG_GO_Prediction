import torch
from sklearn.preprocessing import StandardScaler
from dataframes import team_names, team_data
from LogRegModel import model
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)



def get_team_name_by_id(team_id,df = team_names):
    result = df[df['ID'] == team_id]
    if not result.empty:
        return result.iloc[0]['Name']
    else:
        return f"Team with ID:{team_id} is not found"

def get_team_id_by_name(team_name,df = team_names):
    result = df[df['Name'] == team_name]
    if not result.empty:
        return result.iloc[0]['ID']
    else:
        return f"Team with name:'{team_name}' is not found"


def match_between(id_team_1, id_team_2):


    team1_data = team_data[team_data['match|summaryStats|team_combined|TeamId|$numberLong'] == id_team_1]
    team2_data = team_data[team_data['match|summaryStats|team_combined|TeamId|$numberLong'] == id_team_2]

    if team2_data.empty or team1_data.empty:
        return {'error': 'One of the teams does not exist'}

    team1_data = team1_data.iloc[0]
    team2_data = team2_data.iloc[0]

    delta = 1.01
    combined_data = (team1_data + delta) / (team2_data + delta)
    combined_data = pd.DataFrame([combined_data])
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(combined_data)

    tournament_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    winner = (model(tournament_tensor) > 0.5).float().numpy()

    if winner[0] == 1.0:
        return {'winner': 'Team 1', 'team_name': get_team_name_by_id(id_team_1, team_names)}
    else:
        return {'winner': 'Team 2', 'team_name': get_team_name_by_id(id_team_2, team_names)}



def tournament(teams):
    """
    Simulate a tournament with 8 teams.
    Returns winners of each match and the final winner.
    """
    if len(teams) != 8:
        raise ValueError("Tournament requires exactly 8 teams.")

    results = []

    # Round 1
    round1_winners = []
    for i in range(0, len(teams), 2):
        match_result = match_between(teams[i], teams[i + 1])
        results.append({
            'round': 1,
            'team1': get_team_name_by_id(teams[i]),
            'team2': get_team_name_by_id(teams[i + 1]),
            'winner': match_result['team_name']
        })
        round1_winners.append(teams[i] if match_result['winner'] == 'Team 1' else teams[i + 1])

    # Round 2
    round2_winners = []
    for i in range(0, len(round1_winners), 2):
        match_result = match_between(round1_winners[i], round1_winners[i + 1])
        results.append({
            'round': 2,
            'team1': get_team_name_by_id(round1_winners[i]),
            'team2': get_team_name_by_id(round1_winners[i + 1]),
            'winner': match_result['team_name']
        })
        round2_winners.append(round1_winners[i] if match_result['winner'] == 'Team 1' else round1_winners[i + 1])

    # Final Round
    final_match = match_between(round2_winners[0], round2_winners[1])
    results.append({
        'round': 3,
        'team1': get_team_name_by_id(round2_winners[0]),
        'team2': get_team_name_by_id(round2_winners[1]),
        'winner': final_match['team_name']
    })

    return {
        'matches': results,
        'tournament_winner': final_match['team_name']
    }
