import torch
from dataframes import all_teams, team_data
from LogRegModel import model
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_team_name_by_id(team_id, df=all_teams):
    result = df[df['ID'] == team_id]
    if not result.empty:
        return result.iloc[0]['Name']
    else:
        return f"Team with ID:{team_id} is not found"


def get_team_id_by_name(team_name, df=all_teams):
    result = df[df['Name'] == team_name]
    if not result.empty:
        return result.iloc[0]['ID']
    else:
        return f"Team with name:'{team_name}' is not found"


def match_between(id_team_1, id_team_2):
    team1_data = team_data[team_data['match|summaryStats|team_combined|TeamId|$numberLong'] == id_team_1]
    team2_data = team_data[team_data['match|summaryStats|team_combined|TeamId|$numberLong'] == id_team_2]
    if team2_data.empty or team1_data.empty:
        raise Exception(f'One of the teams does not exist: {id_team_1}:{team1_data}/{id_team_2}:{team2_data}')

    team1_data = team1_data.iloc[0]
    team2_data = team2_data.iloc[0]

    delta = 1.01
    combined_data = (team2_data + delta) - (team1_data + delta)
    # scaler = StandardScaler()
    # combined_data_scaled = scaler.fit_transform(combined_data.values.reshape(1, -1))

    tournament_tensor = torch.tensor(combined_data.values, dtype=torch.float32)
    winner = (model(tournament_tensor) > 0.5).float().numpy()

    if winner[0] == 1.0:
        return {'winner': 'Team 1', 'team_name': get_team_name_by_id(id_team_1, all_teams)}
    else:
        return {'winner': 'Team 2', 'team_name': get_team_name_by_id(id_team_2, all_teams)}


def is_power_of_two(n):
    if n <= 1:
        return 0
    if (n & (~(n - 1))) == n:
        return 1
    return 0

def tournament(teams):
    """
    Simulate a tournament with 8 teams.
    Returns winners of each match and the final winner.
    """
    if not is_power_of_two(len(teams)):
        raise ValueError("Number of teams should be power of 2.")
    if len(teams) > 32:
        raise ValueError("Currently API supports up to 32 teams.")

    results = {}
    tournament_round = 0
    while len(teams) > 1:
        temp_winners = []
        tournament_round += 1
        round_name = f'round_{tournament_round}'
        results[round_name] = []
        for i in range(0, len(teams), 2):
            match_result = match_between(teams[i], teams[i + 1])
            winner_id = teams[i] if match_result['winner'] == 'Team 1' else teams[i + 1]
            results[round_name].append({
                'team1': get_team_name_by_id(teams[i]),
                'team1_id': teams[i],
                'team2': get_team_name_by_id(teams[i + 1]),
                'team2_id': teams[i + 1],
                'winner': match_result['team_name'],
                'winner_id': winner_id
            })
            temp_winners.append(winner_id)
        teams = temp_winners.copy()

    final_round_name = f'round_{tournament_round}'

    return {
        'matches': results,
        'tournament_winner': results[final_round_name][-1]['winner'],
        'tournament_winner_id': results[final_round_name][-1]['winner_id']
    }
