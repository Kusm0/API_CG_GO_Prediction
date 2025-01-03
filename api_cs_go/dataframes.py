import pandas as pd

df = pd.read_csv('data/preprocesed_data.csv')
all_teams = pd.read_csv('data/teams.csv')
team_data = pd.read_csv('data/team_combined_id_mean_features.csv')
present_teams_id = team_data[['match|summaryStats|team_combined|TeamId|$numberLong']]
present_teams = pd.merge(present_teams_id,
                         all_teams,
                         left_on='match|summaryStats|team_combined|TeamId|$numberLong',
                         right_on='ID',
                         how='inner')
present_teams = present_teams.drop('match|summaryStats|team_combined|TeamId|$numberLong',
                                   axis=1)
