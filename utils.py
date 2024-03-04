import requests
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import gamma, norm
import numpy as np
import matplotlib.pyplot as plt


headers = {
	"X-RapidAPI-Key": "0ace358a11msh697dad8fc9e8988p16ff4ajsn1b63985a1b2f",
	"X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
}
current_season = 2023

stat_trans = {'Rebs+Asts': "RA", 
				'Rebounds': "R", 
				'Pts+Rebs': "PR",
       			'Pts+Asts': "PA", 
       			'Pts+Rebs+Asts': "PRA", 
       			'Points': "P"}


def get_projections():
	with open("projections.json") as f:
		tst = json.load(f)
	df_lines = pd.DataFrame([i["attributes"] for i in tst["data"]])
	df_lines["id"] = [i["relationships"]["new_player"]["data"]["id"] for i in tst["data"]]
	df_players = pd.DataFrame([i["attributes"] for i in tst["included"]])
	df_players["id"] = [i["id"] for i in tst["included"]]
	df_nba = df_players[df_players["league"]=="NBA"]
	df_projections = df_lines.merge(df_nba[["id", "display_name"]], how="inner", on="id")
	#df_projections = df_projections[df_projections["odds_type"]=="standard" | df_projections["odds_type"]=="standard" ]
	df_projections["stat_type"] = [stat_trans[i] if i in stat_trans else "N/A" for i in df_projections["stat_type"]]
	df_projections = df_projections[df_projections["stat_type"] != "N/A"]
	df_projections = df_projections.rename(columns = {"stat_type": "Line", "display_name": "Player"})
	return df_projections


def get_games_by_date(date = pd.to_datetime("today", utc=True)):

	url = "https://api-nba-v1.p.rapidapi.com/games"
	querystring = {"date":date.strftime("%Y-%m-%d")}
	response = requests.get(url, headers=headers, params=querystring)
	games = pd.DataFrame(response.json()["response"])
	return games


def return_team(team_name):

	# Get teams
	team_url = "https://api-nba-v1.p.rapidapi.com/teams"
	response = requests.get(team_url, headers=headers)
	resp = response.json()["response"]
	teams = [i for i in resp if i["nbaFranchise"] and not i["allStar"]]
	teams = pd.DataFrame(teams)
	teams["l_name"] = teams["name"].apply(lambda x: x.lower())

	# Have teams now, get team ID and query player ID

	team_id = [_id for _id, name in zip(teams["id"], teams["l_name"]) if team_name in name]
	team_id = team_id[0]

	return team_id


def get_comp_teams(team_name):
	df_rtg = pd.read_csv("d_rating_grps.csv")
	df_rtg = df_rtg[1:]
	df_rtg["l_team"] = df_rtg["TEAM"].apply(lambda x: x.lower())
	for idx, i in enumerate(df_rtg["l_team"]):
		if team_name in i:
			grp = idx//5
			break

	team_grp = df_rtg[max(0,idx-2):max(5,min(len(df_rtg),idx+3))]
	return [return_team(i.split(' ')[1]) for i in team_grp["l_team"]]


def return_players(team_id):

	# Get player ID
	player_url = "https://api-nba-v1.p.rapidapi.com/players"
	querystring = {"team":team_id,"season":current_season}
	response = requests.get(player_url, headers=headers, params=querystring)
	players_data = pd.DataFrame(response.json()["response"])
	players_data["l_name"] = [i.lower() + " " + j.lower() for i,j in zip(players_data["firstname"], players_data["lastname"])]

	return players_data


def return_team_games(team_id):
	
	# Get games metadata
	games_url = "https://api-nba-v1.p.rapidapi.com/games"
	querystring = {"season": current_season,"team": team_id}
	response = requests.get(games_url, headers=headers, params=querystring)
	games = pd.DataFrame(response.json()["response"])
	games["home"] = games["teams"].apply(lambda x: x["home"]["id"])
	games["away"] = games["teams"].apply(lambda x: x["visitors"]["id"])
	games["game_id"] = games["id"]
	games["dt"] = games["date"].apply(lambda x: x["start"])
	games["dt"] = pd.to_datetime(games["dt"])
	return games


def get_player_stats(player_id, team_id):

	# Get stats
	stat_url = "https://api-nba-v1.p.rapidapi.com/players/statistics"
	querystring = {"id":player_id,"season":current_season, "team": team_id}
	response = requests.get(stat_url, headers=headers, params=querystring)
	stats = pd.DataFrame(response.json()["response"])

	stats["game_id"] = stats["game"].apply(lambda x: x["id"])

	return stats


def run_bayes(stats, games, opponent_ids, at_home, params, back_to_back):

	stats["opposing"] = [i in opponent_ids or j in opponent_ids for i,j in zip(stats["home"], stats["away"])]

	l5_games = stats[-5:]
	l7_games = stats[-7:]
	l10_games = stats[-11:-6]
	l20_games = stats[-21:-11]
	away_games = stats[stats["at_home"]==False]
	home_games = stats[stats["at_home"]==True]
	opponent_games = stats[stats["opposing"]==True]
	consec = stats[stats["dt_diff"] <= pd.Timedelta(hours=26)] if back_to_back else stats[stats["dt_diff"] > pd.Timedelta(hours=26)]

	if len(opponent_games) == 1:
		l10_games = pd.concat([l10_games, opponent_games])

	# Build distribution 
	x = np.array(stats[params[0]])
	for i in range(1, len(params)):
		x += np.array(stats[params[i]])
	x = x[x>0]
	data = x
	if len(data)==0:
		return np.nan, np.nan

	mu_0, std_0 = norm.fit(data)
	tau_0 = 1/std_0**2

	# plot histogram 
	lin_arr = np.linspace(0, max(data), len(data))
	pdf = norm.pdf(lin_arr, mu_0, std_0)
	#ax.plot(lin_arr, pdf, 'r', label="prior dist")

	colors = ["blue", "pink", "orange", "green", "purple"]
	labels = ["home/away skew", "l10 skew", "opponent skew"]

	at_games = home_games if at_home else away_games

	for i, evidence in enumerate([at_games, consec, opponent_games, l7_games]):
		x = np.array(evidence[params[0]])
		for j in range(1, len(params)):
			x += np.array(evidence[params[j]])
		x = x[x>0]

		if len(x) < 2:
			continue 

		n = len(evidence)
		data = x
		mu, std = norm.fit(data)
		tau = 1/std**2
		post_mu = 1/(tau_0 + n*tau)*(mu_0*tau_0 + np.sum(data)*tau)
		post_std_2 = 1/(tau_0 + n*tau)
		mu_0 = post_mu
		std_0 = np.sqrt(post_std_2 + 1/tau)
		tau_0 = 1/std_0**2

	return mu_0, std_0 


