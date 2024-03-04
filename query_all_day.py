import requests
import os
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
import numpy as np
import matplotlib.pyplot as plt


from utils import return_players, return_team, return_team_games, run_bayes, get_player_stats,get_games_by_date, get_projections, get_comp_teams
#####

# Pull games
CTR_DATE =  pd.to_datetime("today", utc=True) + pd.Timedelta(days=1)
games_df = get_games_by_date(date = CTR_DATE)
games = []
for game in games_df["teams"]:
	games.append((game["home"]["nickname"].lower(), game["visitors"]["nickname"].lower()))

params_dict = {	"P": ["points"],
			"RA": ["totReb", "assists"],
			"PRA": ["points", "totReb", "assists"],
			"A": ["assists"],
			"R": ["totReb"],
			"PA": ["points", "assists"],
			"PR": ["points", "totReb"]
			}


print(f"Building lines for {games}")

master_book = pd.read_csv("master_book.csv")
master_book["date"] = pd.to_datetime(master_book["date"])
master_book = master_book.dropna(subset=["date"])
dates = [x.strftime("%Y-%m-%d") for x in master_book["date"].unique()]
today_str = pd.to_datetime("today").strftime("%Y-%m-%d")

if today_str not in dates:

	players = []
	param_codes = []
	means = []
	stdvs = []
	opposing = []

	for (home, away) in games:

		home_team = return_team(home)
		home_team_comps = get_comp_teams(home)
		away_team = return_team(away)
		away_team_comps = get_comp_teams(away)
		home_players = return_players(home_team)
		away_players = return_players(away_team)
		home_games = return_team_games(home_team)
		away_games = return_team_games(away_team)

		for i, row in home_players.iterrows():

			player_id = row["id"]
			player_name = row["firstname"] + " " +  row["lastname"]
			stats = get_player_stats(player_id, home_team)
			print(player_name, player_id)
			stats = stats.merge(home_games[["game_id", "home", "away", "dt"]], on="game_id")
			stats = stats.sort_values(by="dt")
			stats["at_home"] = [i==home_team for i in stats["home"]]
			stats["opposing"] = [i==away_team or j==away_team for i,j in zip(stats["home"], stats["away"])]
			stats["dt_diff"] = stats.dt.diff().shift(-1)
			curr_delta = pd.to_datetime("today", utc=True) - stats["dt"].max()
			back_to_back = curr_delta < pd.Timedelta(days=1)

			if len(stats)<20:
				print(f'{player_name} insufficient data')
				continue

			for param_code, params in params_dict.items():
				avg_val, deviations = run_bayes(stats, home_games, away_team_comps, True, params, back_to_back)
				print(f"Compute successful for {player_name}, {param_code}")			
				players.append(player_name)
				param_codes.append(param_code)
				means.append(avg_val)
				stdvs.append(deviations)
				opposing.append(home)


		for i, row in away_players.iterrows():
			player_id = row["id"]
			player_name = row["firstname"] + " " +  row["lastname"]
			stats = get_player_stats(player_id, away_team)
			stats = stats.merge(away_games[["game_id", "home", "away", "dt"]], on="game_id")
			stats = stats.sort_values(by="dt")
			stats = stats[stats["dt"] >= pd.to_datetime("2023-10-28", utc=True)]
			stats["at_home"] = [i==away_team for i in stats["home"]]
			stats["opposing"] = [i==home_team or j==home_team for i,j in zip(stats["home"], stats["away"])]
			print(stats["dt"].max())
			stats["dt_diff"] = stats.dt.diff().shift(-1)
			curr_delta = pd.to_datetime("today", utc=True) - stats["dt"].max()
			back_to_back = curr_delta < pd.Timedelta(days=1)


			if len(stats)<20:
				print(f'{player_name} insufficient data')
				continue

			for param_code, params in params_dict.items():
				avg_val, deviations = run_bayes(stats, away_games, home_team_comps, False, params, back_to_back)
				print(f"Compute successful for {player_name}, {param_code}")
				players.append(player_name)
				param_codes.append(param_code)
				means.append(avg_val)
				stdvs.append(deviations)
				opposing.append(away)

	book = pd.DataFrame({"Player": players, "Line": param_codes, "EV": means, "STD": stdvs, "Team": opposing})
	master_book = pd.concat([master_book, book])
	master_book["date"] = today_str
	master_book.to_csv("master_book.csv", index=False)

df_projections = get_projections()

book = master_book.groupby(["Player", "Line", "EV", "STD", "Team"], as_index = False)["date"].last()
book = book.merge(df_projections[["Player", "Line", "line_score", "odds_type"]], how="left", on=["Player", "Line"])
book["Z-Score"] = (book["line_score"] - book["EV"])/book["STD"]
book["Proba Under"] = norm.cdf(book["Z-Score"])
book["Proba Over"] = 1 - book["Proba Under"]
book["Over/Under"] = ["Over" if x>.5 else "Under" for x in book["Proba Over"]]
book["Max Proba"] = [max(i,j ) for i,j in zip(book["Proba Over"], book["Proba Under"])]
book = book.drop(columns = ["date"])
book = book.dropna(subset = "line_score")
book = book.sort_values(by="Max Proba", ascending=False)
book.to_csv(f"{pd.to_datetime('today').strftime('%Y_%m_%d_%H_%M_%S')}.csv", index=False)
