"""Download and process season-long data, compile stats, and creates visuals."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from os.path import join
from pathlib import Path
from plotly.subplots import make_subplots
from re import search
from .constants import *
from .game import Game, get_player_rates
from .utils import (
    get_data_path,
    get_json_path,
    get_games_path,
    upload_to_bucket,
    download_from_bucket,
)


class Season:
    """This class contains methods for retrieving all of the advanced AUDL stats for a single season."""

    def __init__(
        self, year=CURRENT_YEAR, data_path="data", upload=False, download=False
    ):
        """Initialize parameters of season data.

        Args:
            year (int, optional): Season to get stats from. Currently not used because there are only
                advanced stats for a single season (2021).
                Defaults to CURRENT_YEAR.
            data_path (str, optional): The path to the folder where data
                will be stored.
            upload (bool, optional): Whether to upload data to AWS bucket.
            download (bool, optional): Whether to download data from AWS bucket if it exists.

        """
        # Inputs
        self.upload = upload
        self.download = download
        self.year = year
        self.data_path = get_data_path(data_path)
        self.json_path = get_json_path(self.data_path, "games_raw")
        self.games_path = get_games_path(self.data_path, "all_games")
        self.league_info_path = get_games_path(self.data_path, "league_info")
        self.stats_path = get_games_path(self.data_path, "stats")

        # Create directories if they don't exist
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        Path(self.json_path).mkdir(parents=True, exist_ok=True)
        Path(self.games_path).mkdir(parents=True, exist_ok=True)
        Path(self.league_info_path).mkdir(parents=True, exist_ok=True)
        Path(self.stats_path).mkdir(parents=True, exist_ok=True)

        # URLs to retrieve data from
        self.schedule_url = SCHEDULE_URL
        self.stats_url = STATS_URL
        self.weeks_urls = None
        self.game_info = None

        # All processed data
        self.games = None
        self.start_of_opoints = None
        self.teams = None
        self.players = None
        self.player_stats_by_game = None
        self.team_stats_by_game = None
        self.team_stats_by_season = None

        # QC
        self.game_qc = None

    def get_paginated_urls(self):
        """Get URLs for the schedule for every game of the season."""
        if self.paginated_urls is None:
            # # Get urls for all weeks
            # schedule_page = requests.get(self.schedule_url.format(page=1))
            # schedule_soup = BeautifulSoup(schedule_page.text, "html.parser")

            # # Extract urls from document
            # pages = []
            # for page in schedule_soup.find_all("a"):
            #     if "schedule" in page.get("href").lower():
            #         page_url = self.schedule_url + page.get("href")[16:]
            #         pages.append(page_url)

            self.paginated_urls = [self.schedule_url.format(page=i, year=self.year) for i in range(1, 25)]

        return self.paginated_urls

    def get_game_info(self, override=False, upload=None, download=None):
        """Get teams, date, and url for the advanced stats page of every game in the season."""
        if self.game_info is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            # If file doesn't exists locally, try to retrieve it from AWS
            game_info_path = join(self.league_info_path, "game_info.feather")
            if not Path(game_info_path).is_file() and download and not override:
                download_from_bucket(game_info_path)

            # If file exists locally, load it
            if Path(game_info_path).is_file() and not override:
                df = pd.read_feather(game_info_path)

            else:
                games = []
                for page_url in self.get_paginated_urls():
                    # Get schedule for 1 page
                    page = requests.get(page_url)
                    page_soup = BeautifulSoup(page.text, "html.parser")

                    # Get all game URLs in 1 page
                    for game_center in page_soup.find_all(
                        "div", {"class": "svelte-game-header-links"}
                    ):
                        game_url = (
                            self.stats_url + game_center.find("a").get("href")[8:]
                        )
                        if game_url not in games:
                            games.append(game_url)

                # Parse URLs to get game info
                game_list = []
                for x in sorted(games):
                    game_response = Game(
                        game_url=x, upload=upload, download=download
                    ).get_response()
                    if game_response is None:
                        game_exists = False
                        playoffs = False
                    else:
                        # Determine whether the game has happened yet
                        tsghome = game_response.get("tsgHome", dict())
                        if tsghome is None:
                            game_exists = False
                        elif tsghome.get("events", None) == "[]":
                            game_exists = False
                        elif tsghome.get("events", None) is not None:
                            game_exists = True
                        else:
                            game_exists = False

                        # Determine if it's a playoff game
                        playoffs = not game_response.get("game", dict()).get(
                            "reg_season", False
                        )

                        # Division championship games are not currently coded as playoffs in the response,
                        # but they should be
                        if game_response.get("game", dict()).get(
                            "id", -1
                        ) in PLAYOFF_GAMES.get(self.year, []):
                            playoffs = True

                    game_list.append(
                        [
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(1),
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(2),
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(3),
                            x,
                            game_exists,
                            playoffs,
                        ]
                    )

                # Save info to file
                df = (
                    pd.DataFrame(
                        data=game_list,
                        columns=[
                            "game_date",
                            "away_team",
                            "home_team",
                            "url",
                            "events_exist",
                            "playoffs",
                        ],
                    )
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                df.to_feather(game_info_path)

            if upload:
                upload_to_bucket(game_info_path)

            self.game_info = df

        return self.game_info

    def get_games(
        self,
        small_file=False,
        build_new_file=False,
        upload=None,
        download=None,
        qc=False,
    ):
        """Download and process all game data."""
        needed_columns = [
            "game_id",
            "team_id",
            "opponent_team_id",
            "period",
            "t",
            "t_after",
            "r",
            "r_after",
            "x",
            "y",
            "x_after",
            "y_after",
            "s_before",
            "o_point",
            "possession_outcome_general",
            "throw_outcome",
            "yyards",
            "yyards_raw",
            "xyards",
            "xyards_raw",
            "yards",
            "yards_raw",
        ]
        if self.games is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            file_name_small = join(self.games_path, f"all_games_small.feather")
            file_name = join(self.games_path, f"all_games.feather")
            # Get either the file with all columns or only some
            if small_file:
                all_games_file_name = file_name_small
            else:
                all_games_file_name = file_name

            # If file doesn't exist locally, try to retrieve it from AWS
            if (
                not Path(all_games_file_name).is_file()
                and download
                and not build_new_file
            ):
                download_from_bucket(all_games_file_name)

            # If file exists locally, load it
            if Path(all_games_file_name).is_file() and not build_new_file:
                self.games = pd.read_feather(all_games_file_name)

            # Compile data if file does not already exist
            else:
                all_games = []
                for i, row in self.get_game_info(
                    upload=upload, download=download, override=build_new_file
                ).iterrows():
                    try:
                        if qc:
                            # Print game info
                            print(
                                row["game_date"],
                                row["away_team"],
                                "at",
                                row["home_team"],
                            )

                        # Get the game object
                        g = Game(game_url=row["url"], upload=upload, download=download)
                        events_home_file = g.get_events_filename(home=True)
                        events_away_file = g.get_events_filename(home=True)

                        # Get and process the game events if they don't already exist
                        if not Path(events_home_file).is_file():
                            all_games.append(g.get_home_events(qc=qc))
                        if not Path(events_away_file).is_file():
                            all_games.append(g.get_away_events(qc=qc))
                    except Exception as e:
                        if qc:
                            print(e)
                        pass
                self.games = pd.DataFrame(pd.concat(all_games))
                self.games.reset_index(drop=True).to_feather(file_name)

                self.games[needed_columns].reset_index(drop=True).to_feather(
                    file_name_small
                )
                if upload:
                    upload_to_bucket(file_name)
                    upload_to_bucket(file_name_small)
        if small_file:
            return self.games[needed_columns]
        else:
            return self.games

    def get_start_of_opoints(
        self, upload=None, download=None,
    ):
        """Download and process all game data."""
        if self.start_of_opoints is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            file_name = join(self.games_path, f"start_of_opoints.feather")

            # If file doesn't exist locally, try to retrieve it from AWS
            if not Path(file_name).is_file() and download:
                download_from_bucket(file_name)

            # If file exists locally, load it
            if Path(file_name).is_file():
                self.start_of_opoints = pd.read_feather(file_name)
            else:
                self.start_of_opoints = self.get_games(small_file=True).query("t==1")
                self.start_of_opoints.reset_index().to_feather(file_name)

        return self.start_of_opoints

    def get_teams(self, upload=None, download=None, qc=False):
        """Get all teams and team IDs from game data and save it."""
        if self.teams is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            file_name = join(self.league_info_path, "teams.feather")
            # If file doesn't exist locally, try to retrieve it from AWS
            if not Path(file_name).is_file() and download:
                download_from_bucket(file_name)

            # If file exists locally, load it
            if Path(file_name).is_file():
                self.teams = pd.read_feather(file_name)

            # Compile data if file does not already exist
            else:
                team_ids = (
                    self.get_games(
                        small_file=False,
                        build_new_file=False,
                        upload=upload,
                        download=download,
                        qc=False,
                    )["team_id"]
                    .unique()
                    .tolist()
                )
                team_data = []
                for i, row in self.get_game_info(
                    upload=upload, download=download, override=False
                ).iterrows():
                    # Get games until all teams have been found
                    if len(team_ids) == 0:
                        break
                    g = Game(game_url=row["url"])
                    home_team_name = (
                        g.get_home_team()["city"].iloc[0]
                        + " "
                        + g.get_home_team()["name"].iloc[0]
                    )
                    home_team_abbrev = g.get_home_team()["abbrev"].iloc[0]
                    home_team_id = g.get_home_team()["team_id"].iloc[0]
                    away_team_name = (
                        g.get_away_team()["city"].iloc[0]
                        + " "
                        + g.get_away_team()["name"].iloc[0]
                    )
                    away_team_abbrev = g.get_away_team()["abbrev"].iloc[0]
                    away_team_id = g.get_away_team()["team_id"].iloc[0]
                    if home_team_id in team_ids:
                        team_ids.pop(team_ids.index(home_team_id))
                        team_data.append(
                            [home_team_id, home_team_abbrev, home_team_name]
                        )
                    if away_team_id in team_ids:
                        team_ids.pop(team_ids.index(away_team_id))
                        team_data.append(
                            [away_team_id, away_team_abbrev, away_team_name]
                        )

                self.teams = (
                    pd.DataFrame(
                        data=team_data, columns=["team_id", "team_abbrev", "team_name"]
                    )
                    .sort_values("team_name")
                    .reset_index(drop=True)
                )
                self.teams.to_feather(file_name)
                if upload:
                    upload_to_bucket(file_name)

        return self.teams

    def get_players(self, upload=None, download=None, qc=False):
        """Get all players and player IDs from game data and save it."""
        if self.players is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            file_name = join(self.league_info_path, "players.feather")
            # If file doesn't exist locally, try to retrieve it from AWS
            if not Path(file_name).is_file() and download:
                download_from_bucket(file_name)

            # If file exists locally, load it
            if Path(file_name).is_file():
                self.players = pd.read_feather(file_name)

            # Compile data if file does not already exist
            else:
                player_ids = [
                    int(x)
                    for x in list(self.get_games(upload=upload, download=download))
                    if x.isdigit()
                ]
                player_data = []
                for i, row in self.get_game_info(
                    upload=upload, download=download, override=False
                ).iterrows():
                    # Get games until all teams have been found
                    if len(player_ids) == 0:
                        break
                    g = Game(game_url=row["url"])
                    home_team_player_ids = g.get_home_roster()["id"].values.tolist()
                    away_team_player_ids = g.get_away_roster()["id"].values.tolist()
                    home_team_ids = [
                        g.get_home_team()["team_id"].iloc[0]
                        for _ in home_team_player_ids
                    ]
                    away_team_ids = [
                        g.get_away_team()["team_id"].iloc[0]
                        for _ in away_team_player_ids
                    ]
                    home_team_player_names = (
                        g.get_home_roster()
                        .assign(
                            player_name=lambda x: x["last_name"]
                            .str.strip()
                            .str.capitalize()
                            + ", "
                            + x["first_name"].str.strip().str.capitalize()
                        )["player_name"]
                        .values.tolist()
                    )
                    away_team_player_names = (
                        g.get_away_roster()
                        .assign(
                            player_name=lambda x: x["last_name"]
                            .str.strip()
                            .str.capitalize()
                            + ", "
                            + x["first_name"].str.strip().str.capitalize()
                        )["player_name"]
                        .values.tolist()
                    )
                    for pid, pname, teamid in zip(
                        home_team_player_ids + away_team_player_ids,
                        home_team_player_names + away_team_player_names,
                        home_team_ids + away_team_ids,
                    ):
                        if pid in player_ids:
                            player_ids.pop(player_ids.index(pid))
                            player_data.append([pid, pname, teamid])

                self.players = (
                    pd.DataFrame(
                        data=player_data,
                        columns=["player_id", "player_name", "team_id"],
                    )
                    .sort_values("player_name")
                    .reset_index(drop=True)
                )
                self.players.to_feather(file_name)
                if upload:
                    upload_to_bucket(file_name)

        return self.players

    def get_player_stats_by_game(self, upload=None, download=None):
        """Compile all player game stats for the season into single dataframe."""
        if self.player_stats_by_game is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            file_name = join(self.stats_path, "player_stats_by_game.feather")
            # If file doesn't exist locally, try to retrieve it from AWS
            if not Path(file_name).is_file() and download:
                download_from_bucket(file_name)

            # If file exists locally, load it
            if Path(file_name).is_file():
                self.player_stats_by_game = pd.read_feather(file_name)

            else:
                # Get all games that have events (they've actually happened)
                existing_games = (
                    self.get_game_info(override=False)
                    .query("events_exist==True")
                    .copy()
                )

                player_stats = []
                # Process each game to get player segments
                for i, row in existing_games.iterrows():
                    g = Game(row["url"])
                    home_stats = g.get_player_stats_by_game(home=True)
                    away_stats = g.get_player_stats_by_game(home=False)
                    player_stats.extend([home_stats, away_stats])

                self.player_stats_by_game = pd.concat(player_stats, ignore_index=True)

                # Only keep players who are listed on any roster - removes invalid player IDs e.g. -1
                self.player_stats_by_game.loc[
                    self.player_stats_by_game["playerid"].isin(
                        self.get_players()["player_id"].astype(int).astype(str).unique()
                    )
                ]
                self.player_stats_by_game.to_feather(file_name)

        return self.player_stats_by_game

    def get_team_stats_by_game(self):
        """Compile all team game stats for the season into single dataframe."""
        # TODO: Season team stats by game
        pass

    def get_player_stats_by_season(self, playoffs="all", upload=None, download=None):
        """Compile and aggregate all player stats for the season into single dataframe."""
        if upload is None:
            upload = self.upload
        if download is None:
            download = self.download

        if playoffs == "all":
            playoffs = [True, False]
            playoffs_str = "all"
        elif playoffs:
            playoffs = [playoffs]
            playoffs_str = "playoffs"
        else:
            playoffs = [playoffs]
            playoffs_str = "reg"

        file_name = join(
            self.stats_path, f"player_stats_by_season_{playoffs_str}.feather"
        )
        # If file doesn't exist locally, try to retrieve it from AWS
        if not Path(file_name).is_file() and download:
            download_from_bucket(file_name)

        # If file exists locally, load it
        if Path(file_name).is_file():
            dfout = pd.read_feather(file_name)

        else:
            # Get player stats by game
            df = self.get_player_stats_by_game(upload=upload, download=download)
            info_cols = [
                "playerid",
                "name",
                "team",
                "opponent",
                "game_date",
                "year",
                "playoffs",
            ]
            stat_cols = [col for col in list(df) if col not in info_cols]

            dfout = (
                df.query(f"playoffs=={playoffs}")
                .groupby(["playerid", "name", "team", "year",])[stat_cols]
                .sum()
                .reset_index()
                .pipe(get_player_rates)
                # .pipe(round_player_stats)
            )

            dfout.to_feather(file_name)

        return dfout

    def get_team_stats_by_season(self):
        """Compile and aggregate all team stats for the season into single dataframe."""
        # TODO: Season team stats
        pass

    def visual_field_heatmap_horizontal(
        self,
        outcome_measure,
        outcome,
        metric,
        x_cut=None,
        y_cut=None,
        x_min=-27,
        x_max=27,
        y_min=0,
        y_max=120,
        xyards_min=-60,
        xyards_max=60,
        yyards_min=-120,
        yyards_max=120,
        yards_min=0,
        yards_max=300,
        zmin=None,
        zmax=None,
        o_point=None,
        remove_ob_pull=False,
        throw=True,
        team_ids=None,
        opposing_team_ids=None,
        player_ids=None,
        game_ids=None,
        second_graph=False,
    ):
        """View frequency of possession, scores, turns on the field, similar to shot chart.

        Args:
            outcome_measure (str): Can be throw_outcome or possession_outcome_general.
            outcome (str): Can be Completion or Turnover.
            metric (str): Can be count, pct, yards_raw, yyards_raw, or xyards.

        """
        df = self.get_games(small_file=True, build_new_file=False, qc=False,)

        # Set whether heat map should be for the throw or the catch
        if throw:
            suffix = ""
            opposite_suffix = "_after"
        else:
            suffix = "_after"
            opposite_suffix = ""

        # Run through initial filters
        df = (
            df.query(f"x{suffix} >= {x_min}")
            .query(f"x{suffix} <= {x_max}")
            .query(f"y{suffix} >= {y_min}")
            .query(f"y{suffix} <= {y_max}")
            .query(f"xyards_raw >= {xyards_min}")
            .query(f"xyards_raw <= {xyards_max}")
            .query(f"yyards_raw >= {yyards_min}")
            .query(f"yyards_raw <= {yyards_max}")
            .query(f"yards_raw >= {yards_min}")
            .query(f"yards_raw <= {yards_max}")
        )
        if team_ids is not None:
            df = df.loc[df["team_id"].isin(team_ids)]
        if opposing_team_ids is not None:
            df = df.loc[df["opponent_team_id"].isin(opposing_team_ids)]
        if player_ids is not None:
            if second_graph:
                df = df.loc[df[f"r{opposite_suffix}"].isin(player_ids)]
            else:
                df = df.loc[df[f"r{suffix}"].isin(player_ids)]
        if game_ids is not None:
            df = df.loc[df["game_id"].isin(game_ids)]
        if remove_ob_pull:
            df = df.loc[~((df["x"] == 0) & (df["y"] == 40))]
        if o_point is None:
            o_point = "o_point"

        # x_cut and y_cut are coordinates that are
        # the square where the throw came from if throw=False and the
        # square where the throw went if throw=True
        if x_cut is None:
            x_cut = f"x_cut{opposite_suffix}"
        if y_cut is None:
            y_cut = f"y_cut{opposite_suffix}"

        # Set binning ranges
        x_bins = np.linspace(-27, 27, 6)
        y_bins = np.linspace(0, 120, 13)

        # Get data for heatmap
        df = (
            df.query(f"x{suffix}==x{suffix}")
            .query("t_after==[8, 10, 12, 13, 17, 19, 20, 22, 23, 24, 25, 26, 27]")
            .query(f"o_point=={o_point}")
            .assign(
                x_cut=lambda x: pd.cut(
                    x[f"x"], bins=x_bins, labels=[i for i in range(len(x_bins) - 1)],
                ),
                y_cut=lambda x: pd.cut(
                    x[f"y"], bins=y_bins, labels=[i for i in range(len(y_bins) - 1)],
                ),
                x_cut_after=lambda x: pd.cut(
                    x[f"x_after"],
                    bins=x_bins,
                    labels=[i for i in range(len(x_bins) - 1)],
                ),
                y_cut_after=lambda x: pd.cut(
                    x[f"y_after"],
                    bins=y_bins,
                    labels=[i for i in range(len(y_bins) - 1)],
                ),
                x_cut_final=lambda x: x[f"x_cut{suffix}"],
                y_cut_final=lambda x: x[f"y_cut{suffix}"],
            )
            .query(f"x_cut{opposite_suffix}=={x_cut}")
            .query(f"y_cut{opposite_suffix}=={y_cut}")
            .groupby(["x_cut_final", "y_cut_final", outcome_measure])
            .agg(
                {
                    "game_id": "count",
                    "yards": np.mean,
                    "yards_raw": np.mean,
                    "yyards": np.mean,
                    "yyards_raw": np.mean,
                    "xyards": np.mean,
                    "xyards_raw": np.mean,
                }
            )
            .reset_index()
            .rename(columns={"game_id": "count"})
            .set_index(["x_cut_final", "y_cut_final", outcome_measure])
            .assign(
                freq=lambda x: x.groupby(level=["x_cut_final", "y_cut_final"])[
                    "count"
                ].sum(),
                pct=lambda x: x["count"] / x["freq"],
                hovertext=lambda x: f"Total Touches: "
                + x["freq"].apply(lambda y: f"{y:.0f}")
                + "<br>"
                + f"# of {outcome}s: "
                + x["count"].apply(lambda y: f"{y:.0f}")
                + np.where(
                    x["count"] > 0,
                    "<br>" + f"{outcome} Pct: "
                    # + (x["pct"].round(3) * 100).astype(str).str[:4]
                    + x["pct"].apply(lambda y: f"{y:.1%}")
                    + "<br>"
                    + "Avg Sideways Yards: "
                    + x["xyards"].apply(lambda y: f"{y:.1f}")
                    + "<br>"
                    + "Avg Downfield Yards: "
                    + x["yyards_raw"].apply(lambda y: f"{y:.1f}"),
                    "",
                ),
            )
            .reset_index()
            .query(f"{outcome_measure}=='{outcome}'")
        )

        # Set additional args depending on inputs
        kwargs = dict(reversescale=False)
        if metric == "pct":
            kwargs["colorbar_tickformat"] = ".0%"
            if (zmin is None) or (zmax is None):
                kwargs["zauto"] = True
            else:
                kwargs["zmin"] = zmin
                kwargs["zmax"] = zmax
        elif metric == "count":
            kwargs["zauto"] = True
            kwargs["colorbar_tickformat"] = ".0,"
        else:
            kwargs["zauto"] = True
            kwargs["colorbar_tickformat"] = ".0,"

        # Create figure
        fighm = go.Figure(
            data=go.Heatmap(
                x=df["y_cut_final"],
                y=df["x_cut_final"],
                z=df[metric],
                connectgaps=False,
                zsmooth="best",
                colorscale="portland",
                showscale=True,
                customdata=df["hovertext"],
                # Move colorbar to left of plot
                colorbar_x=0,
                colorbar_y=1,
                colorbar_xanchor="right",
                colorbar_yanchor="top",
                colorbar_lenmode="fraction",
                colorbar_len=0.94,
                colorbar_ticklabelposition="inside bottom",
                colorbar_tickfont_color="black",
                **kwargs,
            )
        )

        # Draw field boundaries
        # Vertical lines
        fighm.add_shape(
            type="line", y0=-0.5, y1=4.5, x0=-0.5, x1=-0.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", y0=-0.5, y1=4.5, x0=1.5, x1=1.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", y0=-0.5, y1=4.5, x0=9.5, x1=9.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", y0=-0.5, y1=4.5, x0=11.5, x1=11.5, line=dict(color="black")
        )

        # Horizontal lines
        fighm.add_shape(
            type="line", y0=-0.5, y1=-0.5, x0=-0.5, x1=11.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", y0=4.5, y1=4.5, x0=-0.5, x1=11.5, line=dict(color="black")
        )

        # Add arrow to indicate attacking direction
        fighm.add_annotation(
            xref="x",
            yref="y",
            y=4.75,
            x=6,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            axref="x",
            ayref="y",
            ay=4.75,
            ax=5,
            text="Attacking",
        )

        # Set layout properties
        left_margin = 40
        fighm.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                autorange="reversed",
                scaleanchor="x",
                scaleratio=1,
            ),
            # Add tick labels to fig
            xaxis=dict(
                # range=[-1, 121],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=0, l=left_margin, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hm = "%{customdata}<extra></extra>"
        fighm.update_traces(hovertemplate=hovertext_hm)

        # Create histograms to display frequency of events near heatmap
        fighy = px.histogram(
            df,
            x="y_cut_final",
            y="count",
            nbins=len(y_bins) - 1,
            histfunc="sum",
            range_x=[-0.5, 11.5],
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.D3,
        )

        # Set histy layout properties
        fighy.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Add tick labels to fig
            xaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=0, l=left_margin, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hy = "<br>".join([f"{outcome}s:", "%{y}", "<extra></extra>",])
        fighy.update_traces(hovertemplate=hovertext_hy)

        fighx = px.histogram(
            df,
            y="x_cut_final",
            x="count",
            nbins=len(x_bins) - 1,
            histfunc="sum",
            orientation="h",
            range_y=[-0.5, 4.5],
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.D3,
        )

        # Set histx layout properties
        fighx.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                autorange="reversed",
                fixedrange=True,
            ),
            # Add tick labels to fig
            xaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=21, l=0, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hx = "<br>".join([f"{outcome}s:", "%{x}", "<extra></extra>",])
        fighx.update_traces(hovertemplate=hovertext_hx)

        return fighm, fighx, fighy

    def visual_field_heatmap_vertical(
        self,
        outcome_measure,
        outcome,
        metric,
        x_cut=None,
        y_cut=None,
        x_min=-27,
        x_max=27,
        y_min=0,
        y_max=120,
        xyards_min=-60,
        xyards_max=60,
        yyards_min=-120,
        yyards_max=120,
        yards_min=0,
        yards_max=300,
        zmin=None,
        zmax=None,
        o_point=None,
        remove_ob_pull=False,
        throw=True,
        team_ids=None,
        opposing_team_ids=None,
        player_ids=None,
        game_ids=None,
        second_graph=False,
    ):
        """View frequency of possession, scores, turns on the field, similar to shot chart.

        Args:
            outcome_measure (str): Can be throw_outcome or possession_outcome_general.
            outcome (str): Can be Completion or Turnover.
            metric (str): Can be count, pct, yards_raw, yyards_raw, or xyards.

        """
        df = self.get_games(small_file=True, build_new_file=False, qc=False,)

        # Set whether heat map should be for the throw or the catch
        if throw:
            suffix = ""
            opposite_suffix = "_after"
        else:
            suffix = "_after"
            opposite_suffix = ""

        # Run through initial filters
        df = (
            df.query(f"x{suffix} >= {x_min}")
            .query(f"x{suffix} <= {x_max}")
            .query(f"y{suffix} >= {y_min}")
            .query(f"y{suffix} <= {y_max}")
            .query(f"xyards_raw >= {xyards_min}")
            .query(f"xyards_raw <= {xyards_max}")
            .query(f"yyards_raw >= {yyards_min}")
            .query(f"yyards_raw <= {yyards_max}")
            .query(f"yards_raw >= {yards_min}")
            .query(f"yards_raw <= {yards_max}")
        )
        if team_ids is not None:
            df = df.loc[df["team_id"].isin(team_ids)]
        if opposing_team_ids is not None:
            df = df.loc[df["opponent_team_id"].isin(opposing_team_ids)]
        if player_ids is not None:
            if second_graph:
                df = df.loc[df[f"r{opposite_suffix}"].isin(player_ids)]
            else:
                df = df.loc[df[f"r{suffix}"].isin(player_ids)]
        if game_ids is not None:
            df = df.loc[df["game_id"].isin(game_ids)]
        if remove_ob_pull:
            df = df.loc[~((df["x"] == 0) & (df["y"] == 40))]
        if o_point is None:
            o_point = "o_point"

        # x_cut and y_cut are coordinates that are
        # the square where the throw came from if throw=False and the
        # square where the throw went if throw=True
        if x_cut is None:
            x_cut = f"x_cut{opposite_suffix}"
        if y_cut is None:
            y_cut = f"y_cut{opposite_suffix}"

        # Set binning ranges
        x_bins = np.linspace(-27, 27, 6)
        y_bins = np.linspace(0, 120, 13)

        # Get data for heatmap
        df = (
            df.query(f"x{suffix}==x{suffix}")
            .query("t_after==[8, 10, 12, 13, 17, 19, 20, 22, 23, 24, 25, 26, 27]")
            .query(f"o_point=={o_point}")
            .assign(
                x_cut=lambda x: pd.cut(
                    x[f"x"], bins=x_bins, labels=[i for i in range(len(x_bins) - 1)],
                ),
                y_cut=lambda x: pd.cut(
                    x[f"y"], bins=y_bins, labels=[i for i in range(len(y_bins) - 1)],
                ),
                x_cut_after=lambda x: pd.cut(
                    x[f"x_after"],
                    bins=x_bins,
                    labels=[i for i in range(len(x_bins) - 1)],
                ),
                y_cut_after=lambda x: pd.cut(
                    x[f"y_after"],
                    bins=y_bins,
                    labels=[i for i in range(len(y_bins) - 1)],
                ),
                x_cut_final=lambda x: x[f"x_cut{suffix}"],
                y_cut_final=lambda x: x[f"y_cut{suffix}"],
            )
            .query(f"x_cut{opposite_suffix}=={x_cut}")
            .query(f"y_cut{opposite_suffix}=={y_cut}")
            .groupby(["x_cut_final", "y_cut_final", outcome_measure])
            .agg(
                {
                    "game_id": "count",
                    "yards": np.mean,
                    "yards_raw": np.mean,
                    "yyards": np.mean,
                    "yyards_raw": np.mean,
                    "xyards": np.mean,
                    "xyards_raw": np.mean,
                }
            )
            .reset_index()
            .rename(columns={"game_id": "count"})
            .set_index(["x_cut_final", "y_cut_final", outcome_measure])
            .assign(
                freq=lambda x: x.groupby(level=["x_cut_final", "y_cut_final"])[
                    "count"
                ].sum(),
                pct=lambda x: x["count"] / x["freq"],
                hovertext=lambda x: f"Total Touches: "
                + x["freq"].apply(lambda y: f"{y:.0f}")
                + "<br>"
                + f"# of {outcome}s: "
                + x["count"].apply(lambda y: f"{y:.0f}")
                + np.where(
                    x["count"] > 0,
                    "<br>" + f"{outcome} Pct: "
                    # + (x["pct"].round(3) * 100).astype(str).str[:4]
                    + x["pct"].apply(lambda y: f"{y:.1%}")
                    + "<br>"
                    + "Avg Sideways Yards: "
                    + x["xyards"].apply(lambda y: f"{y:.1f}")
                    + "<br>"
                    + "Avg Downfield Yards: "
                    + x["yyards_raw"].apply(lambda y: f"{y:.1f}"),
                    "",
                ),
            )
            .reset_index()
            .query(f"{outcome_measure}=='{outcome}'")
        )

        # Set additional args depending on inputs
        kwargs = dict(reversescale=False)
        if metric == "pct":
            kwargs["colorbar_tickformat"] = ".0%"
            if (zmin is None) or (zmax is None):
                kwargs["zauto"] = True
            else:
                kwargs["zmin"] = zmin
                kwargs["zmax"] = zmax
        elif metric == "count":
            kwargs["zauto"] = True
            kwargs["colorbar_tickformat"] = ".0,"
        else:
            kwargs["zauto"] = True
            kwargs["colorbar_tickformat"] = ".0,"

        # Create figure
        fighm = go.Figure(
            data=go.Heatmap(
                y=df["y_cut_final"],
                x=df["x_cut_final"],
                z=df[metric],
                connectgaps=False,
                zsmooth="best",
                colorscale="portland",
                showscale=True,
                customdata=df["hovertext"],
                # Move colorbar to left of plot
                colorbar_x=0.37,
                colorbar_y=1,
                colorbar_xanchor="right",
                colorbar_yanchor="top",
                colorbar_lenmode="fraction",
                colorbar_len=1,
                colorbar_ticklabelposition="inside bottom",
                colorbar_tickfont_color="black",
                **kwargs,
            )
        )

        # Draw field boundaries
        # Vertical lines
        fighm.add_shape(
            type="line", x0=-0.5, x1=4.5, y0=-0.5, y1=-0.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", x0=-0.5, x1=4.5, y0=1.5, y1=1.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", x0=-0.5, x1=4.5, y0=9.5, y1=9.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", x0=-0.5, x1=4.5, y0=11.5, y1=11.5, line=dict(color="black")
        )

        # Horizontal lines
        fighm.add_shape(
            type="line", x0=-0.5, x1=-0.5, y0=-0.5, y1=11.5, line=dict(color="black")
        )
        fighm.add_shape(
            type="line", x0=4.5, x1=4.5, y0=-0.5, y1=11.5, line=dict(color="black")
        )

        # Add arrow to indicate attacking direction
        fighm.add_annotation(
            xref="x",
            yref="y",
            x=5.1,
            y=6,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            axref="x",
            ayref="y",
            ax=5.1,
            ay=5,
            text="Attacking",
        )

        # Set layout properties
        left_margin = 40
        fighm.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                scaleanchor="x",
                scaleratio=1,
            ),
            # Add tick labels to fig
            xaxis=dict(
                # range=[-1, 121],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=0, l=left_margin, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hm = "%{customdata}<extra></extra>"
        fighm.update_traces(hovertemplate=hovertext_hm)

        # Create histograms to display frequency of events near heatmap
        fighy = px.histogram(
            df,
            y="y_cut_final",
            x="count",
            nbins=len(y_bins) - 1,
            histfunc="sum",
            range_y=[-0.5, 11.5],
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.D3,
        )

        # Set histy layout properties
        fighy.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Add tick labels to fig
            xaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=0, l=left_margin, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hy = "<br>".join([f"{outcome}s:", "%{x}", "<extra></extra>",])
        fighy.update_traces(hovertemplate=hovertext_hy)

        fighx = px.histogram(
            df,
            x="x_cut_final",
            y="count",
            nbins=len(x_bins) - 1,
            histfunc="sum",
            range_x=[-0.5, 4.5],
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.D3,
        )

        # Set histx layout properties
        fighx.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Add tick labels to fig
            xaxis=dict(
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=21, l=0, r=0, autoexpand=False),
        )

        # Text to show on hover
        hovertext_hx = "<br>".join([f"{outcome}s:", "%{y}", "<extra></extra>",])
        fighx.update_traces(hovertemplate=hovertext_hx)

        return fighm, fighx, fighy

    def visual_field_heatmap_subplots_horizontal(self, fighm, fighx, fighy):
        """Combine heatmap and histograms into single plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[HEATMAP_RATIO_H_X, 1 - HEATMAP_RATIO_H_X],
            row_heights=[1 - HEATMAP_RATIO_H_Y, HEATMAP_RATIO_H_Y],
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
        )
        if len(fighm.data) > 0:
            fighmdata = fighm.data[0]
        else:
            fighmdata = None
        if len(fighx.data) > 0:
            fighxdata = fighx.data[0]
        else:
            fighxdata = None
        if len(fighy.data) > 0:
            fighydata = fighy.data[0]
        else:
            fighydata = None
        fig.add_trace(go.Heatmap(fighmdata), row=2, col=1)
        fig.add_trace(go.Histogram(fighxdata), row=2, col=2)
        fig.add_trace(go.Histogram(fighydata), row=1, col=1)
        left_margin = 40
        fig.update_layout(
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=20, l=left_margin, r=0, autoexpand=False),
        )
        fig.update_xaxes(
            row=2,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=2,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            autorange="reversed",
        )

        fig.update_xaxes(
            row=1,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )

        fig.update_xaxes(
            row=2,
            col=2,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=2,
            col=2,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            fixedrange=True,
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            y=-0.02,
            x=0.5,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            axref="pixel",
            ayref="pixel",
            ay=0,
            ax=-50,
            text="Attacking",
        )

        # Draw field boundaries
        # Vertical lines
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=0,
            y1=HEATMAP_RATIO_H_Y,
            x0=0,
            x1=0,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=HEATMAP_RATIO_H_Y,
            x0=HEATMAP_RATIO_H_X / 6,
            x1=HEATMAP_RATIO_H_X / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=HEATMAP_RATIO_H_Y,
            x0=HEATMAP_RATIO_H_X * 5 / 6,
            x1=HEATMAP_RATIO_H_X * 5 / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=HEATMAP_RATIO_H_Y,
            x0=HEATMAP_RATIO_H_X,
            x1=HEATMAP_RATIO_H_X,
            line=dict(color="black"),
        )

        # # Horizontal lines
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=0,
            y1=0,
            x0=0,
            x1=HEATMAP_RATIO_H_X,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=HEATMAP_RATIO_H_Y,
            y1=HEATMAP_RATIO_H_Y,
            x0=0,
            x1=HEATMAP_RATIO_H_X,
            line=dict(color="black"),
        )

        # Adjust colorbar
        fig.data[0]["colorbar"]["len"] = HEATMAP_RATIO_H_Y
        fig.data[0]["colorbar"]["y"] = HEATMAP_RATIO_H_Y

        return fig

    def visual_field_heatmap_subplots_vertical(self, fighm, fighx, fighy):
        """Combine heatmap and histograms into single plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[HEATMAP_RATIO_V_X, 1 - HEATMAP_RATIO_V_X],
            row_heights=[1 - HEATMAP_RATIO_V_Y, HEATMAP_RATIO_V_Y],
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
        )
        if len(fighm.data) > 0:
            fighmdata = fighm.data[0]
        else:
            fighmdata = None
        if len(fighx.data) > 0:
            fighxdata = fighx.data[0]
        else:
            fighxdata = None
        if len(fighy.data) > 0:
            fighydata = fighy.data[0]
        else:
            fighydata = None
        fig.add_trace(go.Heatmap(fighmdata), row=2, col=1)
        fig.add_trace(go.Histogram(fighxdata), row=2, col=2)
        fig.add_trace(go.Histogram(fighydata), row=1, col=1)
        left_margin = 40
        fig.update_layout(
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=dict(t=0, b=20, l=left_margin, r=0, autoexpand=False),
        )
        fig.update_xaxes(
            row=2,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=2,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )

        fig.update_xaxes(
            row=1,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )

        fig.update_xaxes(
            row=2,
            col=2,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )
        fig.update_yaxes(
            row=2,
            col=2,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        )

        # Draw field boundaries
        # Vertical lines
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=HEATMAP_RATIO_V_X,
            y0=0,
            y1=0,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=HEATMAP_RATIO_V_X,
            y0=HEATMAP_RATIO_V_Y / 6,
            y1=HEATMAP_RATIO_V_Y / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=HEATMAP_RATIO_V_X,
            y0=HEATMAP_RATIO_V_Y * 5 / 6,
            y1=HEATMAP_RATIO_V_Y * 5 / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=HEATMAP_RATIO_V_X,
            y0=HEATMAP_RATIO_V_Y,
            y1=HEATMAP_RATIO_V_Y,
            line=dict(color="black"),
        )

        # # Horizontal lines
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=0,
            y0=0,
            y1=HEATMAP_RATIO_V_Y,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=HEATMAP_RATIO_V_X,
            x1=HEATMAP_RATIO_V_X,
            y0=0,
            y1=HEATMAP_RATIO_V_Y,
            line=dict(color="black"),
        )

        # Adjust colorbar
        fig.data[0]["colorbar"]["len"] = HEATMAP_RATIO_V_Y
        fig.data[0]["colorbar"]["y"] = HEATMAP_RATIO_V_Y
        fig.data[0]["colorbar"]["x"] = 0

        return fig

    def _sec_to_min(self, seconds):
        """Convert seconds integer into minutes:seconds e.g. 121->2:01."""
        minutes = int(seconds / 60)
        seconds_left = seconds % 60
        return f"{minutes}:{seconds_left:02d}"

    def _bin_labels(self, bins):
        """Create bin labels for a list of bin edges that are in seconds."""
        return [
            f"{self._sec_to_min(bins[i+1])}-{self._sec_to_min(x+1)}"
            for i, x in enumerate(bins[:-1])
        ]

    def _pct_and_total(self, df):
        pct = df.groupby(level=["s_before_bucket"]).apply(lambda y: y / y.sum())
        total = df.groupby(level=["s_before_bucket"]).apply(lambda y: y - y + y.sum())
        df["pct"] = pct
        df["total"] = total
        return df

    def visual_end_of_period(
        self, team_ids=None, opposing_team_ids=None, periods=[1, 2, 3,],
    ):
        """Create a graph of score probability vs time of the point start."""
        df = (
            self.get_start_of_opoints().query(f"period=={periods}")
            # Remove games with bad timestamps
            .query(
                "~((game_id==2658) & (team_id==3)) & ~((game_id==2661) & (team_id==10))"
            )
        )

        if team_ids is not None:
            df = df.query(f"team_id=={team_ids}")
        if opposing_team_ids is not None:
            df = df.query(f"opponent_team_id=={opposing_team_ids}")

        bins = [-1, 10, 20, 30, 40, 50, 60, 120, 240, 480, 720]
        labels = self._bin_labels(bins)

        df = (
            df.assign(
                s_before_bucket=lambda x: pd.Categorical(
                    pd.cut(x["s_before"], bins=bins, labels=labels),
                    categories=reversed(labels),
                    ordered=True,
                ),
            )
            .groupby(["s_before_bucket", "possession_outcome_general"])
            .size()
            .rename("cnt")
            .to_frame()
            .pipe(self._pct_and_total)
            .reset_index()
            .query("possession_outcome_general=='Score'")
            .rename(
                columns={
                    "s_before_bucket": "Time at Start of Point",
                    "pct": "Score Pct",
                    "cnt": "Num of Scores",
                    "total": "Num of Points",
                }
            )
        )

        fig = px.bar(
            df,
            x="Time at Start of Point",
            y="Score Pct",
            hover_data=["Num of Scores", "Num of Points"],
        )

        fig.update_layout(
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            # margin=dict(t=0, b=20, l=left_margin, r=0, autoexpand=False),
            yaxis_tickformat=".0%",
            # Add title
            title={
                "text": "Probability of Scoring on O-Point w/ No Turnovers",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
        )

        return fig

    def get_game_qc(self):
        """Get QC results for each game.

        Segments in each game without 7 players.
        Events in raw data without 7 players.
        Segments in each game with a negative elapsed number of seconds.
        Events in raw data with negative time stamps.
        Event types without existing labels.
        Raw data labels in each game (t, l, x, y, etc.).

        """
        # 2021-07-02 MAD CHI has order messed up for a point - d-line completions come before pull and block events
        # 2021-06-18 LA DAL has messed up timestamps
        if self.game_qc is None:
            # More or less than 7 players in the processed data
            not_seven_players_home = []
            not_seven_players_away = []

            # More or less than 7 players in the raw data
            not_seven_players_home_raw = []
            not_seven_players_away_raw = []

            # Negative elapsed durations for point
            negative_elapsed_home = []
            negative_elapsed_away = []

            # Negative timestamps for event in raw data
            negative_time_home_raw = []
            negative_time_away_raw = []

            # Types of events that we do not have labels for
            no_event_label_home = []
            no_event_label_away = []

            # Data labels in each game, such as t, x, y, etc.
            data_labels_home_raw = []
            data_labels_away_raw = []

            # Get all games that have events (they've actually happened)
            existing_games = (
                self.get_game_info(override=False).query("events_exist==True").copy()
            )

            # Process each game to get player segments
            for i, row in existing_games.iterrows():
                g = Game(row["url"])
                times_home, psegs_home = g.get_player_segments(
                    roster=g.get_home_roster(), events=g.get_home_events()
                )
                times_away, psegs_away = g.get_player_segments(
                    roster=g.get_away_roster(), events=g.get_away_events()
                )

                # Record the number of segments without 7 players in processed data for each team
                not_seven_players_home.append((psegs_home.sum() != 7).sum())
                not_seven_players_away.append((psegs_away.sum() != 7).sum())

                # Record the number of segments without 7 players in raw data for each team
                not_seven_players_home_raw.append(
                    sum([len(x["l"]) != 7 for x in g.get_home_events_raw() if "l" in x])
                )
                not_seven_players_away_raw.append(
                    sum([len(x["l"]) != 7 for x in g.get_away_events_raw() if "l" in x])
                )

                # Record the number of segments with a negative elapsed time
                negative_elapsed_home.append(
                    (
                        times_home.groupby(["s_before_total", "s_after_total"])[
                            "elapsed"
                        ].min()
                        < 0
                    ).sum()
                )
                negative_elapsed_away.append(
                    (
                        times_away.groupby(["s_before_total", "s_after_total"])[
                            "elapsed"
                        ].min()
                        < 0
                    ).sum()
                )

                # Record the number of events with a negative timestamp in raw data
                negative_time_home_raw.append(
                    sum([x["s"] < 0 for x in g.get_home_events_raw() if "s" in x])
                )
                negative_time_away_raw.append(
                    sum([x["s"] < 0 for x in g.get_away_events_raw() if "s" in x])
                )

                # Record the events that we do not have labels for
                no_event_label_home.append(
                    g.get_home_events()
                    .query("event_name!=event_name")["t"]
                    .unique()
                    .tolist()
                )
                no_event_label_away.append(
                    g.get_away_events()
                    .query("event_name!=event_name")["t"]
                    .unique()
                    .tolist()
                )

                # Record the data labels in each game
                event_types = []
                for x in g.get_home_events_raw():
                    event_types.extend(x.keys())
                data_labels_home_raw.append(sorted(list(set(event_types))))
                event_types = []
                for x in g.get_away_events_raw():
                    event_types.extend(x.keys())
                data_labels_away_raw.append(sorted(list(set(event_types))))

            # Add results to the games dataframe
            existing_games["not_seven_players_home"] = not_seven_players_home
            existing_games["not_seven_players_away"] = not_seven_players_away
            existing_games["not_seven_players_home_raw"] = not_seven_players_home_raw
            existing_games["not_seven_players_away_raw"] = not_seven_players_away_raw
            existing_games["negative_elapsed_home"] = negative_elapsed_home
            existing_games["negative_elapsed_away"] = negative_elapsed_away
            existing_games["negative_time_home_raw"] = negative_time_home_raw
            existing_games["negative_time_away_raw"] = negative_time_away_raw
            existing_games["no_event_label_home"] = no_event_label_home
            existing_games["no_event_label_away"] = no_event_label_away
            existing_games["data_labels_home_raw"] = data_labels_home_raw
            existing_games["data_labels_away_raw"] = data_labels_away_raw

            self.game_qc = existing_games

        return self.game_qc
