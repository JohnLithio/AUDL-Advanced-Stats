import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from ast import literal_eval
from bs4 import BeautifulSoup
from json import loads
from os.path import basename, join
from pathlib import Path
from sklearn.cluster import KMeans, Birch
from .constants import *
from .utils import get_data_path, get_json_path, upload_to_bucket, download_from_bucket


def get_player_rates(df):
    dfout = df.assign(
        throwaways_pp=lambda x: x["throwaways"] / x["o_possessions"],
        drops_pp=lambda x: x["drops"] / x["o_possessions"],
        stalls_pp=lambda x: x["stalls"] / x["o_possessions"],
        completions_pp=lambda x: x["completions"] / x["o_possessions"],
        receptions_pp=lambda x: x["receptions"] / x["o_possessions"],
        turnovers_pp=lambda x: x["turnovers"] / x["o_possessions"],
        assists_pp=lambda x: x["assists"] / x["o_possessions"],
        goals_pp=lambda x: x["goals"] / x["o_possessions"],
        blocks_pp=lambda x: x["blocks"] / x["d_possessions"],
        yx_ratio_throwing=lambda x: x["yyards_throwing_total"]
        / x["xyards_throwing_total"],
        yx_ratio_receiving=lambda x: x["yyards_receiving_total"]
        / x["xyards_receiving_total"],
        xyards_throwing_pp=lambda x: x["xyards_throwing_total"] / x["o_possessions"],
        yyards_throwing_pp=lambda x: x["yyards_throwing_total"] / x["o_possessions"],
        yards_throwing_pp=lambda x: x["yards_throwing_total"] / x["o_possessions"],
        yyards_raw_throwing_pp=lambda x: x["yyards_raw_throwing_total"]
        / x["o_possessions"],
        yards_raw_throwing_pp=lambda x: x["yards_raw_throwing_total"]
        / x["o_possessions"],
        xyards_receiving_pp=lambda x: x["xyards_receiving_total"] / x["o_possessions"],
        yyards_receiving_pp=lambda x: x["yyards_receiving_total"] / x["o_possessions"],
        yards_receiving_pp=lambda x: x["yards_receiving_total"] / x["o_possessions"],
        yyards_raw_receiving_pp=lambda x: x["yyards_raw_receiving_total"]
        / x["o_possessions"],
        yards_raw_receiving_pp=lambda x: x["yards_raw_receiving_total"]
        / x["o_possessions"],
        xyards_pp=lambda x: x["xyards_total"] / x["o_possessions"],
        yyards_pp=lambda x: x["yyards_total"] / x["o_possessions"],
        yards_pp=lambda x: x["yards_total"] / x["o_possessions"],
        yyards_raw_pp=lambda x: x["yyards_raw_total"] / x["o_possessions"],
        yards_raw_pp=lambda x: x["yards_raw_total"] / x["o_possessions"],
        xyards_throwing_percompletion=lambda x: x["xyards_throwing_total"]
        / x["completions"],
        yyards_throwing_percompletion=lambda x: x["yyards_throwing_total"]
        / x["completions"],
        yards_throwing_percompletion=lambda x: x["yards_throwing_total"]
        / x["completions"],
        yyards_raw_throwing_percompletion=lambda x: x["yyards_raw_throwing_total"]
        / x["completions"],
        yards_raw_throwing_percompletion=lambda x: x["yards_raw_throwing_total"]
        / x["completions"],
        xyards_receiving_perreception=lambda x: x["xyards_receiving_total"]
        / x["receptions"],
        yyards_receiving_perreception=lambda x: x["yyards_receiving_total"]
        / x["receptions"],
        yards_receiving_perreception=lambda x: x["yards_receiving_total"]
        / x["receptions"],
        yyards_raw_receiving_perreception=lambda x: x["yyards_raw_receiving_total"]
        / x["receptions"],
        yards_raw_receiving_perreception=lambda x: x["yards_raw_receiving_total"]
        / x["receptions"],
        xyards_throwing_perthrowaway=lambda x: x["xyards_throwaway"] / x["throwaways"],
        yyards_throwing_perthrowaway=lambda x: x["yyards_throwaway"] / x["throwaways"],
        yards_throwing_perthrowaway=lambda x: x["yards_throwaway"] / x["throwaways"],
        yyards_raw_throwing_perthrowaway=lambda x: x["yyards_raw_throwaway"]
        / x["throwaways"],
        yards_raw_throwing_perthrowaway=lambda x: x["yards_raw_throwaway"]
        / x["throwaways"],
        o_point_score_pct=lambda x: x["o_point_scores"] / x["o_points"],
        d_point_score_pct=lambda x: x["d_point_scores"] / x["d_points"],
        o_point_noturn_pct=lambda x: x["o_point_noturns"] / x["o_points"],
        d_point_turn_pct=lambda x: x["d_point_turns"] / x["d_points"],
        xyards_total_pct=lambda x: x["xyards_total"] / x["xyards_team"],
        yyards_total_pct=lambda x: x["yyards_total"] / x["yyards_team"],
        yards_total_pct=lambda x: x["yards_total"] / x["yards_team"],
        yyards_raw_total_pct=lambda x: x["yyards_raw_total"] / x["yyards_raw_team"],
        yards_raw_total_pct=lambda x: x["yards_raw_total"] / x["yards_raw_team"],
        xyards_throwing_total_pct=lambda x: x["xyards_throwing_total"]
        / x["xyards_team"],
        yyards_throwing_total_pct=lambda x: x["yyards_throwing_total"]
        / x["yyards_team"],
        yards_throwing_total_pct=lambda x: x["yards_throwing_total"] / x["yards_team"],
        yyards_raw_throwing_total_pct=lambda x: x["yyards_raw_throwing_total"]
        / x["yyards_raw_team"],
        yards_raw_throwing_total_pct=lambda x: x["yards_raw_throwing_total"]
        / x["yards_raw_team"],
        xyards_receiving_total_pct=lambda x: x["xyards_receiving_total"]
        / x["xyards_team"],
        yyards_receiving_total_pct=lambda x: x["yyards_receiving_total"]
        / x["yyards_team"],
        yards_receiving_total_pct=lambda x: x["yards_receiving_total"]
        / x["yards_team"],
        yyards_raw_receiving_total_pct=lambda x: x["yyards_raw_receiving_total"]
        / x["yyards_raw_team"],
        yards_raw_receiving_total_pct=lambda x: x["yards_raw_receiving_total"]
        / x["yards_raw_team"],
        completion_pct=lambda x: x["completions"] / x["throw_attempts"],
        reception_pct=lambda x: x["receptions"] / x["catch_attempts"],
        assists_perthrowattempt=lambda x: x["assists"] / x["throw_attempts"],
        goals_percatchattempt=lambda x: x["goals"] / x["catch_attempts"],
        points_team_pct=lambda x: x["total_points"] / x["points_team"],
        o_points_team_pct=lambda x: x["o_points"] / x["o_points_team"],
        d_points_team_pct=lambda x: x["d_points"] / x["d_points_team"],
        possessions_team_pct=lambda x: x["total_possessions"] / x["possessions_team"],
        o_possessions_team_pct=lambda x: x["o_possessions"] / x["o_possessions_team"],
        d_possessions_team_pct=lambda x: x["d_possessions"] / x["d_possessions_team"],
        throwaways_team_pct=lambda x: x["throwaways"] / x["throwaways_team"],
        drops_team_pct=lambda x: x["drops"] / x["drops_team"],
        receptions_team_pct=lambda x: x["receptions"] / x["completions_team"],
        completions_team_pct=lambda x: x["completions"] / x["completions_team"],
        stalls_team_pct=lambda x: x["stalls"] / x["stalls_team"],
        assists_team_pct=lambda x: x["assists"] / x["goals_team"],
        goals_team_pct=lambda x: x["goals"] / x["goals_team"],
        blocks_team_pct=lambda x: x["blocks"] / x["blocks_team"],
        callahans_team_pct=lambda x: x["callahans"] / x["callahans_team"],
        o_possession_score_pct=lambda x: x["o_possession_scores"] / x["o_possessions"],
        d_possession_score_allowed_pct=lambda x: x["d_possession_scores_allowed"]
        / x["d_possessions"],
    )

    throw_types = [
        x[9:] for x in list(dfout) if ("attempts_" in x) and ("_pct" not in x)
    ]
    for throw_type in throw_types:
        dfout[f"completion_{throw_type}_pct"] = (
            dfout[f"completions_{throw_type}"] / dfout[f"attempts_{throw_type}"]
        )

        dfout[f"attempts_{throw_type}_pct"] = dfout[
            f"attempts_{throw_type}"
        ] / pd.concat([dfout[f"attempts_{tt}"] for tt in throw_types], axis=1).sum(
            axis=1
        )

        dfout[f"receptions_{throw_type}_pct"] = dfout[
            f"receptions_{throw_type}"
        ] / pd.concat([dfout[f"receptions_{tt}"] for tt in throw_types], axis=1).sum(
            axis=1
        )

    return dfout


def round_player_stats(df):
    # Remove decimal points
    numeric_cols = [col for col in df if df.dtypes[col] == "float64"]
    for col in numeric_cols:
        if (
            ("_percompletion" in col)
            or ("_perthrowaway" in col)
            or ("_perthrowattempt" in col)
            or ("_percatchattempt" in col)
            or ("_pp" in col)
            or ("_ratio_" in col)
        ) and ("yards" not in col):
            df[col] = df[col].round(2)
        elif "pct" in col:
            df[col] = (df[col] * 100).round(0)
        else:
            df[col] = df[col].round(0)

    return df


class Game:
    def __init__(
        self,
        game_url,
        year=CURRENT_YEAR,
        data_path="data",
        upload=False,
        download=False,
    ):
        """Initial parameters of game data.

        Args:
            game_url (str): URL to get game response data.
            year (int, optional): Season to get stats from. Currently not used because there are only
                advanced stats for a single season (2021).
                Defaults to CURRENT_YEAR.
            data_path (str, optional): The path to the folder where data
                will be stored.

        """
        # Inputs
        self.upload = upload
        self.download = download
        self.year = year
        self.game_url = game_url
        self.data_path = get_data_path(data_path)
        self.json_path = get_json_path(self.data_path, "games_raw")
        self.events_path = get_json_path(self.data_path, "games_processed")

        # Create directories if they don't exist
        Path(self.json_path).mkdir(parents=True, exist_ok=True)
        Path(self.events_path).mkdir(parents=True, exist_ok=True)

        # Game info
        self.response = None
        self.game_info = None
        self.home_team = None
        self.home_roster = None
        self.home_events_raw = None
        self.home_events = None
        self.away_team = None
        self.away_roster = None
        self.away_events_raw = None
        self.away_events = None

    def get_game_name(self):
        """Get the game name (no file extension) based on the URL."""
        self.game_name = basename(self.game_url)
        return self.game_name

    def get_game_file_name(self):
        """Get the game name (with file extension) based on the URL."""
        self.game_file_name = self.get_game_name() + ".json"
        return self.game_file_name

    def get_response(self, upload=None, download=None):
        """Get the response of a single game and save it.

        Returns:
            str: The json response string.

        """
        if self.response is None:
            if upload is None:
                upload = self.upload
            if download is None:
                download = self.download

            json_path = join(self.json_path, self.get_game_file_name())
            # If file doesn't exist locally, try to retrieve it from AWS
            if not Path(json_path).is_file() and download:
                download_from_bucket(json_path)

            # If events have already been processed and saved, load them
            if Path(json_path).is_file():
                with open(json_path, "r") as f:
                    response_text = f.read()
            else:
                response_text = requests.get(self.game_url).text
                with open(json_path, "w") as f:
                    f.write(response_text)
                if upload:
                    upload_to_bucket(json_path)

            # Parse the json response
            response = loads(response_text)

            self.response = response

        return self.response

    def get_game_info(self):
        """Get dataframe of basic game info."""
        if self.game_info is None:
            df = pd.DataFrame.from_records([self.get_response()["game"]]).drop(
                columns=[
                    "score_times_home",
                    "score_times_away",
                    "team_season_home",
                    "team_season_away",
                ]
            )
            self.game_info = df

        return self.game_info

    def get_events_raw(self, home=True):
        """Get the response of events for one team in a single game.

        Args:
            home (bool, optional): If True get the home team events, otherwise get
                the away team events.

        Returns:
            df: All the events for that team

        """
        # Set parameters for home or away
        if home:
            events_str = "tsgHome"
        else:
            events_str = "tsgAway"

        assert (
            self.get_response()[events_str] is not None
        ), "Events not available for this game."

        # Get the events
        events = literal_eval(
            self.get_response()[events_str]["events"]
            .replace("true", "True")
            .replace("false", "False")
        )

        return events

    def get_home_events_raw(self):
        """Get a list of the home team's events in the game.

        Returns:
            list: list of dictionaries containing events
        """
        if self.home_events_raw is None:
            self.home_events_raw = self.get_events_raw(home=True)
        return self.home_events_raw

    def get_away_events_raw(self):
        """Get a list of the away team's events in the game.

        Returns:
            list: list of dictionaries containing events
        """
        if self.away_events_raw is None:
            self.away_events_raw = self.get_events_raw(home=False)
        return self.away_events_raw

    def get_team(self, home=True):
        """Get dataframe of basic info about the home or away team."""
        # Set parameters for home or away
        if home:
            team_str = "home"
        else:
            team_str = "away"

        team_raw = self.get_response()["game"][f"team_season_{team_str}"]

        # Convert to dataframe and un-nest dicts
        team = (
            pd.DataFrame.from_records(team_raw, index=[0])
            .merge(
                pd.DataFrame.from_records([team_raw["team"]]),
                how="left",
                left_on=["team_id"],
                right_on=["id"],
                suffixes=["", "_y"],
            )
            .drop(columns=["id_y", "team"])
        )

        return team

    def get_home_team(self):
        """Get dict of basic info about the home team."""
        if self.home_team is None:
            self.home_team = self.get_team(home=True)
        return self.home_team

    def get_away_team(self):
        """Get dict of basic info about the away team."""
        if self.away_team is None:
            self.away_team = self.get_team(home=False)
        return self.away_team

    def get_roster(self, home=True):
        """Get dataframe of all players on roster for the game."""
        # Set parameters for home or away
        if home:
            roster_str = "Home"
        else:
            roster_str = "Away"

        # Get list of dicts of roster
        roster_raw = self.get_response()[f"rosters{roster_str}"]

        # Convert to dataframe and un-nest dicts
        roster = (
            pd.DataFrame.from_records(roster_raw)
            .merge(
                pd.DataFrame.from_records([x["player"] for x in roster_raw]),
                how="left",
                left_on=["player_id"],
                right_on=["id"],
                suffixes=["", "_y"],
            )
            .drop(columns=["id_y", "player", "active"])
            .drop_duplicates()
        )

        # Get active players
        active_raw = self.response[f"tsg{roster_str}"]["rosterIds"]
        active = pd.DataFrame(data=[[x] for x in active_raw], columns=["id"])
        active["active"] = True

        # Add column for whether players were active for this game
        roster = roster.merge(active, how="left", on=["id"]).assign(
            active=lambda x: x["active"].fillna(False)
        )

        return roster

    def get_home_roster(self):
        """Get dataframe of all players on home roster for the game."""
        if self.home_roster is None:
            self.home_roster = self.get_roster(home=True)
        return self.home_roster

    def get_away_roster(self):
        """Get dataframe of all players on away roster for the game."""
        if self.away_roster is None:
            self.away_roster = self.get_roster(home=False)
        return self.away_roster

    def add_fourth_period(self, df):
        """If necessary, add a row to represent the end of the fourth period."""
        # Some games do not have the event for the end of the 4th period, so we have to add it manually
        if 26 not in df["t"].unique():
            fourth_period_row = pd.DataFrame(
                data=[[None for _ in list(df)]], columns=list(df)
            )
            fourth_period_row["t"] = 26
            df = df.append(fourth_period_row)
        return df

    def get_events_basic_info(self, df, home):
        # Set parameters for home or away
        if home:
            team = self.get_home_team()
            opponent_team = self.get_away_team()
        else:
            team = self.get_away_team()
            opponent_team = self.get_home_team()

        df = df.assign(
            # Basic game info
            game_id=self.get_game_info()["id"].iloc[0],
            team_id=team["team_id"].iloc[0],
            opponent_team_id=opponent_team["team_id"].iloc[0],
            # Set ID for each event incrementally
            event_number=lambda x: x.index,
            # Convert events to human-readable labels
            event_name=lambda x: x["t"].map(EVENT_TYPES),
        )
        return df

    def get_events_possession_labels(self, df):
        df = (
            df.assign(
                # Classify each point as O or D point
                o_point=lambda x: np.where(x["t"].isin([1]), True, None),
                # Sometimes the event for the start of a d-point is missing. One game had two pulls with a foul in-between.
                d_point=lambda x: np.where(
                    x["t"].isin([2])
                    | ((x["t"].isin([3, 4])) & ~(x["t"].shift(1).isin([2, 13]))),
                    True,
                    None,
                ),
                # Mark the first event of each point
                point_start=lambda x: (x["o_point"] == True) | (x["d_point"] == True),
                # Count the number of points played and label each one
                point_number=lambda x: np.where(
                    x["point_start"], x.groupby(["point_start"]).cumcount() + 2, None
                ),
                # Mark the first event of each possession
                # These occur after turns, scores, and the end of periods
                # Sometimes both a throwaway and drop are recorded, so we ignore one of these
                possession_change=lambda x: x["t"]
                .shift(1)
                .isin([5, 6, 7, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
                & ~(x["t"].isin([8, 19]) & x["t"].shift(1).isin([8, 19])),
                # Label each possession incrementally
                possession_number=lambda x: np.where(
                    x["possession_change"],
                    x.groupby(["possession_change"]).cumcount() + 2,
                    None,
                ),
                # Set the outcome of each point and possession
                point_outcome=lambda x: np.where(
                    x["t"].isin([6, 7, 21, 22]), x["t"].map(EVENT_TYPES), None
                ),
                possession_outcome=lambda x: np.where(
                    x["t"].isin([5, 6, 7, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
                    & ~(x["t"].isin([8, 19]) & x["t"].shift(1).isin([8, 19])),
                    x["t"].map(EVENT_TYPES),
                    None,
                ),
                possession_outcome_general=lambda x: np.where(
                    x["t"].isin([5, 6, 7, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
                    & ~(x["t"].isin([8, 19]) & x["t"].shift(1).isin([8, 19])),
                    x["t"].map(EVENT_TYPES_GENERAL),
                    None,
                ),
            )
            .assign(
                # Set the outcome of each point and possession
                point_outcome=lambda x: np.where(
                    x["t"].isin([23, 24, 25, 26, 27]),
                    "End of Period",
                    x["point_outcome"],
                ),
                # Fill in the point number for every event
                point_number=lambda x: x["point_number"]
                .fillna(method="ffill")
                .fillna(1),
                # Fill in the possession number for every event
                possession_number=lambda x: x["possession_number"]
                .fillna(method="ffill")
                .fillna(1),
                # Set o_point to True for start of o point, False for start of d point,
                #     and null for all other events
                o_point=lambda x: np.where(x["d_point"], False, x["o_point"]),
                # If the event for the start of a d-point is missing, we need to guess at the lineup info
                l=lambda x: np.where(
                    x["d_point"] & (x["l"].isna()),
                    x["l"].fillna(method="bfill"),
                    x["l"],
                ),
            )
            .assign(
                # Fill in o_point for every event
                o_point=lambda x: x["o_point"].fillna(method="ffill"),
                # Fill in the outcome of every point and possession for every event
                point_outcome=lambda x: x["point_outcome"].fillna(method="bfill"),
                possession_outcome=lambda x: x["possession_outcome"].fillna(
                    method="bfill"
                ),
                possession_outcome_general=lambda x: x[
                    "possession_outcome_general"
                ].fillna(method="bfill"),
                point_number=lambda x: np.where(
                    x["t"].isin([23, 24, 25, 26, 27]),
                    x["point_number"].shift(1),
                    x["point_number"],
                ),
                possession_number=lambda x: np.where(
                    x["t"].isin([23, 24, 25, 26, 27]),
                    x["possession_number"].shift(1),
                    x["possession_number"],
                ),
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (
                        (x["point_outcome"] == EVENT_TYPES[22])
                        | (x["point_outcome"] == EVENT_TYPES[6])
                    )
                    & (x["o_point"]),
                    "Hold",
                    "End of Period",
                )
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (
                        (x["point_outcome"] == EVENT_TYPES[21])
                        | (x["point_outcome"] == EVENT_TYPES[7])
                    )
                    & (~x["o_point"]),
                    "Opponent Hold",
                    x["point_hold"],
                )
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (
                        (x["point_outcome"] == EVENT_TYPES[22])
                        | (x["point_outcome"] == EVENT_TYPES[6])
                    )
                    & (~x["o_point"]),
                    "Break",
                    x["point_hold"],
                )
            )
            .assign(
                point_hold=lambda x: np.where(
                    (
                        (x["point_outcome"] == EVENT_TYPES[21])
                        | (x["point_outcome"] == EVENT_TYPES[7])
                    )
                    & (x["o_point"]),
                    "Opponent Break",
                    x["point_hold"],
                )
            )
            .assign(point_hold=lambda x: x["point_hold"].fillna(method="bfill"))
            # Create a column for each player to indicate if they were on or off the field
            .drop(columns=["point_start", "possession_change", "d_point"])
        )

        # Identify which team was on offense for each possession
        df = (
            df.merge(
                # Get the first possession ID for every point
                df.groupby(["point_number"])["possession_number"]
                .min()
                .reset_index()[["point_number", "possession_number"]],
                how="left",
                on=["point_number"],
                suffixes=["", "_min"],
            )
            .merge(
                # Get the last possession ID for every point
                df.groupby(["point_number"])["possession_number"]
                .max()
                .reset_index()[["point_number", "possession_number"]],
                how="left",
                on=["point_number"],
                suffixes=["", "_max"],
            )
            .assign(
                # Offensive possession if the possession ID is the same as the
                #     initial possession ID for that point, or if it's 2 higher,
                #     4 higher, 6 higher, etc.
                offensive_possession=lambda x: np.where(
                    x["o_point"],
                    (x["possession_number_min"] - x["possession_number"]) % 2 == 0,
                    (x["possession_number_min"] - x["possession_number"]) % 2 != 0,
                ),
                offensive_team=lambda x: np.where(
                    x["offensive_possession"], x["team_id"], x["opponent_team_id"]
                ),
                num_turnovers=lambda x: (
                    x["possession_number_max"] - x["possession_number_min"]
                ).astype(int),
            )
        )

        return df

    def get_events_pull_info(self, df):
        df = df.assign(
            # Convert pull hangtime from milliseconds to seconds
            hangtime=lambda x: x["ms"]
            / 1000
        ).drop(columns=["ms"])
        return df

    def get_events_stalls(self, df):
        # Set x and y positions for stalls
        stalls = (
            df.query("t==[17, 20]")
            .groupby(["possession_number"])[["r", "x", "y"]]
            .shift(1)
            .rename(columns=lambda x: x + "_new")
        )
        stalls = stalls.loc[stalls.index.isin(df.query("t==[17]").index)]
        df = (
            pd.concat([df, stalls], axis=1)
            .assign(
                x=lambda x: x["x"].fillna(x["x_new"]),
                y=lambda x: x["y"].fillna(x["y_new"]),
                r=lambda x: x["r"].fillna(x["r_new"]),
            )
            .assign(
                x=lambda x: np.where(
                    x["t"].isin([17, 20]),
                    x.groupby(["possession_number"])["x"].fillna(method="ffill"),
                    x["x"],
                ),
                y=lambda x: np.where(
                    x["t"].isin([17, 20]),
                    x.groupby(["possession_number"])["y"].fillna(method="ffill"),
                    x["y"],
                ),
                r=lambda x: np.where(
                    x["t"].isin([17, 20]),
                    x.groupby(["possession_number"])["r"].fillna(method="ffill"),
                    x["r"],
                ),
            )
            .drop(columns=["x_new", "y_new", "r_new"])
        )
        return df

    def get_events_o_penalties(self, df):
        # Adjust x and y positions for penalties against the offense
        o_penalties = (
            df.query("t==[10,20]")
            .groupby(["possession_number"])[["r", "x", "y"]]
            .shift(1)
            # Minimum of 0 yards in y direction, which is the back of the endzone (range is 0 to 120)
            .assign(y=lambda x: (x["y"] - 10).clip(lower=0))
            .rename(columns=lambda x: x + "_new")
        )
        o_penalties = o_penalties.loc[o_penalties.index.isin(df.query("t==[10]").index)]
        df = (
            pd.concat([df, o_penalties], axis=1)
            .assign(
                x=lambda x: x["x"].fillna(x["x_new"]),
                y=lambda x: x["y"].fillna(x["y_new"]),
                r=lambda x: x["r"].fillna(x["r_new"]),
            )
            .assign(
                x=lambda x: np.where(
                    x["t"].isin([10, 20]),
                    x.groupby(["possession_number"])["x"].fillna(method="ffill"),
                    x["x"],
                ),
                y=lambda x: np.where(
                    x["t"].isin([10, 20]),
                    x.groupby(["possession_number"])["y"].fillna(method="ffill"),
                    x["y"],
                ),
                r=lambda x: np.where(
                    x["t"].isin([10, 20]),
                    x.groupby(["possession_number"])["r"].fillna(method="ffill"),
                    x["r"],
                ),
            )
            .drop(columns=["x_new", "y_new", "r_new"])
        )
        return df

    def get_events_d_penalties(self, df):
        # Adjust x and y positions for penalties against the defense
        d_penalties = (
            df.query("t==[12,20]")
            .groupby(["possession_number"])[["r", "x", "y"]]
            .shift(1)
            .assign(
                # Maximum of 100 yards in y direction, which is the goal-line (range is 0 to 120)
                y=lambda x: (x["y"] + 10).clip(upper=100),
            )
        )
        # If penalty is within 10 yards of the endzone, center it on the goalline
        # This is indicated in the data by the c variable
        if "c" in list(d_penalties):
            d_penalties["x"] = np.where(d_penalties["c"], 0, d_penalties["x"])

        d_penalties = d_penalties.loc[d_penalties.index.isin(df.query("t==[12]").index)]
        df = (
            pd.concat([df, d_penalties.rename(columns=lambda x: x + "_new")], axis=1)
            .assign(
                x=lambda x: x["x"].fillna(x["x_new"]),
                y=lambda x: x["y"].fillna(x["y_new"]),
                r=lambda x: x["r"].fillna(x["r_new"]),
            )
            .assign(
                x=lambda x: np.where(
                    x["t"].isin([12, 20]),
                    x.groupby(["possession_number"])["x"].fillna(method="ffill"),
                    x["x"],
                ),
                y=lambda x: np.where(
                    x["t"].isin([12, 20]),
                    x.groupby(["possession_number"])["y"].fillna(method="ffill"),
                    x["y"],
                ),
                r=lambda x: np.where(
                    x["t"].isin([12, 20]),
                    x.groupby(["possession_number"])["r"].fillna(method="ffill"),
                    x["r"],
                ),
            )
            .drop(columns=["x_new", "y_new", "r_new"])
        )
        return df

    def fill_missing_info(self, df):
        return df.assign(
            x=lambda x: np.where(
                x["t"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                x.groupby(["possession_number"])["x"].fillna(method="ffill"),
                x["x"],
            ),
            y=lambda x: np.where(
                x["t"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                x.groupby(["possession_number"])["y"].fillna(method="ffill"),
                x["y"],
            ),
            r=lambda x: np.where(
                x["t"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                x.groupby(["possession_number"])["r"].fillna(method="ffill"),
                x["r"],
            ),
        )

    def get_events_yardage(self, df):
        # Get the x,y position for the next throwaway, travel, defensive foul, drop, completion, or score
        next_event = (
            df.query("t==[7,8,10,12,17,19,20,22]")
            .groupby(["possession_number"])[["r", "x", "y", "t", "event_name"]]
            .shift(-1)
            .rename(columns=lambda x: x + "_after")
        )
        # Add columns for the next event to each row so we can calculate yardage
        df = (
            pd.concat([df, next_event], axis=1)
            .assign(
                x=lambda x: np.where(
                    x["t_after"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                    x.groupby(["possession_number"])["x"].fillna(method="ffill"),
                    x["x"],
                ),
                y=lambda x: np.where(
                    x["t_after"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                    x.groupby(["possession_number"])["y"].fillna(method="ffill"),
                    x["y"],
                ),
                r=lambda x: np.where(
                    x["t_after"].isin([7, 8, 10, 12, 17, 19, 20, 22]),
                    x.groupby(["possession_number"])["r"].fillna(method="ffill"),
                    x["r"],
                ),
            )
            .assign(
                # Set x and y values for pulls to align with the format for other events
                x_after=lambda x: np.where(x["t"].isin([3, 4]), x["x"], x["x_after"]),
                y_after=lambda x: np.where(x["t"].isin([3, 4]), x["y"], x["y_after"]),
                x=lambda x: np.where(x["t"].isin([3, 4]), 0, x["x"]),
                y=lambda x: np.where(x["t"].isin([3, 4]), 20, x["y"]),
                # Calculate yardage in x direction
                xyards_raw=lambda x: x["x_after"] - x["x"],
                # Calculate absolute value of yardage in x direction
                xyards=lambda x: np.abs(x["xyards_raw"]),
                # Calculate yards in y direction, including yards in the endzone
                yyards_raw=lambda x: x["y_after"] - x["y"],
                # Calculate yards in y direction, excluding yards in the endzone
                yyards=lambda x: x["y_after"].clip(upper=100) - x["y"],
                # Calculate the total distance of the throw, including yards in the endzone
                yards_raw=lambda x: (x["xyards"].pow(2) + x["yyards_raw"].pow(2)).pow(
                    0.5
                ),
                # Calculate the total distance of the throw, excluding yards in the endzone
                yards=lambda x: (x["xyards"].pow(2) + x["yyards"].pow(2)).pow(0.5),
                throw_outcome=lambda x: np.where(
                    x["t_after"].isin([7, 8, 17, 19]), "Turnover", None
                ),
            )
            .assign(
                throw_outcome=lambda x: np.where(
                    x["t_after"].isin([20, 22]), "Completion", x["throw_outcome"]
                )
            )
        )
        return df

    def get_events_periods(self, df):
        df = df.assign(
            period=lambda x: np.where(
                x["t"].isin([23, 24, 25, 26, 27, 28]), x["t"] - 22, None
            )
        ).assign(period=lambda x: x["period"].fillna(method="bfill"))
        return df

    def get_events_times(self, df):
        # Set the times for the first and last event of each period
        df = (
            df.assign(
                # First event of each quarter
                s=lambda x: np.where(
                    x.index.isin(x.groupby(["period"]).head(1).index)
                    & x["period"].isin([1, 2, 3, 4]),
                    12 * 60,
                    x["s"],
                )
            )
            .assign(
                # First event of overtime
                s=lambda x: np.where(
                    x.index.isin(x.groupby(["period"]).head(1).index)
                    & x["period"].isin([5]),
                    5 * 60,
                    x["s"],
                )
            )
            .assign(
                # Last event of each period
                s=lambda x: np.where(
                    x.index.isin(x.groupby(["period"]).tail(1).index), 0, x["s"]
                )
            )
            .assign(
                # Injuries do not have a timestamp, so we need to estimate a timestamp for them so the subs don't cause >7 players
                #   Here we just take the midpoint of the closest timestamps before and after the injury
                s=lambda x: np.where(
                    x["t"].isin([42, 43]),
                    (x["s"].fillna(method="bfill") + x["s"].fillna(method="ffill")) / 2,
                    x["s"],
                )
            )
            .assign(
                # Total time elapsed from start of game. 0=start of 1st quarter
                s_total=lambda x: np.where(
                    x["period"].isin([1, 2, 3, 4]),
                    -x["s"] + (x["period"]) * 60 * 12,
                    -x["s"] + 4 * 60 * 12 + (x["period"] - 4) * 60 * 5,
                )
            )
            .assign(
                # Some time stamps are negative, which seem to be the time elapsed from the start of the period
                s_total=lambda x: np.where(
                    x["s"] < 0, -x["s"] + (x["period"] - 1) * 60 * 12, x["s_total"]
                )
            )
            .assign(
                # Get the most recent time stamp before each event
                s_before=lambda x: x["s"].fillna(method="ffill"),
                # Get the closest time stamp following each event
                s_after=lambda x: x["s"].fillna(method="bfill"),
                # Get the most recent time stamp before each event
                s_before_total=lambda x: x["s_total"].fillna(method="ffill"),
                # Get the closest time stamp following each event
                s_after_total=lambda x: x["s_total"].fillna(method="bfill"),
                # Calculate the time elapsed
                elapsed=lambda x: x["s_before"] - x["s_after"],
            )
        )

        return df

    def get_player_columns(self, df):
        s = df["l"].explode()
        return df.join(pd.crosstab(s.index, s))

    def get_events_lineups(self, df):
        df = (
            df.assign(
                # Fill-in lineup info for every event
                l=lambda x: x["l"].fillna(method="ffill")
            )
            # Create a column for each player to indicate if they were on or off the field
            .pipe(self.get_player_columns).drop(columns=["l"])
        )
        return df

    def get_events_throw_classifications(self, df):
        # TODO: Try clustering analysis for these
        df["centering_pass"] = (
            (df["t_after"].isin([20]))  # Completed pass
            & (df["t"].shift(1).isin([1]))  # Previous event was the start of an o-point
            & ~((df["x"] == 0) & (df["y"] == 20))  # Not starting from the brick mark
            & (
                df["x_after"].abs() <= df["x"].abs()
            )  # Disc is moved closer to center of the field
            & (df["y_after"] > df["y"])  # Disc is moved forward
        )

        # See throw classification diagram
        df["throw_type"] = None

        # Unders/Others
        df["throw_type"] = np.where(
            (df["t_after"].isin([8, 19, 20, 22]))
            & (df["yyards_raw"] >= 5)
            & (df["yyards_raw"] < 45),
            "Throw",
            df["throw_type"],
        )

        # Swings
        df["throw_type"] = np.where(
            (df["t_after"].isin([8, 19, 20, 22]))
            & (np.degrees(np.arctan2(df["yyards_raw"], df["xyards"])) > -30)
            & (df["yyards_raw"] < 5)
            & (df["yards_raw"] >= 12),
            "Swing",
            df["throw_type"],
        )

        # Dish
        df["throw_type"] = np.where(
            (df["t_after"].isin([8, 19, 20, 22])) & (df["yards_raw"] < 12),
            "Dish",
            df["throw_type"],
        )

        # Dumps
        df["throw_type"] = np.where(
            (df["t_after"].isin([8, 19, 20, 22]))
            & (np.degrees(np.arctan2(df["yyards_raw"], df["xyards"])) <= -30),
            "Dump",
            df["throw_type"],
        )

        # Hucks
        df["throw_type"] = np.where(
            (df["t_after"].isin([8, 19, 20, 22])) & (df["yyards_raw"] >= 45),
            "Huck",
            df["throw_type"],
        )

        return df

    def play_description(self, df, home=True):
        """Create a description of the play."""
        if home:
            roster = self.get_home_roster()
        else:
            roster = self.get_away_roster()

        # Convert player IDs to names
        player_names = (
            roster.assign(
                name=lambda x: x["first_name"].str.strip()
                + " "
                + x["last_name"].str.strip()
            )[["id", "name"]]
            .set_index("id")
            .to_dict()["name"]
        )

        # Normal completion
        df["play_description"] = np.where(
            df["t_after"].isin([20]),
            "Completion: "
            + df["r"].map(player_names)
            + " "
            + df["throw_type"]
            + " to<br>"
            + df["r_after"].map(player_names)
            + " for "
            + df["yyards"].round(0).fillna(0).astype(int).astype(str)
            + " yards",
            "",
        )

        # Callahan caught
        df["play_description"] = np.where(
            df["t_after"].isin([6]),
            "Score: " + df["r"].map(player_names) + " Callahan",
            df["play_description"],
        )

        # Score
        df["play_description"] = np.where(
            df["t_after"].isin([22]),
            "Score: "
            + df["r"].map(player_names)
            + " "
            + df["throw_type"]
            + " to<br>"
            + df["r_after"].map(player_names)
            + " for "
            + df["yyards"].round(0).fillna(0).astype(int).astype(str)
            + " yards",
            df["play_description"],
        )

        # Drop
        df["play_description"] = np.where(
            df["t_after"].isin([19]),
            "Turnover: "
            + df["r"].map(player_names)
            + " "
            + df["throw_type"]
            + " for "
            + df["yyards"].round(0).fillna(0).astype(int).astype(str)
            + " yards<br>dropped by "
            + df["r_after"].map(player_names),
            df["play_description"],
        )

        # Throwaway
        df["play_description"] = np.where(
            df["t_after"].isin([8]),
            "Turnover: "
            + df["r"].map(player_names)
            + " "
            + df["throw_type"]
            + " for<br>"
            + df["yyards"].round(0).fillna(0).astype(int).astype(str)
            + " yards thrown away",
            df["play_description"],
        )

        # Callahan thrown
        df["play_description"] = np.where(
            df["t_after"].isin([7]),
            "Turnover: "
            + df["r"].map(player_names)
            + " "
            + df["throw_type"]
            + " for<br>"
            + df["yyards"].round(0).fillna(0).astype(int).astype(str)
            + " yards thrown for Callahan",
            df["play_description"],
        )

        # Stall
        df["play_description"] = np.where(
            df["t_after"].isin([17]),
            "Turnover: " + df["r"].map(player_names) + " stall",
            df["play_description"],
        )

        # Travel
        df["play_description"] = np.where(
            df["t_after"].isin([10]),
            "Travel: " + df["r"].map(player_names),
            df["play_description"],
        )

        # Offensive Foul
        df["play_description"] = np.where(
            df["t_after"].isin([13]),
            "Offensive Foul: " + df["r"].map(player_names),
            df["play_description"],
        )

        # Defensive Foul
        df["play_description"] = np.where(
            df["t_after"].isin([12]), "Defensive Foul", df["play_description"]
        )

        # End of Period
        df["play_description"] = np.where(
            df["t_after"].isin([23, 24, 25, 26, 27]),
            "End of Period",
            df["play_description"],
        )

        return df

    def events_print_qc(self, df, qc=False):
        """Print basic QC info about the processed events data."""
        if qc:
            print("Number of events:", df.shape[0])
            print(
                "New Event Types:",
                ", ".join(
                    str(x)
                    for x in sorted(
                        list(set(df["t"].unique()).difference(set(EVENT_TYPES.keys())))
                    )
                ),
            )
            print("Event Attributes:", ", ".join(list(df)[: list(df).index("game_id")]))

        return df

    def get_events_filename(self, home=True):
        """Get string for events file."""
        homeawaystr = {True: "home", False: "away"}
        return (
            f"{self.get_game_info()['ext_game_id'].iloc[0]}_{homeawaystr[home]}.feather"
        )

    def get_events(self, home=True, upload=None, download=None, qc=False):
        """Process the events for a single team to get yardage, event labels, etc."""
        if upload is None:
            upload = self.upload
        if download is None:
            download = self.download

        events_file_name = join(self.events_path, self.get_events_filename(home=home))
        # If file doesn't exist locally, try to retrieve it from AWS
        if not Path(events_file_name).is_file() and download:
            download_from_bucket(events_file_name)

        # If events have already been processed and saved, load them
        if Path(events_file_name).is_file():
            try:
                df = pd.read_feather(events_file_name)
            except FileNotFoundError as e:
                time.sleep(1)
                df = pd.read_feather(events_file_name)

        # If events have not been processed and saved before, do so
        else:
            # Set parameters for home or away
            if home:
                events_raw = self.get_home_events_raw()
            else:
                events_raw = self.get_away_events_raw()

            df = (
                pd.DataFrame.from_records(events_raw)
                .pipe(self.add_fourth_period)
                .pipe(self.get_events_basic_info, home=home)
                .pipe(self.get_events_periods)
                .pipe(self.get_events_possession_labels)
                .pipe(self.get_events_pull_info)
                .pipe(self.get_events_stalls)
                .pipe(self.get_events_o_penalties)
                .pipe(self.get_events_d_penalties)
                .pipe(self.fill_missing_info)
                .pipe(self.get_events_yardage)
                .pipe(self.get_events_times)
                .pipe(self.get_events_lineups)
                .pipe(self.get_events_throw_classifications)
                .pipe(self.play_description, home=home)
                .pipe(self.events_print_qc, qc=qc)
                .rename(columns=lambda x: str(x))
            )
            df.to_feather(events_file_name)
            if upload:
                upload_to_bucket(events_file_name)

        return df

    def get_home_events(self, qc=False):
        """Get processed dataframe of home team events."""
        if self.home_events is None:
            self.home_events = self.get_events(home=True, qc=qc)
        return self.home_events

    def get_away_events(self, qc=False):
        """Get processed dataframe of away team events."""
        if self.away_events is None:
            self.away_events = self.get_events(home=False, qc=qc)
        return self.away_events

    def total_time_to_readable(self, time):
        """Convert total seconds from start of game to min:sec left in quarter."""
        total_left_quarter = pd.Series(
            np.where(time < 2880, 720 - time % 720, 300 - time % 720), index=time.index
        )
        minutes = np.floor(total_left_quarter / 60)
        seconds = total_left_quarter % 60
        return (
            minutes.astype(int).astype(str)
            + ":"
            + seconds.astype(int).astype(str).str.zfill(2)
        )

    def get_player_segments(self, roster, events, qc=False):
        # Columns we'll keep for plotting
        final_cols = [
            "playerid",
            "name",
            "lastname",
            "firstname",
            "o_point",
            "point_outcome",
            "point_hold",
            "num_turnovers",
            "period",
            "s_before_total",
            "s_after_total",
            "elapsed",
            # "s_before_readable",
            # "s_after_readable",
        ]

        # Get the points that each player was on the field and stack them so
        #    that each player has their own row for every point they played
        dfs = []
        for i, player in roster.iterrows():
            playerid = str(player["id"])
            # Only get players who played in the game
            if playerid in events:
                df = (
                    events.loc[events[playerid] == 1]
                    .groupby(["period", "s_before", "s_after"])
                    .head(1)
                    .query("elapsed!=0")
                    .assign(
                        playerid=playerid,
                        firstname=player["first_name"].strip(),
                        lastname=player["last_name"].strip(),
                        name=player["first_name"].strip()
                        + " "
                        + player["last_name"].strip(),
                        period=lambda x: x["period"].astype(str),
                    )
                )
                dfs.append(df)

        # First time each player played in the game
        firsttime = (
            pd.concat(dfs)
            .groupby("playerid")["s_before_total"]
            .min()
            .rename("firsttime")
            .reset_index()
        )

        # Sort the players by when they first entered the game
        times = (
            pd.concat(dfs)
            .merge(firsttime, how="left", on=["playerid"])
            .sort_values("firsttime")[final_cols]
            .reset_index(drop=True)
        )

        # Get all unique segments in the game and label them from 0 to n
        segments = (
            times.groupby(["s_before_total", "s_after_total"])
            .head(1)
            .assign(segment=lambda x: range(x.shape[0]))[
                ["s_before_total", "s_after_total", "segment"]
            ]
        )

        # Identify which players were on the field for each segment
        psegs = (
            times.merge(segments, how="left", on=["s_before_total", "s_after_total"])
            .sort_values("segment")[["playerid", "segment"]]
            .drop_duplicates()
            .assign(exists=1)
            .set_index(["playerid", "segment"])
            .unstack(level="segment", fill_value=0)
        )
        psegs.columns = psegs.columns.get_level_values(1)

        if qc:
            print("Segments w/o 7 players:", (psegs.sum() != 7).sum())

        return times, psegs

    def get_player_clusters(self, times, psegs):
        """Separate team into 3 clusters (O, D1, D2) based on points played together."""
        # Sort each player into 1 of 3 clusters based on which segments they played
        #    3 segments were chosen to mimic the common pattern of O, D1, and D2
        X = psegs.values
        # cluster_model = KMeans(n_clusters=3, random_state=0).fit(X)
        cluster_model = Birch().fit(X)

        # Add clusters to player segment data
        psegs["cluster"] = [str(x) for x in cluster_model.labels_]

        # Combine the player time data with the clusters
        player_clusters = times.merge(
            psegs.reset_index()[["playerid", "cluster"]], how="left", on=["playerid"]
        ).sort_values(["cluster"])

        # Get the avg. number of o-point each cluster played and the avg. earliest time they played
        #     Most o-points=O-line
        #     First non-o-line group on the field=D1
        #     Remaining group=D2
        clusters = (
            player_clusters.groupby(["playerid", "cluster"])
            .agg({"s_before_total": "min", "o_point": "sum"})
            .reset_index()
            .groupby(["cluster"])
            .agg({"s_before_total": "mean", "o_point": "mean"})
            .sort_values(["o_point", "s_before_total"], ascending=[False, True])
            .reset_index()
            # .assign(cluster_name=["O-Line", "D1", "D2"])
        )

        # Get the number of points each player played, which is used to sort players within each cluster
        playingtime = (
            player_clusters.groupby("playerid")["elapsed"]
            .count()
            .rename("segment_count")
            .reset_index()
        )

        # Combine and sort the data
        player_clusters = (
            player_clusters.merge(
                clusters, how="left", on=["cluster"], suffixes=["", "_c"]
            )
            .merge(playingtime, how="left", on=["playerid"])
            .sort_values(
                [
                    "o_point_c",
                    "s_before_total_c",
                    "segment_count",
                    "lastname",
                    "firstname",
                ],
                ascending=[False, True, False, True, True],
            )
            .assign(
                o_point=lambda x: np.where(x["o_point"], "O-Point", "D-Point"),
                s_before_readable=lambda x: self.total_time_to_readable(
                    x["s_before_total"]
                ),
                s_after_readable=lambda x: self.total_time_to_readable(
                    x["s_after_total"]
                ),
            )
            .assign(
                s_after_readable=lambda x: x["s_after_readable"]
                .map({"12:00": "0:00"})
                .fillna(x["s_after_readable"])
            )
            .reset_index(drop=True)
        )

        return player_clusters

    def get_game_flow_margins(self):
        """Get the margin properties based on the length of player names."""
        # Get all players who were in the game on either team
        events = pd.concat(
            [self.get_home_events(qc=False), self.get_away_events(qc=False)]
        )
        rosters = pd.concat([self.get_home_roster(), self.get_away_roster()])
        playerids = events.query("r==r")["r"].unique()

        # Get the length of the longest name on either team and set the margin based on that
        longest_name = (
            rosters.loc[rosters["id"].isin(playerids)]
            .assign(
                name_length=lambda x: x["first_name"].str.len()
                + x["last_name"].str.len()
            )["name_length"]
            .max()
        )
        left_margin = longest_name * 6

        return dict(t=5, b=20, l=left_margin, r=150, autoexpand=False)

    def visual_game_score(self, qc=False):
        """Line chart showing scoring progression throughout the game."""
        # Use the home team events for the score timestamps
        events = self.get_home_events(qc=qc)

        # Get home and away team cities for labeling
        home_team = self.get_home_team()["city"].iloc[0]
        away_team = self.get_away_team()["city"].iloc[0]

        # Get the time of all scores
        df = (
            events.query("t==[21,22]")[["t", "s_total", "period"]]
            .reset_index(drop=True)
            .reset_index()
            .assign(
                team=lambda x: np.where(x["t"] == 22, home_team, away_team),
                points=lambda x: x.groupby(["team"])["t"].cumcount() + 1,
                s_readable=lambda x: self.total_time_to_readable(x["s_total"]),
            )
            .drop(columns=["index", "t"])
            # Create a record for both teams on the timestamp of every score
            .set_index(["s_total", "s_readable", "team"])
            .unstack(["team"])
            .fillna(method="ffill")
            .stack()
            .reset_index()
        )

        # Add points for the start of the game when the score was 0-0
        start = pd.DataFrame(
            {
                "s_total": [0, 0],
                "s_readable": ["12:00", "12:00"],
                "team": [home_team, away_team],
                "points": [0, 0],
            }
        )

        # Add points for the end of the game
        if df["s_total"].max() > 4 * 12 * 60 + 5 * 60:
            end_time = df["s_total"].max()
        elif df["s_total"].max() > 4 * 12 * 60:
            end_time = 4 * 12 * 60 + 5 * 60
        else:
            end_time = 4 * 12 * 60

        xrange = [-20, end_time + 10]

        end = pd.DataFrame(
            {
                "s_total": [end_time, end_time],
                "s_readable": ["0:00", "0:00"],
                "team": [home_team, away_team],
                "points": [
                    df.query(f"team=='{home_team}'")["points"].max(),
                    df.query(f"team=='{away_team}'")["points"].max(),
                ],
            }
        )

        df = start.append(df).append(end)

        # Create line graph
        fig = px.line(
            df,
            x="s_total",
            y="points",
            color="team",
            line_shape="hv",
            hover_name="s_readable",
            custom_data=["team", "s_readable"],
            height=250,
        )

        # Set the labels for the x-axis tick marks
        xticks = {
            0: "Game Start",
            60 * 12 * 1: "End of Q1",
            60 * 12 * 2: "End of Q2",
            60 * 12 * 3: "End of Q3",
            60 * 12 * 4: "End of Q4",
        }

        # Add OT if there are events that take place beyond 4 quarters
        if df["s_total"].max() > 4 * 12 * 60:
            xticks[60 * 12 * 4 + 60 * 5 * 1] = "End of OT1"

        # Add vertical lines to mark quarters
        for xval, label in xticks.items():
            line_dash = "solid"
            line_color = "lightgray"
            line_width = 1
            fig.add_shape(
                type="line",
                y0=0,
                y1=df["points"].max(),
                x0=xval,
                x1=xval,
                line_width=line_width,
                line_color=line_color,
                line_dash=line_dash,
                layer="below",
            )

            # Add labels for each quarter
            fig.add_annotation(
                xref="x", yref="y", x=xval, y=-2, showarrow=False, text=label
            )

        # Add all times corresponding with scores
        for i, row in (
            df[["s_total", "s_readable", "period"]].drop_duplicates().iterrows()
        ):
            if row["s_total"] not in xticks.keys():
                if row["period"] <= 4:
                    period_str = f"Q{int(row['period'])}"
                elif row["period"] == 5:
                    period_str = "OT1"
                else:
                    period_str = "OT2"
                xticks[row["s_total"]] = f"{period_str}: {row['s_readable']}"

        # Change y-axis label
        fig.update_layout(
            # Add tick labels to fig
            xaxis=dict(
                title=None,
                range=xrange,
                tickmode="array",
                tickvals=list(xticks.keys()),
                ticktext=list(xticks.values()),
                ticks="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showspikes=True,
                spikesnap="cursor",
                spikemode="across",
                spikecolor="black",
                spikethickness=2,
                spikedash="solid",
                fixedrange=True,
            ),
            # Change y axis title
            yaxis=dict(
                title="Points",
                showgrid=False,
                zerolinecolor="lightgray",
                fixedrange=True,
            ),
            # Remove legend title
            legend=dict(title=None),
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=self.get_game_flow_margins(),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change hovermode to show info for both lines
            hovermode="x unified",
        )

        # Customize info shown on hover
        hovertext = "".join(["%{customdata[0]}: %{y}", "<extra></extra>"])
        fig.update_traces(hovertemplate=hovertext, legendgroup="a")

        return fig

    def visual_game_flow(self, color="point_outcome", home=True, qc=False):
        """Gantt chart showing the substitution patterns of a team throughout 1 game."""
        # Monkey patch for plotly so that we don't need to use datetimes for x-axis of gantt
        def my_process_dataframe_timeline(args):
            """Massage input for bar traces for px.timeline()."""
            args["is_timeline"] = True
            if args["x_start"] is None or args["x_end"] is None:
                raise ValueError("Both x_start and x_end are required")

            x_start = args["data_frame"][args["x_start"]]
            x_end = args["data_frame"][args["x_end"]]

            # We are not adding any columns to the data frame here, so no risk of overwrite
            args["data_frame"][args["x_end"]] = x_end - x_start
            args["x"] = args["x_end"]
            del args["x_end"]
            args["base"] = args["x_start"]
            del args["x_start"]
            return args

        px._core.process_dataframe_timeline = my_process_dataframe_timeline

        # Get data based on home/away team selection
        if home:
            events = self.get_home_events(qc=False)
            roster = self.get_home_roster()
        else:
            events = self.get_away_events(qc=False)
            roster = self.get_away_roster()

        # Get the segments that each player played
        times, psegs = self.get_player_segments(roster=roster, events=events, qc=qc)

        # Group players into O-line, D1, and D2
        player_clusters = self.get_player_clusters(times=times, psegs=psegs)

        # Get the order in which we want to show the players on the y-axis
        name_order = player_clusters.groupby(["name"]).head(1)["name"].values
        point_hold_order = [
            "Hold",
            "Opponent Break",
            "Opponent Hold",
            "Break",
            "End of Period",
        ]
        point_outcome_order = [
            "Score",
            "Callahan",
            "Opponent Score",
            "Opponent Callahan",
            "End of Period",
        ]
        o_point_order = ["O-Point", "D-Point"]

        # Text to show on hover
        hovertext = "<br>".join(
            [
                "%{customdata[0]}",
                "Segment Start: %{customdata[1]}%{customdata[6]}",
                "Segment End: %{customdata[2]}%{customdata[7]}",
                "Initial Possession: %{customdata[3]}",
                "Outcome: %{customdata[4]}",
                "Total Turns During Point: %{customdata[5]}",
                "<extra></extra>",
            ]
        )

        # Set the labels for the x-axis tick marks
        xticks = {
            0: "Game Start",
            60 * 12 * 1: "End of Q1",
            60 * 12 * 2: "End of Q2",
            60 * 12 * 3: "End of Q3",
            60 * 12 * 4: "End of Q4",
        }

        # Add OT if there are events that take place beyond 4 quarters
        if player_clusters["s_after_total"].max() > 4 * 12 * 60:
            xticks[60 * 12 * 4 + 60 * 5 * 1] = "End of OT1"

        # Create second x-axis that is only for timeouts and injuries
        x2ticks = dict()

        # Mark timeouts
        for i, row in events.query("t==[14, 15]").iterrows():
            x2ticks[row["s_total"]] = "Timeout"

        # Mark injuries
        for i, row in events.query("t==[42, 43]").iterrows():
            x2ticks[row["s_total"]] = "Injury"

        # Add note about timeouts at the start of the segment
        player_clusters["before_event"] = np.where(
            player_clusters["s_before_total"].isin(
                events.query("t==[14, 15]")["s_total"].unique()
            ),
            " (Timeout)",
            "",
        )

        # Add note about injuries at the start of the segment
        player_clusters["before_event"] = np.where(
            player_clusters["s_before_total"].isin(
                events.query("t==[42, 43]")["s_total"].unique()
            ),
            " (Injury)",
            player_clusters["before_event"],
        )

        # Add note about timeouts at the end of the segment
        player_clusters["after_event"] = np.where(
            player_clusters["s_after_total"].isin(
                events.query("t==[14, 15]")["s_total"].unique()
            ),
            " (Timeout)",
            "",
        )

        # Add note about injuries at the end of the segment
        player_clusters["after_event"] = np.where(
            player_clusters["s_after_total"].isin(
                events.query("t==[42, 43]")["s_total"].unique()
            ),
            " (Injury)",
            player_clusters["after_event"],
        )

        # Create the figure
        fig = px.timeline(
            player_clusters,
            x_start="s_before_total",
            x_end="s_after_total",
            y="name",
            color=color,
            category_orders={
                "name": name_order,
                "point_hold": point_hold_order,
                "point_outcome": point_outcome_order,
                "o_point": o_point_order,
            },
            custom_data=[
                "name",
                "s_before_readable",
                "s_after_readable",
                "o_point",
                "point_hold",
                "num_turnovers",
                "before_event",
                "after_event",
            ],
        )
        fig.layout.xaxis.type = "linear"

        fig.update_traces(hovertemplate=hovertext)

        # Add vertical lines to mark quarters
        for xval, label in xticks.items():
            line_dash = "solid"
            line_color = "white"
            line_width = 2
            fig.add_vline(
                x=xval,
                line_width=line_width,
                line_color=line_color,
                line_dash=line_dash,
                layer="above",
            )

        # Add vertical lines to mark timeouts and injuries
        for xval, label in x2ticks.items():
            line_dash = "dash"
            line_color = "white"
            line_width = 2
            fig.add_vline(
                x=xval,
                line_width=line_width,
                line_color=line_color,
                line_dash=line_dash,
                layer="above",
            )

        # Get x-axis range
        xrange = [0, events["s_total"].max()]

        fig.update_layout(
            # Add tick labels to fig
            xaxis=dict(
                range=xrange,
                tickmode="array",
                tickvals=list(xticks.keys()),
                ticktext=list(xticks.values()),
                ticks="",
                showgrid=False,
                fixedrange=True,
            ),
            # Remove y axis title
            yaxis=dict(title=None, fixedrange=True),
            # Remove legend title
            legend=dict(
                title=None,
                # # Change to horizontal legend on top
                # orientation="h",
                # yanchor="bottom",
                # y=1.02,
                # xanchor="right",
                # x=1,
            ),
            # Set title and colors for turnovers
            coloraxis_colorbar_title="Turns During Point",
            coloraxis_colorscale=[
                [0, "rgb(150,150,150)"],
                [0.33, "rgb(200,200,50)"],
                [0.67, "rgb(200,100,50)"],
                [1, "rgb(200,50,50)"],
            ],
            coloraxis_colorbar_tick0=0,
            coloraxis_colorbar_dtick=1,
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            # Set margins
            margin=self.get_game_flow_margins(),
        )

        # Set the ranges for shading the chart by cluster
        cluster_shading_ranges = (
            player_clusters.groupby(["name"])
            .head(1)
            .reset_index(drop=True)
            .reset_index()
            .assign(index_reverse=lambda x: x["index"].max() - x["index"])
            .groupby(["cluster"])["index_reverse"]
            .agg(["min", "max"])
            .sort_values(["min"], ascending=[False])
            .reset_index()
        )

        # Shade each of the clusters
        for i, row in cluster_shading_ranges.iterrows():
            fig.add_hrect(
                y0=row["min"] - 0.5,
                y1=row["max"] + 0.5,
                line=dict(width=0),
                fillcolor=px.colors.qualitative.Set2[i],
                opacity=0.25,
                layer="below",
            )

        return fig

    def visual_possession_map_horizontal(self, possession_number, home=True):
        """Map of all throws in a possession by 1 team."""
        # Get data based on home/away team selection
        if home:
            events = self.get_home_events(qc=False)
            roster = self.get_home_roster()
        else:
            events = self.get_away_events(qc=False)
            roster = self.get_away_roster()

        # Only keep some events
        df = (
            events.query("t==[10, 12, 13, 20]")
            .query(f"possession_number=={possession_number}")
            .reset_index(drop=True)
            .reset_index()
            .assign(
                event=lambda x: x["index"] + 1,
                x=lambda x: x["x"] * -1,
                x_after=lambda x: x["x_after"] * -1,
                xyards_raw=lambda x: x["xyards_raw"] * -1,
            )
            .drop(columns=["index"])
            .copy()
        )

        # Re-label first event
        df.loc[df["event"] == 1, "event_name"] = "Start of Possession"
        df.loc[df["event"] == 1, "t"] = 0

        # Set figure width
        width = 800

        # Draw possession if there's data, otherwise draw a blank field
        try:
            last_row = df.loc[df["event"] == df["event"].max()].iloc[0].copy()

            # Add row for last event
            df = df.append(
                pd.Series(
                    {
                        "x": last_row["x_after"],
                        "y": last_row["y_after"],
                        "t": last_row["t_after"],
                        "yyards_raw": last_row["yyards_raw"],
                        "xyards_raw": last_row["xyards_raw"],
                        "yards_raw": last_row["yards_raw"],
                        "play_description": last_row["play_description"],
                        "event_name": last_row["event_name_after"],
                        "event": last_row["event"] + 1,
                        "r": last_row["r_after"],
                    }
                ),
                ignore_index=True,
            )

            # Shift play descriptions and yardages
            df["play_description"] = df["play_description"].shift(1)
            df["yyards_raw"] = df["yyards_raw"].shift(1)
            df["xyards_raw"] = df["xyards_raw"].shift(1)
            df["yards_raw"] = df["yards_raw"].shift(1)

            # Re-label first play description and yardages
            df.loc[df["event"] == 1, "play_description"] = "Start of Possession"
            df.loc[df["event"] == 1, "yyards_raw"] = 0
            df.loc[df["event"] == 1, "xyards_raw"] = 0
            df.loc[df["event"] == 1, "yards_raw"] = 0

            # Add colors
            event_colors = {
                10: "orange",
                12: "orange",
                13: "orange",
                20: "gray",
                22: "green",
                19: "red",
                17: "red",
                8: "red",
                0: "purple",
            }
            df["event_color"] = df["t"].map(event_colors)

            # Create animated scatter plot to represent the disc
            fig = px.scatter(
                data_frame=df,
                x="y",
                y="x",
                color_discrete_sequence=["gray"],
                symbol_sequence=["circle"],
                animation_frame="event",
                size=[1 for x in range(df.shape[0])],
                size_max=10,
                width=width,
                height=width * 54 / 120 + 100,
            )

            # Remove hover info for the disc
            fig.update_traces(
                selector={"name": ""}, hoverinfo="skip", hovertemplate=None
            )

            # Plot each type of event as a different color
            for event_name in df.sort_values("event")["event_name"].unique():
                group = df.query(f"event_name=='{event_name}'")
                fig.add_scatter(
                    x=group["y"],
                    y=group["x"],
                    marker_color=group["event_color"],
                    marker_symbol="diamond",
                    mode="markers",
                    name=group["event_name"].iloc[0],
                    hovertemplate="%{text}<extra></extra>",
                    text=[
                        "<br>".join(
                            [
                                f"<b>{row['play_description']}</b>",
                                f"Downfield Yards: {row['yyards_raw']:.1f}",
                                f"Sideways Yards: {row['xyards_raw']:.1f}",
                                f"Total Yards: {row['yards_raw']:.1f}",
                            ]
                        )
                        for i, row in group.iterrows()
                    ],
                )

            # Change slider labels
            for i, step in enumerate(fig.layout["sliders"][0]["steps"]):
                fig.layout["sliders"][0]["steps"][i]["label"] = df.loc[
                    df["event"] == i + 1, "play_description"
                ].iloc[0]

            # Remove slider prefix
            fig.layout["sliders"][0]["currentvalue"]["prefix"] = ""

            # Adjust slider position
            fig.layout["sliders"][0]["pad"] = {"b": -25, "t": 25}
            fig.layout["sliders"][0]["y"] = 0.1

            # Adjust play and stop button position
            fig.layout["updatemenus"][0]["x"] = 0.1
            fig.layout["updatemenus"][0]["y"] = 0.15

            # Remove hover info for the disc
            for i, _ in enumerate(fig.frames):
                fig.frames[i]["data"][0]["hovertemplate"] = None

            # Add a line for each of the throws
            for i, row in df.iterrows():
                if (row["x"] == row["x"]) and (row["x_after"] == row["x_after"]):
                    fig.add_shape(
                        type="line",
                        x0=row["y"],
                        y0=row["x"],
                        x1=row["y_after"],
                        y1=row["x_after"],
                        line=dict(color=event_colors[row["t_after"]]),
                        layer="below",
                    )
        except IndexError as e:
            fig = go.Figure()
            fig.update_layout(width=width, height=width * 54 / 120)

        # Draw field boundaries
        # Vertical lines
        fig.add_shape(type="line", y0=-25, y1=25, x0=0, x1=0, line=dict(color="black"))
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=20, x1=20, line=dict(color="black")
        )
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=100, x1=100, line=dict(color="black")
        )
        fig.add_shape(
            type="line", y0=-25, y1=25, x0=120, x1=120, line=dict(color="black")
        )

        # Horizontal lines
        fig.add_shape(
            type="line", y0=-25, y1=-25, x0=0, x1=120, line=dict(color="black")
        )
        fig.add_shape(type="line", y0=25, y1=25, x0=0, x1=120, line=dict(color="black"))

        # Add arrow to indicate attacking direction
        fig.add_annotation(
            xref="x",
            yref="y",
            y=28,
            x=80,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            axref="x",
            ayref="y",
            ay=28,
            ax=60,
            text="Attacking",
        )

        # Set figure properties
        fig.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            yaxis=dict(
                range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
            # Add tick labels to fig
            xaxis=dict(
                range=[-1, 121],
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
            showlegend=False,
            # Remove margins
            margin=dict(t=0, b=0, l=0, r=0),
        )

        return fig

    def visual_possession_map_vertical(self, possession_number, home=True):
        """Map of all throws in a possession by 1 team."""
        # Get data based on home/away team selection
        if home:
            events = self.get_home_events(qc=False)
            roster = self.get_home_roster()
        else:
            events = self.get_away_events(qc=False)
            roster = self.get_away_roster()

        # Only keep some events
        df = (
            events.query("t==[10, 12, 13, 20]")
            .query(f"possession_number=={possession_number}")
            .reset_index(drop=True)
            .reset_index()
            .assign(
                event=lambda x: x["index"] + 1,
                x=lambda x: x["x"],
                x_after=lambda x: x["x_after"],
                xyards_raw=lambda x: x["xyards_raw"],
            )
            .drop(columns=["index"])
            .copy()
        )

        # Re-label first event
        df.loc[df["event"] == 1, "event_name"] = "Start of Possession"
        df.loc[df["event"] == 1, "t"] = 0

        # Set figure width
        height = 800

        # Draw possession if there's data, otherwise draw a blank field
        try:
            last_row = df.loc[df["event"] == df["event"].max()].iloc[0].copy()

            # Add row for last event
            df = df.append(
                pd.Series(
                    {
                        "x": last_row["x_after"],
                        "y": last_row["y_after"],
                        "t": last_row["t_after"],
                        "yyards_raw": last_row["yyards_raw"],
                        "xyards_raw": last_row["xyards_raw"],
                        "yards_raw": last_row["yards_raw"],
                        "play_description": last_row["play_description"],
                        "event_name": last_row["event_name_after"],
                        "event": last_row["event"] + 1,
                        "r": last_row["r_after"],
                    }
                ),
                ignore_index=True,
            )

            # Shift play descriptions and yardages
            df["play_description"] = df["play_description"].shift(1)
            df["yyards_raw"] = df["yyards_raw"].shift(1)
            df["xyards_raw"] = df["xyards_raw"].shift(1)
            df["yards_raw"] = df["yards_raw"].shift(1)

            # Re-label first play description and yardages
            df.loc[df["event"] == 1, "play_description"] = "Start of Possession"
            df.loc[df["event"] == 1, "yyards_raw"] = 0
            df.loc[df["event"] == 1, "xyards_raw"] = 0
            df.loc[df["event"] == 1, "yards_raw"] = 0

            # Add colors
            event_colors = {
                10: "orange",
                12: "orange",
                13: "orange",
                20: "gray",
                22: "green",
                19: "red",
                17: "red",
                8: "red",
                0: "purple",
            }
            df["event_color"] = df["t"].map(event_colors)

            # Create animated scatter plot to represent the disc
            fig = px.scatter(
                data_frame=df,
                x="x",
                y="y",
                color_discrete_sequence=["gray"],
                symbol_sequence=["circle"],
                animation_frame="event",
                size=[1 for x in range(df.shape[0])],
                size_max=10,
                width=height * 54 / 120 + 25,
                height=height,
            )

            # Remove hover info for the disc
            fig.update_traces(
                selector={"name": ""}, hoverinfo="skip", hovertemplate=None
            )

            # Plot each type of event as a different color
            for event_name in df.sort_values("event")["event_name"].unique():
                group = df.query(f"event_name=='{event_name}'")
                fig.add_scatter(
                    x=group["x"],
                    y=group["y"],
                    marker_color=group["event_color"],
                    marker_symbol="diamond",
                    mode="markers",
                    name=group["event_name"].iloc[0],
                    hovertemplate="%{text}<extra></extra>",
                    text=[
                        "<br>".join(
                            [
                                f"<b>{row['play_description']}</b>",
                                f"Downfield Yards: {row['yyards_raw']:.1f}",
                                f"Sideways Yards: {row['xyards_raw']:.1f}",
                                f"Total Yards: {row['yards_raw']:.1f}",
                            ]
                        )
                        for i, row in group.iterrows()
                    ],
                )

            # Change slider labels
            for i, step in enumerate(fig.layout["sliders"][0]["steps"]):
                fig.layout["sliders"][0]["steps"][i]["label"] = df.loc[
                    df["event"] == i + 1, "play_description"
                ].iloc[0]

            # Remove slider prefix
            fig.layout["sliders"][0]["currentvalue"]["prefix"] = ""

            # Adjust slider position
            fig.layout["sliders"][0]["pad"] = {"b": -25, "t": 25}
            fig.layout["sliders"][0]["y"] = 0.1

            # Adjust play and stop button position
            fig.layout["updatemenus"][0]["x"] = 0.1
            fig.layout["updatemenus"][0]["y"] = 0.125

            # Remove hover info for the disc
            for i, _ in enumerate(fig.frames):
                fig.frames[i]["data"][0]["hovertemplate"] = None

            # Add a line for each of the throws
            for i, row in df.iterrows():
                if (row["x"] == row["x"]) and (row["x_after"] == row["x_after"]):
                    fig.add_shape(
                        type="line",
                        x0=row["x"],
                        y0=row["y"],
                        x1=row["x_after"],
                        y1=row["y_after"],
                        line=dict(color=event_colors[row["t_after"]]),
                        layer="below",
                    )
        except IndexError as e:
            fig = go.Figure()
            fig.update_layout(width=height * 54 / 120 + 25, height=height)

        # Draw field boundaries
        # Vertical lines
        fig.add_shape(type="line", x0=-25, x1=25, y0=0, y1=0, line=dict(color="black"))
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=20, y1=20, line=dict(color="black")
        )
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=100, y1=100, line=dict(color="black")
        )
        fig.add_shape(
            type="line", x0=-25, x1=25, y0=120, y1=120, line=dict(color="black")
        )

        # Horizontal lines
        fig.add_shape(
            type="line", x0=-25, x1=-25, y0=0, y1=120, line=dict(color="black")
        )
        fig.add_shape(type="line", x0=25, x1=25, y0=0, y1=120, line=dict(color="black"))

        # Add arrow to indicate attacking direction
        fig.add_annotation(
            xref="x",
            yref="y",
            x=30,
            y=80,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            axref="x",
            ayref="y",
            ax=30,
            ay=60,
            text="Attacking",
        )

        # Set figure properties
        fig.update_layout(
            # Remove axis titles
            xaxis_title=None,
            yaxis_title=None,
            # Add tick labels to fig
            xaxis=dict(
                range=[-27, 35],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            # Add tick labels to fig
            yaxis=dict(
                range=[-1, 121],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
            # Transparent background
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # Change font
            font_family="TW Cen MT",
            hoverlabel_font_family="TW Cen MT",
            showlegend=False,
            # Remove margins
            margin=dict(t=0, b=0, l=0, r=0),
        )

        return fig

    def flatten_columns(self, df):
        df.columns = ["_".join(col).rstrip("_") for col in df.columns.values]
        return df

    def get_player_points_played(self, df, start_only=True):
        """Get the number of O and D points played by each player.

        start_only=True means we only count who was on at the start of the point.
        start_only=False means we only count who was on at the end of the point.

        """
        points_played = []
        # Get all player IDs for the team in this game
        players = [x for x in list(df) if x.isdigit()]
        for playerid in players:
            if start_only:
                dftemp = df.groupby(["point_number"]).head(1)
            else:
                dftemp = df.groupby(["point_number"]).tail(1)

            # Get the number of o and d points each player played
            player_points = (
                dftemp.assign(turnover=lambda x: x["num_turnovers"] > 0)
                .groupby([playerid, "o_point", "turnover", "point_outcome"])[
                    "point_number"
                ]
                .nunique()
                .rename("points")[1]
                .reset_index()
                .assign(playerid=playerid)
            )
            points_played.append(player_points)

        # Get number of o and d points for the entire team
        points_team = df["point_number"].nunique()
        o_points_team = df.query("o_point==True")["point_number"].nunique()
        d_points_team = df.query("o_point==False")["point_number"].nunique()

        # Combine all players into a single dataframe
        dfout = (
            pd.concat(points_played, ignore_index=True)
            .assign(
                o_point=lambda x: x["o_point"].map({True: "o_point", False: "d_point"}),
                turnover=lambda x: x["turnover"].map(
                    {True: "at least 1 turn", False: "no turns"}
                ),
            )
            .set_index(["playerid", "o_point", "turnover", "point_outcome"])
            .unstack(level=["o_point", "turnover", "point_outcome"], fill_value=0)
            .reset_index()
        )
        dfout.columns = ["_".join(col).rstrip("_") for col in dfout.columns.values]
        o_point_cols = [x for x in list(dfout) if "o_point" in x]
        o_point_score_cols = [
            x
            for x in list(dfout)
            if ("o_point" in x)
            and (("Score" in x) or ("Callahan" in x))
            and ("Opponent" not in x)
        ]
        o_point_noturn_cols = [
            x for x in list(dfout) if ("o_point" in x) and ("no turns" in x)
        ]
        d_point_cols = [x for x in list(dfout) if "d_point" in x]
        d_point_score_cols = [
            x
            for x in list(dfout)
            if ("d_point" in x)
            and (("Score" in x) or ("Callahan" in x))
            and ("Opponent" not in x)
        ]
        d_point_turn_cols = [
            x for x in list(dfout) if ("d_point" in x) and ("at least 1 turn" in x)
        ]
        dfout["total_points"] = dfout.sum(axis=1)
        dfout["o_points"] = dfout[o_point_cols].sum(axis=1)
        dfout["d_points"] = dfout[d_point_cols].sum(axis=1)
        dfout["o_point_scores"] = dfout[o_point_score_cols].sum(axis=1)
        dfout["d_point_scores"] = dfout[d_point_score_cols].sum(axis=1)
        dfout["o_point_noturns"] = dfout[o_point_noturn_cols].sum(axis=1)
        dfout["d_point_turns"] = dfout[d_point_turn_cols].sum(axis=1)
        dfout["points_team"] = points_team
        dfout["o_points_team"] = o_points_team
        dfout["d_points_team"] = d_points_team

        return dfout[
            [
                "playerid",
                "total_points",
                "o_points",
                "o_point_scores",
                "o_point_noturns",
                "d_points",
                "d_point_scores",
                "d_point_turns",
                "points_team",
                "o_points_team",
                "d_points_team",
            ]
        ]

    def get_player_yards(self, df):
        """Get the receiving and throwing yards for each player in a single game."""
        # Get total yards by the team when each player is on the field
        # Get all player IDs for the team in this game
        team_yards = []
        players = [x for x in list(df) if x.isdigit()]
        for playerid in players:
            df_team_yards_player = (
                df.query("t_after==[20, 22]")
                .query(f"`{playerid}`==1")[
                    ["xyards", "yyards", "yyards_raw", "yards", "yards_raw",]
                ]
                .sum()
                .to_frame()
                .T.add_suffix("_team")
                .assign(playerid=playerid)
            )
            team_yards.append(df_team_yards_player)
        df_team_yards = pd.concat(team_yards)

        df_throw = (
            df.query("t_after==[20, 22]")
            .assign(
                centering_pass=lambda x: x["centering_pass"].map(
                    {True: "center", False: ""}
                ),
                playerid=lambda x: x["r"].astype(int).astype(str),
            )
            .groupby(["playerid", "centering_pass"])
            .agg(
                {
                    "xyards": "sum",
                    "yyards": "sum",
                    "yyards_raw": "sum",
                    "yards": "sum",
                    "yards_raw": "sum",
                }
            )
            .rename(columns=lambda x: x + "_throwing")
            .unstack(level=["centering_pass"], fill_value=0)
            .reset_index()
            .pipe(self.flatten_columns)
            .assign(
                xyards_throwing_total=lambda x: x["xyards_throwing"]
                + x["xyards_throwing_center"],
                yyards_throwing_total=lambda x: x["yyards_throwing"]
                + x["yyards_throwing_center"],
                yyards_raw_throwing_total=lambda x: x["yyards_raw_throwing"]
                + x["yyards_raw_throwing_center"],
                yards_throwing_total=lambda x: x["yards_throwing"]
                + x["yards_throwing_center"],
                yards_raw_throwing_total=lambda x: x["yards_raw_throwing"]
                + x["yards_raw_throwing_center"],
            )
        )

        df_throwaway = (
            df.query("t_after==[7, 8]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .agg(
                {
                    "xyards": "sum",
                    "yyards": "sum",
                    "yyards_raw": "sum",
                    "yards": "sum",
                    "yards_raw": "sum",
                }
            )
            .rename(columns=lambda x: x + "_throwaway")
        )

        df_receive = (
            df.query("t_after==[20, 22]")
            .assign(
                centering_pass=lambda x: x["centering_pass"].map(
                    {True: "center", False: ""}
                ),
                playerid=lambda x: x["r_after"].astype(int).astype(str),
            )
            .groupby(["playerid", "centering_pass"])
            .agg(
                {
                    "xyards": "sum",
                    "yyards": "sum",
                    "yyards_raw": "sum",
                    "yards": "sum",
                    "yards_raw": "sum",
                }
            )
            .rename(columns=lambda x: x + "_receiving")
            .unstack(level=["centering_pass"], fill_value=0)
            .reset_index()
            .pipe(self.flatten_columns)
            .assign(
                xyards_receiving_total=lambda x: x["xyards_receiving"]
                + x["xyards_receiving_center"],
                yyards_receiving_total=lambda x: x["yyards_receiving"]
                + x["yyards_receiving_center"],
                yyards_raw_receiving_total=lambda x: x["yyards_raw_receiving"]
                + x["yyards_raw_receiving_center"],
                yards_receiving_total=lambda x: x["yards_receiving"]
                + x["yards_receiving_center"],
                yards_raw_receiving_total=lambda x: x["yards_raw_receiving"]
                + x["yards_raw_receiving_center"],
            )
        )

        dfout = (
            df_throw.merge(df_receive, how="outer", on=["playerid"])
            .merge(df_team_yards, how="outer", on=["playerid"])
            .merge(df_throwaway, how="outer", on=["playerid"])
            .fillna(0)
            .assign(
                xyards_total=lambda x: x["xyards_receiving_total"]
                + x["xyards_throwing_total"],
                yyards_total=lambda x: x["yyards_receiving_total"]
                + x["yyards_throwing_total"],
                yards_total=lambda x: x["yards_receiving_total"]
                + x["yards_throwing_total"],
                yyards_raw_total=lambda x: x["yyards_raw_receiving_total"]
                + x["yyards_raw_throwing_total"],
                yards_raw_total=lambda x: x["yards_raw_receiving_total"]
                + x["yards_raw_throwing_total"],
                xyards_center=lambda x: x["xyards_receiving_center"]
                + x["xyards_throwing_center"],
                yyards_center=lambda x: x["yyards_receiving_center"]
                + x["yyards_throwing_center"],
                yards_center=lambda x: x["yards_receiving_center"]
                + x["yards_throwing_center"],
                yyards_raw_center=lambda x: x["yyards_raw_receiving_center"]
                + x["yyards_raw_throwing_center"],
                yards_raw_center=lambda x: x["yards_raw_receiving_center"]
                + x["yards_raw_throwing_center"],
                xyards=lambda x: x["xyards_receiving"] + x["xyards_throwing"],
                yyards=lambda x: x["yyards_receiving"] + x["yyards_throwing"],
                yards=lambda x: x["yards_receiving"] + x["yards_throwing"],
                yyards_raw=lambda x: x["yyards_raw_receiving"]
                + x["yyards_raw_throwing"],
                yards_raw=lambda x: x["yards_raw_receiving"] + x["yards_raw_throwing"],
            )
        )

        return dfout

    def get_player_touches(self, df):
        """Get all touches (completions and turns) for a player in a game."""
        team_counts = []
        players = [x for x in list(df) if x.isdigit()]
        for playerid in players:
            throwaways = df.query("t_after==[7, 8]").query(f"`{playerid}`==1").shape[0]
            drops = df.query("t_after==[19,]").query(f"`{playerid}`==1").shape[0]
            completions = (
                df.query("t_after==[20, 22]").query(f"`{playerid}`==1").shape[0]
            )
            stalls = df.query("t_after==[17,]").query(f"`{playerid}`==1").shape[0]
            goals = df.query("t_after==[22,]").query(f"`{playerid}`==1").shape[0]
            blocks = df.query("t==[5,6]").query(f"`{playerid}`==1").shape[0]
            callahans = df.query("t_after==[6,]").query(f"`{playerid}`==1").shape[0]
            team_counts.append(
                [
                    playerid,
                    throwaways,
                    drops,
                    completions,
                    stalls,
                    goals,
                    blocks,
                    callahans,
                ]
            )
        df_team_counts = pd.DataFrame(
            data=team_counts,
            columns=[
                "playerid",
                "throwaways_team",
                "drops_team",
                "completions_team",
                "stalls_team",
                "goals_team",
                "blocks_team",
                "callahans_team",
            ],
        )

        df_throwaway = (
            df.query("t_after==[7, 8]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("throwaways")
            .reset_index()
        )

        df_drop = (
            df.query("t_after==[19,]")
            .assign(playerid=lambda x: x["r_after"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("drops")
            .reset_index()
        )

        df_completion = (
            df.query("t_after==[20, 22]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("completions")
            .reset_index()
        )

        df_reception = (
            df.query("t_after==[20, 22]")
            .assign(playerid=lambda x: x["r_after"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("receptions")
            .reset_index()
        )

        df_stall = (
            df.query("t_after==[17,]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("stalls")
            .reset_index()
        )

        df_assist = (
            df.query("t_after==[22,]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("assists")
            .reset_index()
        )

        df_goal = (
            df.query("t_after==[22,]")
            .assign(playerid=lambda x: x["r_after"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("goals")
            .reset_index()
        )

        df_block = (
            df.query("t==[5,6]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("blocks")
            .reset_index()
        )

        df_callahan = (
            df.query("t==[6,]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid"])
            .size()
            .rename("callahans")
            .reset_index()
        )

        dfout = (
            df_throwaway.merge(df_completion, how="outer", on=["playerid"])
            .merge(df_reception, how="outer", on=["playerid"])
            .merge(df_drop, how="outer", on=["playerid"])
            .merge(df_stall, how="outer", on=["playerid"])
            .merge(df_assist, how="outer", on=["playerid"])
            .merge(df_goal, how="outer", on=["playerid"])
            .merge(df_block, how="outer", on=["playerid"])
            .merge(df_callahan, how="outer", on=["playerid"])
            .merge(df_team_counts, how="outer", on=["playerid"])
            .fillna(0)
            .assign(
                turnovers=lambda x: x["stalls"] + x["drops"] + x["throwaways"],
                plus_minus=lambda x: x["goals"]
                + x["assists"]
                + x["callahans"]
                - x["turnovers"],
                throw_attempts=lambda x: x["throwaways"]
                + x["completions"]
                + x["stalls"],
                catch_attempts=lambda x: x["receptions"] + x["drops"],
            )
        )

        return dfout

    def get_player_possessions_played(self, df, start_only=False):
        """Get the number of O and D possessions played by each player.

        start_only=True means we only count who was on at the start of the possession.
        start_only=False means we only count who was on at the end of the possession.

        """
        possessions_played = []
        # Get all player IDs for the team in this game
        players = [x for x in list(df) if x.isdigit()]
        for playerid in players:
            if start_only:
                dftemp = df.groupby(["possession_number"]).head(1)
            else:
                dftemp = df.groupby(["possession_number"]).tail(1)

            # Skip players that were not on for the beginning/end of any possessions
            if 1 not in dftemp[playerid].values:
                continue

            # Get the number of o and d possessions each player played
            player_possessions = (
                dftemp.assign(
                    turnover=lambda x: x["possession_outcome_general"] == "Turnover"
                )
                .groupby(
                    [playerid, "offensive_possession", "turnover", "possession_outcome"]
                )["possession_number"]
                .nunique()
                .rename("possessions")[1]
                .reset_index()
                .assign(playerid=playerid)
            )
            possessions_played.append(player_possessions)

        # Get the number of o and d possessions for the entire team
        possessions_team = df["possession_number"].nunique()
        o_possessions_team = df.query("offensive_possession==True")[
            "possession_number"
        ].nunique()
        d_possessions_team = df.query("offensive_possession==False")[
            "possession_number"
        ].nunique()

        # Combine all players into a single dataframe
        dfout = (
            pd.concat(possessions_played, ignore_index=True)
            .assign(
                offensive_possession=lambda x: x["offensive_possession"].map(
                    {True: "offensive_possession", False: "defensive_possession"}
                ),
                turnover=lambda x: x["turnover"].map(
                    {True: "Turnover", False: "No Turnover"}
                ),
            )
            .set_index(
                ["playerid", "offensive_possession", "turnover", "possession_outcome"]
            )
            .unstack(
                level=["offensive_possession", "turnover", "possession_outcome"],
                fill_value=0,
            )
            .reset_index()
        )
        dfout.columns = ["_".join(col).rstrip("_") for col in dfout.columns.values]
        o_possession_cols = [x for x in list(dfout) if "offensive_possession" in x]
        o_possession_score_cols = [
            x
            for x in list(dfout)
            if ("offensive_possession" in x)
            and ("Score" in x)
            and ("Opponent" not in x)
        ]
        d_possession_cols = [x for x in list(dfout) if "defensive_possession" in x]
        d_possession_score_cols = [
            x
            for x in list(dfout)
            if ("defensive_possession" in x) and ("Score" in x) and ("Opponent" in x)
        ]
        dfout["total_possessions"] = dfout.sum(axis=1)
        dfout["o_possessions"] = dfout[o_possession_cols].sum(axis=1)
        dfout["d_possessions"] = dfout[d_possession_cols].sum(axis=1)
        dfout["o_possession_scores"] = dfout[o_possession_score_cols].sum(axis=1)
        dfout["d_possession_scores_allowed"] = dfout[d_possession_score_cols].sum(
            axis=1
        )
        dfout["possessions_team"] = possessions_team
        dfout["o_possessions_team"] = o_possessions_team
        dfout["d_possessions_team"] = d_possessions_team

        return dfout[
            [
                "playerid",
                "total_possessions",
                "o_possessions",
                "o_possession_scores",
                "d_possessions",
                "d_possession_scores_allowed",
                "possessions_team",
                "o_possessions_team",
                "d_possessions_team",
            ]
        ]

    def get_player_time(self, roster, events):
        times, psegs = self.get_player_segments(roster=roster, events=events)
        dfout = (
            (times.groupby("playerid")["elapsed"].sum() / 60)
            .rename("minutes_played")
            .reset_index()
        )

        return dfout

    def get_player_throw_types(self, df):
        df_completion = (
            df.query("t_after==[20, 22]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid", "throw_type"])
            .size()
            .rename("completions")
            .reset_index()
        )

        df_reception = (
            df.query("t_after==[20, 22]")
            .assign(playerid=lambda x: x["r_after"].astype(int).astype(str),)
            .groupby(["playerid", "throw_type"])
            .size()
            .rename("receptions")
            .reset_index()
        )

        df_throwaway = (
            df.query("t_after==[7, 8]")
            .assign(playerid=lambda x: x["r"].astype(int).astype(str),)
            .groupby(["playerid", "throw_type"])
            .size()
            .rename("throwaways")
            .reset_index()
        )

        dfout = (
            df_completion.merge(
                df_throwaway, how="outer", on=["playerid", "throw_type"]
            )
            .merge(df_reception, how="outer", on=["playerid", "throw_type"])
            .fillna(0)
            .assign(attempts=lambda x: x["completions"] + x["throwaways"],)
            .set_index(["playerid", "throw_type"])
            .unstack(level=["throw_type"], fill_value=0)
            .pipe(self.flatten_columns)
            .rename(columns=lambda x: x.lower())
            .reset_index()
        )

        return dfout

    def get_player_info(self, roster, team, opponent, game_name):
        dfout = roster.assign(
            playerid=lambda x: x["id"].astype(int).astype(str),
            name=lambda x: x["first_name"].str.strip()
            + " "
            + x["last_name"].str.strip(),
            team=team,
            opponent=opponent,
            game_date=game_name[:10],
            year=self.year,
            games=1,
        )[
            ["playerid", "name", "team", "opponent", "game_date", "year", "games"]
        ].drop_duplicates()
        return dfout

    def get_player_stats_by_game(self, home=True):
        if home:
            events = self.get_home_events()
            roster = self.get_home_roster()
            team = self.get_home_team()["abbrev"].iloc[0]
            opponent = self.get_away_team()["abbrev"].iloc[0]
        else:
            events = self.get_away_events()
            roster = self.get_away_roster()
            team = self.get_away_team()["abbrev"].iloc[0]
            opponent = self.get_home_team()["abbrev"].iloc[0]
        game_name = self.get_game_info()["ext_game_id"].iloc[0]

        dfout = (
            self.get_player_info(
                roster=roster, team=team, opponent=opponent, game_name=game_name
            )
            .merge(
                self.get_player_points_played(df=events, start_only=True),
                how="right",
                on=["playerid"],
            )
            .merge(
                self.get_player_possessions_played(df=events, start_only=False),
                how="outer",
                on=["playerid"],
            )
            .merge(
                self.get_player_time(roster=roster, events=events),
                how="outer",
                on=["playerid"],
            )
            .merge(self.get_player_touches(df=events), how="outer", on=["playerid"])
            .merge(self.get_player_throw_types(df=events), how="outer", on=["playerid"])
            .merge(self.get_player_yards(df=events), how="outer", on=["playerid"])
            # Only keep records with a valid player ID that can be connected to a name
            .query("name==name")
        )

        # Fill in missing values
        for col in list(dfout):
            if ("pct" not in col) and (
                col not in ["playerid", "name", "team", "opponent", "game_date"]
            ):
                dfout[col] = dfout[col].fillna(0)

        # The below games have incorrect timestamp data, leading to incorrect minute totals
        # We'll make the minutes for these games null
        bad_timestamps = [
            ("LA", "2021-06-18"),
            ("CHI", "2021-07-11"),
        ]
        for team, game_date in bad_timestamps:
            dfout.loc[
                (dfout["team"] == team) & (dfout["game_date"] == game_date),
                "minutes_played",
            ] = None
        dfout = dfout.pipe(get_player_rates)  # .pipe(round_player_stats)

        return dfout

    def get_team_stats_by_game(self):
        # Completions
        # Hucks
        # Holds
        # Breaks
        # Red Zone
        # Blocks
        # Turns
        # Time of possession
        # Y raw Yds Rcv (and per throw)
        # Y Yds Rcv (and per throw)
        # X Yds Rcv (and per throw)
        # Total raw Yds Rcv (and per throw)
        # Total Yds Rcv  (and per throw)
        # Y raw Yds Thr (and per throw)
        # Y Yds Thr (and per throw)
        # X Yds Thr (and per throw)
        # Total raw Yds Thr (and per throw)
        # Total Yds Thr (and per throw)
        # seconds per throw? Maybe only possible if no turns or after a timeout. Would reflect both distance and time held
        pass

    # TODO What is the q attribute? It's 1 sometimes. In MIN-MAD, it was 1 for a score where the x and y vals were way off. q=questionable stat-keeping? Present on own-team score.
    # TODO What is the c attribute?  True or False. Present for opponent foul. Might be True when the disc gets centered. Present for week 1 MAD vs MIN
    # TODO What is the o attribute?  True or False. Possibly true/false for OT. Present for end of 4th quarter. Present for week 1 MAD vs MIN
    # TODO What is the lr attribute? True or False. Present for end of 4th quarter. Present for week 1 MAD vs MIN
    # TODO What is the h attribute? 1 on some scores and 1 block in TB-BOS.
    # TODO Align docstrings with some auto documentation
