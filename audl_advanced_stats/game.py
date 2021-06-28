import numpy as np
import pandas as pd
import plotly.express as px
import requests
from ast import literal_eval
from bs4 import BeautifulSoup
from json import loads
from os.path import basename, join
from pathlib import Path
from sklearn.cluster import KMeans
from .constants import *
from .utils import (
    get_database_path,
    get_json_path,
    create_connection,
)


class Game:
    def __init__(self, game_url, year=CURRENT_YEAR, database_path="data"):
        """Initial parameters of game data.

        Args:
            game_url (str): URL to get game response data.
            year (int, optional): Season to get stats from. Currently not used because there are only
                advanced stats for a single season (2021).
                Defaults to CURRENT_YEAR.
            database_path (str, optional): The path to the folder where data
                will be stored.

        """
        # Inputs
        self.year = year
        self.game_url = game_url
        self.database_path = get_database_path(database_path)
        self.json_path = get_json_path(database_path, "games")

        # Create directories/databases if they don't exist
        Path(self.json_path).mkdir(parents=True, exist_ok=True)
        conn = create_connection(self.database_path)
        conn.close()

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

    def get_response(self):
        """Get the response of a single game and save it.

        Returns:
            str: The json response string.

        """
        if self.response is None:
            # Get stored response if possible.
            try:
                with open(join(self.json_path, self.get_game_file_name()), "r") as f:
                    response_text = f.read()
            # If file does not exist, get response from url and save it.
            except FileNotFoundError:
                response_text = requests.get(self.game_url).text
                with open(join(self.json_path, self.game_file_name), "w") as f:
                    f.write(response_text)

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
                # Sometimes the event for the start of a d-point is missing
                d_point=lambda x: np.where(
                    x["t"].isin([2]) | ((x["t"].isin([3, 4])) & (x["t"].shift(1) != 2)),
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
                .isin([5, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
                & ~(x["t"].isin([8, 19]) & x["t"].shift(1).isin([8, 19])),
                # Label each possession incrementally
                possession_number=lambda x: np.where(
                    x["possession_change"],
                    x.groupby(["possession_change"]).cumcount() + 2,
                    None,
                ),
                # Set the outcome of each point and possession
                point_outcome=lambda x: np.where(
                    x["t"].isin([21, 22,]), x["t"].map(EVENT_TYPES), None,
                ),
                possession_outcome=lambda x: np.where(
                    x["t"].isin([5, 8, 9, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,])
                    & ~(x["t"].isin([8, 19]) & x["t"].shift(1).isin([8, 19])),
                    x["t"].map(EVENT_TYPES),
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
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (x["point_outcome"] == EVENT_TYPES[22]) & (x["o_point"]),
                    "Hold",
                    "End of Quarter",
                ),
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (x["point_outcome"] == EVENT_TYPES[21]) & (~x["o_point"]),
                    "Opponent Hold",
                    x["point_hold"],
                ),
            )
            # Mark whether each point was a hold, break, or neither
            .assign(
                point_hold=lambda x: np.where(
                    (x["point_outcome"] == EVENT_TYPES[22]) & (~x["o_point"]),
                    "Break",
                    x["point_hold"],
                ),
            )
            .assign(
                point_hold=lambda x: np.where(
                    (x["point_outcome"] == EVENT_TYPES[21]) & (x["o_point"]),
                    "Opponent Break",
                    x["point_hold"],
                ),
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
            / 1000,
        ).drop(columns=["ms"])
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
                # If penalty is within 10 yards of the endzone, center it on the goalline
                x=lambda x: np.where(x["y"] >= 90, 0, x["x"]),
            )
            .rename(columns=lambda x: x + "_new")
        )
        d_penalties = d_penalties.loc[d_penalties.index.isin(df.query("t==[12]").index)]
        df = (
            pd.concat([df, d_penalties], axis=1)
            .assign(
                x=lambda x: x["x"].fillna(x["x_new"]),
                y=lambda x: x["y"].fillna(x["y_new"]),
                r=lambda x: x["r"].fillna(x["r_new"]),
            )
            .drop(columns=["x_new", "y_new", "r_new"])
        )
        return df

    def get_events_yardage(self, df):
        # Get the x,y position for the next throwaway, travel, defensive foul, drop, completion, or score
        next_event = (
            df.query("t==[8,10,12,19,20,22]")
            .groupby(["possession_number"])[["r", "x", "y", "t", "event_name"]]
            .shift(-1)
            .rename(columns=lambda x: x + "_after")
        )
        # Add columns for the next event to each row so we can calculate yardage
        df = pd.concat([df, next_event], axis=1).assign(
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
            yards_raw=lambda x: (x["xyards"].pow(2) + x["yyards_raw"].pow(2)).pow(0.5),
            # Calculate the total distance of the throw, excluding yards in the endzone
            yards=lambda x: (x["xyards"].pow(2) + x["yyards"].pow(2)).pow(0.5),
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
                    & x["period"].isin([5,]),
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
                l=lambda x: x["l"].fillna(method="ffill"),
            )
            # Create a column for each player to indicate if they were on or off the field
            .pipe(self.get_player_columns).drop(columns=["l",])
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
            "Under/Other",
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

    def events_print_qc(self, df, qc=True):
        """Print basic QC info about the processed events data."""
        if qc:
            print("Number of events:", df.shape[0])
            print(
                "New Event Types:",
                ", ".join(
                    sorted(
                        list(set(df["t"].unique()).difference(set(EVENT_TYPES.keys())))
                    )
                ),
            )
            print("Event Attributes:", ", ".join(list(df)[: list(df).index("game_id")]))

        return df

    def get_events(self, home=True, qc=True):
        """Process the events for a single team to get yardage, event labels, etc."""
        # Set parameters for home or away
        if home:
            events_raw = self.get_home_events_raw()
        else:
            events_raw = self.get_away_events_raw()

        df = (
            pd.DataFrame.from_records(events_raw)
            .pipe(self.get_events_basic_info, home=home)
            .pipe(self.get_events_periods)
            .pipe(self.get_events_possession_labels)
            .pipe(self.get_events_pull_info)
            .pipe(self.get_events_o_penalties)
            .pipe(self.get_events_d_penalties)
            .pipe(self.get_events_yardage)
            .pipe(self.get_events_times)
            .pipe(self.get_events_lineups)
            .pipe(self.get_events_throw_classifications)
            .pipe(self.events_print_qc, qc=qc)
        )

        return df

    def get_home_events(self, qc=True):
        """Get processed dataframe of home team events."""
        if self.home_events is None:
            self.home_events = self.get_events(home=True, qc=qc)
        return self.home_events

    def get_away_events(self, qc=True):
        """Get processed dataframe of away team events."""
        if self.away_events is None:
            self.away_events = self.get_events(home=False, qc=qc)
        return self.away_events

    def team_player_clusters(self, times, qc=True):
        """Separate team into 3 clusters (O, D1, D2) based on points played together."""
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

        # Sort each player into 1 of 3 clusters based on which segments they played
        #    3 segments were chosen to mimic the common pattern of O, D1, and D2
        X = psegs.values
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

        # Add clusters to player segment data
        psegs["cluster"] = [str(x) for x in kmeans.labels_]

        # Combine the player time data with the clusters
        times_cluster = times.merge(
            psegs.reset_index()[["playerid", "cluster"]], how="left", on=["playerid"]
        ).sort_values(["cluster"])

        # Get the avg. number of o-point each cluster played and the avg. earliest time they played
        #     Most o-points=O-line
        #     First non-o-line group on the field=D1
        #     Remaining group=D2
        clusters = (
            times_cluster.groupby(["playerid", "cluster"])
            .agg({"s_before_total": "min", "o_point": "sum"})
            .reset_index()
            .groupby(["cluster"])
            .agg({"s_before_total": "mean", "o_point": "mean"})
            .sort_values(["o_point", "s_before_total"], ascending=[False, True])
            .reset_index()
            .assign(cluster_name=["O-Line", "D1", "D2"])
        )

        # Get the number of points each player played, which is used to sort players within each cluster
        playingtime = (
            times_cluster.groupby("playerid")["elapsed"]
            .count()
            .rename("segment_count")
            .reset_index()
        )

        # Combine and sort the data
        times_cluster = (
            times_cluster.merge(
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
            .assign(o_point=lambda x: np.where(x["o_point"], "O-Point", "D-Point"),)
            .reset_index(drop=True)
        )

        if qc:
            print(
                "Segments w/ >7 players:",
                (psegs.drop(columns=["cluster"]).sum() > 7).sum(),
            )

        return times_cluster

    def visual_game_flow(self, color="point_outcome", home=True, qc=True):
        """Gantt chart showing the substitution patterns of a team throughout 1 game."""
        # Monkey patch for plotly so that we don't need to use datetimes for x-axis of gantt
        def my_process_dataframe_timeline(args):
            """Massage input for bar traces for px.timeline()."""
            args["is_timeline"] = True
            if args["x_start"] is None or args["x_end"] is None:
                raise ValueError("Both x_start and x_end are required")

            x_start = args["data_frame"][args["x_start"]]
            x_end = args["data_frame"][args["x_end"]]

            # Note that we are not adding any columns to the data frame here, so no risk of overwrite
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
        ]

        # Get the points that each player was on the field and stack them so
        #    that each player has their own row for every point they played
        dfs = []
        for i, player in roster.iterrows():
            playerid = player["id"]
            # Only get players who played in the game
            if playerid in events:
                df = (
                    events.loc[events[playerid] == 1]
                    .groupby(["period", "s_before", "s_after"])
                    .head(1)
                    .query("elapsed!=0")
                    .assign(
                        playerid=str(playerid),
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

        # Group players into O-line, D1, and D2
        times_cluster = self.team_player_clusters(times, qc=qc)

        # Get the order in which we want to show the players on the y-axis
        name_order = times_cluster.groupby(["name"]).head(1)["name"].values
        point_hold_order = [
            "Hold",
            "Opponent Break",
            "Opponent Hold",
            "Break",
            "End of Quarter",
        ]
        point_outcome_order = [
            "Score",
            "Opponent Score",
            "End of Period",
        ]
        o_point_order = [
            "O-Point",
            "D-Point",
        ]

        # Create the figure
        fig = px.timeline(
            times_cluster,
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
        )
        fig.layout.xaxis.type = "linear"

        # Set the labels for the x-axis tick marks
        xticks = {
            0: "Game Start",
            60 * 12 * 1: "End of Q1",
            60 * 12 * 2: "End of Q2",
            60 * 12 * 3: "End of Q3",
            60 * 12 * 4: "End of Q4",
        }

        # Add OT if there are events that take place beyond 4 quarters
        if times_cluster["s_after_total"].max() > 4 * 12 * 60:
            xticks[60 * 12 * 4 + 60 * 5 * 1] = "End of OT1"

        # Create second x-axis that is only for timeouts and injuries
        x2ticks = dict()

        # Mark timeouts
        for i, row in events.query("t==[14, 15]").iterrows():
            x2ticks[row["s_total"]] = "Timeout"

        # Mark injuries
        for i, row in events.query("t==[42, 43]").iterrows():
            x2ticks[row["s_total"]] = "Injury"

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
            ),
            # Remove y axis title
            yaxis=dict(title=None,),
            # Remove legend title
            legend_title=None,
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
        )

        # Set the ranges for shading the chart by cluster
        cluster_shading_ranges = (
            times_cluster.groupby(["name"])
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

        # Show info about o-point/d-point
        # Mark the end of periods and give option to mark timeouts and injuries
        # Align another graph on the x-axis that shows that scoring progression?
        # On hover, show start and end time of possession, total time, outcome, possession numbers, player stats?
        #     player stats could be completions, receptions, yards, Ds, whether they scored
        # TODO: Clean up hover text
        # TODO: Option for 2 clusters instead of 3?
        # TODO: Change colors
        # TODO: Try annotating graph to label O, D1, D2
        return fig

    def get_player_stats(self):
        # TODO: Identify and remove yardage from centering passes
        # team
        # Number
        # Points played
        # O points played
        # D points played
        # O points ending in goal/total o points (ignoring turns)
        # D points ending in goal for other team/total d points (ignoring turns)
        # Pts on field for offensive score/pts on field for o-poss that ended in score or turn
        # Pts on field for defensive scored on/pts on field for d-poss that ended in score or turn
        # AST (and per offensive possession, minute)
        # GLS (and per offensive possession, minute)
        # BLK (and per defensive possession, minute)
        # +/- (and per offensive possession, minute)
        # completions (and per offensive possession, minute)
        # Throwaways (and per offensive possession, minute)
        # Throw attempts (and per offensive possession, minute)
        # Cmp%
        # receptions (and per offensive possession, minute)
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
        # Hockey assists
        # Stalls
        # Ds
        # Callahans
        # Time played
        pass

    def get_team_stats(self):
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
        pass

        # Do not count as 2 changes of possession if block, throwaway, score
        #   occurred with 0 seconds left (1Q DC at NY)
        # TODO What is the q attribute? It's 1 sometimes. In MIN-MAD, it was 1 for a score where the x and y vals were way off. q=questionable stat-keeping? Present on own-team score. TODO
        # TODO What is the c attribute?  True or False. Present for opponent foul. Might be True when the disc gets centered. Present for week 1 MAD vs MIN TODO
        # TODO What is the o attribute?  True or False. Possibly true/false for OT. Present for end of 4th quarter. Present for week 1 MAD vs MIN TODO
        # TODO What is the lr attribute? True or False. Present for end of 4th quarter. Present for week 1 MAD vs MIN
        # TODO What is the h attribute? 1 on some scores and 1 block in TB-BOS.

        # MATCHING UP POSSESSIONS
        # Run into issues at the end of quarters
        #     If a drop occurs, then the other team completes a pass but
        #     does not turn or score it (2Q IND at DET)
        #     But if there's a heave, one team might get credit
        #     for a block while the other team does not get a throwaway
        # Events with x/y: completion, throwaway, score, OB/IB pull (defense), drop
        #     Set xend/yend, then adjust xstart/ystart for penalties/travels
        # Each row should be event:
        #     completion, throwaway, travel, penalty, stall, score, timeout, end of quarter, pull, injury, etc.
        # and who to attribute it to on that team:
        #     thrower, receiver, fouler/traveler, puller
        # and how many x/y/total yards it accounted for
        # and who was on the field
        # and the most recent time and closest following time
