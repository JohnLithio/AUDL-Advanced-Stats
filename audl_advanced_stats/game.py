import pandas as pd
import requests
from ast import literal_eval
from bs4 import BeautifulSoup
from json import loads
from os.path import basename, join
from pathlib import Path
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
        self.home_team = None
        self.home_roster = None
        self.home_events_raw = None
        self.home_events = None  # TODO
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
        events = literal_eval(self.get_response()[events_str]["events"])

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

    def get_events(self, home=True):
        # Set parameters for home or away
        if home:
            events_raw = self.home_events
        else:
            events_raw = self.away_events

        df = pd.DataFrame.from_records(events_raw)
        # event_num, Thrower, receiver, xthrow, ythrow, xcatch, ycatch, goal(y/n), block(y/n), drop(y/n), throwaway(y/n)
        # foul(y/n), travel(y/n), stall(y/n), pull(y/n), hangtime, in-bounds pull(y/n), timeout(y/n), column for every active player (y/n)

