import pandas as pd
import requests
import sqlite3
from ast import literal_eval
from bs4 import BeautifulSoup
from json import loads
from os.path import basename, join
from pathlib import Path
from .constants import *
from .game import Game
from .utils import get_database_path, get_json_path, create_connection


class Season:
    """This class contains methods for retrieving all of the advanced AUDL stats for a single season."""

    def __init__(self, year=CURRENT_YEAR, database_path="data"):
        """Initialize parameters of season data.

        Args:
            year (int, optional): Season to get stats from. Currently not used because there are only
                advanced stats for a single season (2021).
                Defaults to CURRENT_YEAR.
            database_path (str, optional): The path to the folder where data
                will be stored.

        """
        # Inputs
        self.year = year
        self.database_path = get_database_path(database_path)
        self.json_path = get_json_path(database_path, "games")

        # Create directories/databases if they don't exist
        Path(self.json_path).mkdir(parents=True, exist_ok=True)
        conn = create_connection(self.database_path)
        conn.close()

        # URLs to retrieve data from
        self.schedule_url = SCHEDULE_URL
        self.stats_url = STATS_URL
        self.weeks_urls = None
        self.game_urls = None

    def get_weeks_urls(self):
        """Get URLs for the schedule of each week of the season."""
        if self.weeks_urls is None:
            # Get urls for all weeks
            schedule_page = requests.get(self.schedule_url)
            schedule_soup = BeautifulSoup(schedule_page.text, "html.parser")

            # Extract urls from document
            weeks = []
            for week in schedule_soup.find_all("a"):
                if "schedule" in week.get("href").lower():
                    week_url = self.schedule_url + week.get("href")[16:]
                    weeks.append(week_url)

            self.weeks_urls = weeks

        return self.weeks_urls

    def get_game_urls(self, override=False):
        """Get URLs for the advanced stats page of every game in the season."""
        if self.game_urls is None:
            # Check if URLs already exist in DB
            games = []
            try:
                query = """
                select distinct url
                from game_urls
                """
                conn = create_connection(self.database_path)
                games = pd.read_sql(sql=query, con=conn)["url"].unique().tolist()
                conn.close()
            except:
                conn.close()

            # Retrieve URLs from website if they are not in DB or override is True
            if len(games) == 0 or override:
                print("going online")
                # Get all games in all weeks
                games = []
                for week_url in self.get_weeks_urls():
                    # Get schedule for 1 week
                    week_page = requests.get(week_url)
                    week_soup = BeautifulSoup(week_page.text, "html.parser")

                    # Get all game URLs in 1 week
                    for game_center in week_soup.find_all(
                        "span", {"class": "audl-schedule-gc-link"}
                    ):
                        game_url = (
                            self.stats_url + game_center.find("a").get("href")[8:]
                        )
                        games.append(game_url)

            self.game_urls = sorted(games)

        return self.game_urls

    def save_game_urls(self):
        """Save all game urls into a database."""
        df = pd.DataFrame(
            data=[[x] for x in self.get_game_urls(override=True)], columns=["url"],
        )

        conn = create_connection(self.database_path)
        df.to_sql("game_urls", con=conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()


# Create database with league info
#    Team rosters (ID and names - other info too? Height/weight? Number?)
#    Teams (ID and team names)
# Get raw response from URL
# Parse/encode response
# Re-format response
# Save response
