import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
from ast import literal_eval
from bs4 import BeautifulSoup
from json import loads
from os.path import basename, join
from pathlib import Path
from plotly.subplots import make_subplots
from re import search
from .constants import *
from .game import Game
from .utils import get_database_path, get_json_path, get_games_path, create_connection


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
        self.games_path = get_games_path(database_path, "all_games")

        # Create directories/databases if they don't exist
        Path(self.database_path).mkdir(parents=True, exist_ok=True)
        Path(self.json_path).mkdir(parents=True, exist_ok=True)
        Path(self.games_path).mkdir(parents=True, exist_ok=True)
        conn = create_connection(self.database_path)
        conn.close()

        # URLs to retrieve data from
        self.schedule_url = SCHEDULE_URL
        self.stats_url = STATS_URL
        self.weeks_urls = None
        self.game_info = None

        # All processed game data
        self.games = None

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

    def get_game_info(self, override=False):
        """Get teams, date, and url for the advanced stats page of every game in the season."""
        if self.game_info is None:
            # Check if games already exist in DB
            retrieved_games = False
            try:
                query = """
                select *
                from games
                """
                conn = create_connection(self.database_path)
                df = pd.read_sql(sql=query, con=conn)
                conn.close()
                retrieved_games = True
            except:
                conn.close()

            # Retrieve games from website if they are not in DB or override is True
            if not retrieved_games or override:
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

                    # Parse URLs to get game info
                    game_list = [
                        [
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(1),
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(2),
                            search(r"(\d{4}-\d{2}-\d{2})-(.*?)-(.*?)$", x).group(3),
                            x,
                        ]
                        for x in sorted(games)
                    ]

                    # Save info to database table
                    df = (
                        pd.DataFrame(
                            data=game_list,
                            columns=["game_date", "away_team", "home_team", "url"],
                        )
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                    conn = create_connection(self.database_path)
                    df.to_sql("games", con=conn, if_exists="replace", index=False)
                    conn.commit()
                    conn.close()

            self.game_info = df

        return self.game_info

    def get_games(self, small_file=False, build_new_file=False, qc=True):
        """Download and process all game data."""
        if self.games is None:

            file_name_small = join(self.games_path, f"all_games_small.feather")
            file_name = join(self.games_path, f"all_games.feather")
            # Get either the file with all columns or only some
            if small_file:
                all_games_file_name = file_name_small
            else:
                all_games_file_name = file_name

            # Check if file already exists
            if Path(all_games_file_name).is_file() and not build_new_file:
                self.games = pd.read_feather(all_games_file_name)

            # Compile data if file does not already exist
            else:
                all_games = []
                for i, row in self.get_game_info(override=build_new_file).iterrows():
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
                        g = Game(game_url=row["url"])
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

                needed_columns = [
                    "game_id",
                    "team_id",
                    "opponent_team_id",
                    "t",
                    "t_after",
                    "r",
                    "r_after",
                    "x",
                    "y",
                    "x_after",
                    "y_after",
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
                self.games[needed_columns].reset_index(drop=True).to_feather(
                    file_name_small
                )

        return self.games

    def visual_field_heatmap_horizontal(
        self,
        outcome_measure,
        outcome,
        metric,
        x_cut=None,
        y_cut=None,
        o_point=None,
        remove_ob_pull=False,
        throw=True,
        team_ids=None,
        player_ids=None,
        game_ids=None,
    ):
        """View frequency of possession, scores, turns on the field, similar to shot chart."""
        df = self.get_games(small_file=True, build_new_file=False, qc=False,)

        # Run through initial filters
        if team_ids is not None:
            df = df.loc[df["team_id"].isin(team_ids)]
        if player_ids is not None:
            df = df.loc[df["player_id"].isin(player_ids)]
        if game_ids is not None:
            df = df.loc[df["game_id"].isin(game_ids)]
        if remove_ob_pull:
            df = df.loc[~((df["x"] == 0) & (df["y"] == 40))]
        if o_point is None:
            o_point = "o_point"

        # Set whether heat map should be for the throw or the catch
        if throw:
            suffix = ""
            opposite_suffix = "_after"
        else:
            suffix = "_after"
            opposite_suffix = ""

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
        # TODO: Do this for yardages too?
        kwargs = dict(reversescale=False)
        if metric == "pct":
            kwargs["colorbar_tickformat"] = ".0%"
            if outcome_measure == "throw_outcome":
                if outcome == "Completion":
                    kwargs["zmin"] = 0.8
                    kwargs["zmax"] = 1.0
                elif outcome == "Turnover":
                    kwargs["zmin"] = 0.0
                    kwargs["zmax"] = 0.2
            elif outcome_measure == "possession_outcome_general":
                if outcome == "Score":
                    kwargs["zmin"] = 0.2
                    kwargs["zmax"] = 1.0
                elif outcome == "Turnover":
                    kwargs["zmin"] = 0.0
                    kwargs["zmax"] = 0.8

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
        height = 530
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
            # # Set figure size
            # height=height,
            # width=height * 120 / 54,
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

        hyratio = df.groupby(["y_cut_final"])["count"].sum().max()
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
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                # scaleanchor="x",
                # scaleratio=1 / hyratio,
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
            # # Set figure size
            # height=100,
            # width=height * 120 / 54,
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

        hxratio = df.groupby(["x_cut_final"])["count"].sum().max()
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
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                autorange="reversed",
                fixedrange=True,
                # scaleanchor="x",
                # scaleratio=hxratio,
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
            # # Set figure size
            # height=height - 34,
            # width=100,
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
        o_point=None,
        remove_ob_pull=False,
        throw=True,
        team_ids=None,
        player_ids=None,
        game_ids=None,
    ):
        """View frequency of possession, scores, turns on the field, similar to shot chart."""
        df = self.get_games(small_file=True, build_new_file=False, qc=False,)

        # Run through initial filters
        if team_ids is not None:
            df = df.loc[df["team_id"].isin(team_ids)]
        if player_ids is not None:
            df = df.loc[df["player_id"].isin(player_ids)]
        if game_ids is not None:
            df = df.loc[df["game_id"].isin(game_ids)]
        if remove_ob_pull:
            df = df.loc[~((df["x"] == 0) & (df["y"] == 40))]
        if o_point is None:
            o_point = "o_point"

        # Set whether heat map should be for the throw or the catch
        if throw:
            suffix = ""
            opposite_suffix = "_after"
        else:
            suffix = "_after"
            opposite_suffix = ""

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
        # TODO: Do this for yardages too?
        kwargs = dict(reversescale=False)
        if metric == "pct":
            kwargs["colorbar_tickformat"] = ".0%"
            if outcome_measure == "throw_outcome":
                if outcome == "Completion":
                    kwargs["zmin"] = 0.8
                    kwargs["zmax"] = 1.0
                elif outcome == "Turnover":
                    kwargs["zmin"] = 0.0
                    kwargs["zmax"] = 0.2
            elif outcome_measure == "possession_outcome_general":
                if outcome == "Score":
                    kwargs["zmin"] = 0.2
                    kwargs["zmax"] = 1.0
                elif outcome == "Turnover":
                    kwargs["zmin"] = 0.0
                    kwargs["zmax"] = 0.8

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
        height = 530
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
            # # Set figure size
            # height=height,
            # width=height * 120 / 54,
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

        hyratio = df.groupby(["y_cut_final"])["count"].sum().max()
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
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                # scaleanchor="x",
                # scaleratio=1 / hyratio,
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
            # # Set figure size
            # height=100,
            # width=height * 120 / 54,
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

        hxratio = df.groupby(["x_cut_final"])["count"].sum().max()
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
                # range=[-27, 30],
                showticklabels=False,
                ticks="",
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                # scaleanchor="x",
                # scaleratio=hxratio,
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
            # # Set figure size
            # height=height - 34,
            # width=100,
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

    def visual_field_heatmap_subplots_horizontal(self, fighm, fighx, fighy):
        """Combine heatmap and histograms into single plot."""

        hmratioy = 0.8
        hmratiox = 0.85
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[hmratiox, 1 - hmratiox],
            row_heights=[1 - hmratioy, hmratioy],
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
        )
        fig.add_trace(go.Heatmap(fighm.data[0]), row=2, col=1)
        fig.add_trace(go.Histogram(fighx.data[0]), row=2, col=2)
        fig.add_trace(go.Histogram(fighy.data[0]), row=1, col=1)
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
            y1=hmratioy,
            x0=0,
            x1=0,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=hmratioy,
            x0=hmratiox / 6,
            x1=hmratiox / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=hmratioy,
            x0=hmratiox * 5 / 6,
            x1=hmratiox * 5 / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=-0,
            y1=hmratioy,
            x0=hmratiox,
            x1=hmratiox,
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
            x1=hmratiox,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            y0=hmratioy,
            y1=hmratioy,
            x0=0,
            x1=hmratiox,
            line=dict(color="black"),
        )

        # Adjust colorbar
        fig.data[0]["colorbar"]["len"] = hmratioy
        fig.data[0]["colorbar"]["y"] = hmratioy

        return fig

    def visual_field_heatmap_subplots_vertical(self, fighm, fighx, fighy):
        """Combine heatmap and histograms into single plot."""

        hmratioy = 0.85
        hmratiox = 0.8
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[hmratiox, 1 - hmratiox],
            row_heights=[1 - hmratioy, hmratioy],
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
        )
        fig.add_trace(go.Heatmap(fighm.data[0]), row=2, col=1)
        fig.add_trace(go.Histogram(fighx.data[0]), row=2, col=2)
        fig.add_trace(go.Histogram(fighy.data[0]), row=1, col=1)
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
            x1=hmratiox,
            y0=0,
            y1=0,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=hmratiox,
            y0=hmratioy / 6,
            y1=hmratioy / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=hmratiox,
            y0=hmratioy * 5 / 6,
            y1=hmratioy * 5 / 6,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=0,
            x1=hmratiox,
            y0=hmratioy,
            y1=hmratioy,
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
            y1=hmratioy,
            line=dict(color="black"),
        )
        fig.add_shape(
            xref="paper",
            yref="paper",
            type="line",
            x0=hmratiox,
            x1=hmratiox,
            y0=0,
            y1=hmratioy,
            line=dict(color="black"),
        )

        # Adjust colorbar
        fig.data[0]["colorbar"]["len"] = hmratioy
        fig.data[0]["colorbar"]["y"] = hmratioy
        fig.data[0]["colorbar"]["x"] = 0

        return fig
