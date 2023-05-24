"""Global constants to be used in the game and season files."""

# Don't need this if you're just going to store the data locally
BUCKET_NAME = "audl-heroku-data"
YEARS = [2023, 2022, 2021]
CURRENT_YEAR = YEARS[0]

# URL to scrape all the games
SCHEDULE_URL = "https://www.backend.audlstats.com/web-api/games?limit={games_per_page}&years={year}&page={page_num}"

# Endpoint for getting advanced stats data
STATS_URL = "https://www.backend.audlstats.com/stats-pages/"

# Ratios for heatmap subplots
HEATMAP_RATIO_H_X = 0.85
HEATMAP_RATIO_H_Y = 0.8
HEATMAP_RATIO_V_X = 0.8
HEATMAP_RATIO_V_Y = 0.9

# Playoff game IDs
PLAYOFF_GAMES = {
    2021: [2794, 2795, 2796, 2797, 2799, 2800, 2801, 2802],
    2022: [2963, 2962, 2961, 2960, 2959, 2958, 2955, 2956, 2953, 2954],
}

# All-star game IDs
ALLSTAR_GAMES = [
    2964,
]

# List of event type encodings from what I figured out.
# Should confirm with Ben Nelson about this though.
EVENT_TYPES = {
    1: "Start of O-Point",
    2: "Start of D-Point",
    3: "In-bounds Pull",
    4: "Out-of-bounds Pull",
    5: "Block",
    6: "Callahan",
    7: "Opponent Callahan",
    8: "Throwaway",
    9: "Throwaway Caused",
    10: "Travel",
    11: "Opponent Travel",
    12: "Opponent Foul",
    13: "Own Foul",
    14: "Timeout",
    15: "Opponent Timeout",
    # 16: "",
    17: "Stall",
    18: "Opponent Stall",
    19: "Drop",
    20: "Completion",
    21: "Opponent Score",
    22: "Score",
    23: "End of 1st Quarter",
    24: "End of 2nd Quarter",
    25: "End of 3rd Quarter",
    26: "End of 4th Quarter",
    27: "End of 1st Overtime",
    28: "End of 2nd Overtime",
    # 29: "",
    # 30: "",
    # 31: "",
    # 32: "",
    # 33: "",
    34: "Unsportsmanlike Conduct",
    # 35: "",
    # 36: "",
    # 37: "",
    # 38: "",
    # 39: "",
    40: "Substitutions",
    41: "Substitutions",
    42: "Opponent Injury",
    43: "Injury",
    44: "Offsides",
    45: "Opponent Offsides",
    # 46: "",
    # 47: "",
    # 48: "",
    # 49: "",
    50: "Start of Game",
    51: "Weather - Delayed",
    52: "Weather - Resumed",
    53: "Play Resumed",
    54: "Play Stopped",
}

# General descriptions of the end of possessions
EVENT_TYPES_GENERAL = {
    5: "Turnover",
    6: "Score",
    7: "Turnover",
    8: "Turnover",
    9: "Turnover",
    17: "Turnover",
    18: "Turnover",
    19: "Turnover",
    21: "Score",
    22: "Score",
    23: "End of Period",
    24: "End of Period",
    25: "End of Period",
    26: "End of Period",
    27: "End of Period",
}
