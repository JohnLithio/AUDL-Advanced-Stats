AWS_BUCKET_NAME = "audl-heroku-data"
CURRENT_YEAR = 2021
SCHEDULE_URL = "https://theaudl.com/league/schedule"
STATS_URL = "https://audl-stat-server.herokuapp.com/stats-pages/"

# Ratios for heatmap subplots
HEATMAP_RATIO_H_X = 0.85
HEATMAP_RATIO_H_Y = 0.8
HEATMAP_RATIO_V_X = 0.8
HEATMAP_RATIO_V_Y = 0.9

EVENT_TYPES = {
    1: "Start of O-Point",
    2: "Start of D-Point",
    3: "In-bounds Pull",
    4: "Out-of-bounds Pull",
    5: "Block",
    # 6: "", # TODO: What event is this? In 2021-06-04 TB at PHI
    # 7: "", # TODO: What event is this? In 2021-06-04 TB at PHI
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
    # 28: "",
    # 29: "",
    # 30: "",
    # 31: "",
    # 32: "",
    # 33: "",
    # 34: "",
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
    # 45: "", # TODO: What event is this? In 2021-07-09 AUS at DAL
    # 46: "",
    # 47: "",
    # 48: "",
    # 49: "",
    # 50: "",
    # 51: "",
    # 52: "",
    # 53: "",
    # 54: "",
    # 55: "",
    # 56: "",
    # 57: "",
    # 58: "",
    # 59: "",
    # 60: "",
    # 61: "",
    # 62: "",
    # 63: "",
    # 64: "",
    # 65: "",
    # 66: "",
    # 67: "",
    # 68: "",
    # 69: "",
}

EVENT_TYPES_GENERAL = {
    5: "Turnover",
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
