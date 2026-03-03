# Player Data

min_start_year = 1999
max_end_year = 2025

# player positions needed for fantasy scoring metrics
player_fantasy_positions = ['TE', 'QB', 'WR', 'K', 'RB']

position_groups = {
    "skill": ["QB", "RB", "WR"],  # Premium starters
    "pass_catch": ["RB", "WR", "TE"],  # PPR focus
    "te": ["TE"],  # Premium TE only
    "kicker": ["K"],
    "defense": ["DST"],  # D/ST or DST
    "flex": ["RB", "WR", "TE"]  # Flex eligibility
}

# fantasy scoring fields (with a few extra for info)
player_fantasy_cols = [
    # Player identification
    'player_id', 'player_display_name', 'position', 'recent_team', 'season',

    # Passing (QB)
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'passing_2pt_conversions',

    # Rushing (RB/QB)
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions',

    # Receiving (WR/RB/TE) - PPR CRITICAL
    'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
    'receiving_2pt_conversions',

    # Kicking
    'fg_made', 'fg_missed', 'pat_made', 'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49',
    'fg_made_50_59', 'fg_made_60_',

    # ALREADY CALCULATED (use these directly)
    'fantasy_points', 'fantasy_points_ppr'
]

# Team Data
# scoring categories for team data (defense and special teams in this case)
team_fantasy_dst_cols = [
    # IDENTIFICATION
    'season', 'team', 'games',

    # SACKS
    'def_sacks',

    # TURNOVERS
    'def_interceptions', 'def_fumbles_forced', 'fumble_recovery_opp',

    # TDs (Defense/Special Teams)
    'def_tds', 'special_teams_tds', 'fumble_recovery_tds',

    # SAFETIES
    'def_safeties',
]

# Combined Data
combined_fantasy_columns = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
    # rushing
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    # receiving
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
    "receiving_fumbles_lost",
    # kicking
    "pat_made",
    "fg_missed",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    # defense / special teams
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
    "special_teams_tds",
    "def_safeties",
]
