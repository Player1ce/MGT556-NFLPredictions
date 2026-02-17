import nflreadpy as nfl
import pathlib
import polars as pl

def get_player_fantasy_sheet(start_season=2016, end_season=2025) -> pl.DataFrame:
    # Shared variables

    # seasons to retrieve data for
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    # player positions needed for fantasy scoring metrics
    fantasy_relevant_positions = ['TE', 'QB', 'WR', 'K', 'RB']

    # fantasy scoring fields (with a few extra for info)
    fantasy_cols = [
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

    # retrieval choices
    positions_to_retrieve = fantasy_relevant_positions
    cols_to_retrieve = fantasy_cols

    # create a query to get the relevant data
    get_relevant_player_data = (nfl.load_player_stats(
        seasons=seasons_to_retrieve,
        summary_level="reg"
    )[cols_to_retrieve].lazy()
                                .filter(pl.col('position').is_in(positions_to_retrieve))
                                .sort(['season', 'position'])
                                )

    return get_relevant_player_data.collect()

def get_team_fantasy_sheet(start_season=2016, end_season=2025) -> pl.DataFrame:
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    # team data
    # scoring categories for team data (defense and special teams in this case)
    fantasy_dst_cols = [
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


    # Team Data
    # retrieval choices
    cols_to_retrieve = fantasy_dst_cols

    # retrieve and export team data for chosen seasons
    get_relevant_team_data = (
        nfl.load_team_stats(
            seasons=seasons_to_retrieve,
            summary_level="reg"
        )[cols_to_retrieve].lazy()
        .sort(['season'])
    )

    return get_relevant_team_data.collect()



def get_combined_data(start_season=2016, end_season=2025) -> pl.DataFrame:
    # Shared variables


    player_data = get_player_fantasy_sheet(start_season, end_season)
    team_data = get_team_fantasy_sheet(start_season, end_season)
    frames = [player_data, team_data]

    frames = [
        frame.rename({'recent_team': 'team'})
        if 'recent_team' in frame.columns
        else frame
        for frame in frames
    ]

    # combine files
    combined_frame = (pl.concat(frames, how='diagonal_relaxed')
                      .fill_null(strategy='zero'))

    # print(f"nulls: {combined_frame.null_count().sum_horizontal()}")

    # for column in combined_frame.columns:
    #     print(f"Column: {column} | Dtype: {combined_frame[column].dtype}")

    return combined_frame