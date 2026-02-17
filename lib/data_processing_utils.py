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

    combined_frame = (combined_frame.lazy().with_columns(
        pl.when(pl.col('player_id').str.len_chars() > 0)
        .then(pl.col('player_id'))
        .otherwise(pl.col('team'))
        .alias('id')
    )).collect()

    # print(f"nulls: {combined_frame.null_count().sum_horizontal()}")

    # for column in combined_frame.columns:
    #     print(f"Column: {column} | Dtype: {combined_frame[column].dtype}")

    return combined_frame


def add_fantasy_scoring_inplace(frame):
    # create scoring column

    class_scoring_expression = (
        # passing
            pl.col("passing_yards") / 25
            + pl.col("passing_tds") * 4
            + pl.col("passing_interceptions") * -2
            + pl.col("passing_2pt_conversions") * 2
            # rushing
            + pl.col("rushing_yards") / 10
            + pl.col("rushing_tds") * 6
            + pl.col("rushing_fumbles_lost") * -2
            + pl.col("rushing_2pt_conversions") * 2
            # receiving
            + pl.col("receptions") * 1
            + pl.col("receiving_yards") / 10
            + pl.col("receiving_tds") * 6
            + pl.col("receiving_2pt_conversions") * 2
            + pl.col("receiving_fumbles_lost") * -2
            # kicking
            + pl.col("pat_made")
            + pl.col("fg_missed") * -1
            + pl.col("fg_made_0_19") * 3
            + pl.col("fg_made_20_29") * 3
            + pl.col("fg_made_30_39") * 3
            + pl.col("fg_made_40_49") * 4
            + pl.col("fg_made_50_59") * 5
            + pl.col("fg_made_60_") * 5
            # defense / special teams
            + pl.col("def_sacks")
            + pl.col("def_interceptions") * 2
            + pl.col("fumble_recovery_opp") * 2
            + pl.col("def_tds") * 6
            + pl.col("special_teams_tds") * 6
            + pl.col("def_safeties") * 2
        # TODO: use data aggregation to generate points allowed statistics NOTE: points allowed would require much data aggregation so it is left as a todo for now
    ).alias("class_ppr_score")

    frame.insert_column(len(frame.columns), class_scoring_expression)
    return


def output_frame(frame : pl.DataFrame, path : pathlib.Path):
    # output as parquet and csv
    print(frame.write_csv(str(path) + ".csv", include_header=True))
    print(frame.write_excel(str(path) + ".xlsx", include_header=True))
    print(frame.write_parquet(str(path) + ".parquet"))