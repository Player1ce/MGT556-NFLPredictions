import numpy as np
import polars as pl
import pandas as pd

import nflreadpy as nfl

from lib import preset_selections as presets
from lib import file_utils as futils


def get_player_fantasy_sheet(
        start_season=2016,
        end_season=2025,
        positions_to_retrieve=presets.player_fantasy_positions,
        cols_to_retrieve=presets.player_fantasy_cols
) -> pl.LazyFrame:
    # Shared variables

    # seasons to retrieve data for
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    # Player Data

    # create a query to get the relevant data
    get_relevant_player_data = (nfl.load_player_stats(
        seasons=seasons_to_retrieve,
        summary_level="reg"
    )[cols_to_retrieve].lazy()
                                .filter(pl.col('position').is_in(positions_to_retrieve))
                                .sort(['season', 'position'])
                                )

    return get_relevant_player_data


def get_team_fantasy_sheet(
        start_season=2016,
        end_season=2025,
        cols_to_retrieve=presets.team_fantasy_dst_cols
) -> pl.LazyFrame:
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    # Team Data

    # retrieve and export team data for chosen seasons
    get_relevant_team_data = (
        nfl.load_team_stats(
            seasons=seasons_to_retrieve,
            summary_level="reg"
        )[cols_to_retrieve].lazy()
        .sort(['season'])
        .with_columns(pl.lit('DST').alias('position'))
    )

    return get_relevant_team_data


def get_season_aggregate_data(
        start_season=2016,
        end_season=2025
) -> pl.LazyFrame:
    data = futils.read_source_frame("SeasonData", futils.FileType.EXCEL)
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    return (data.lazy()
            .rename({
        'Season': 'season',
        'NumGames': 'num_games',
        'NumTeams': 'num_teams',
        'AFC Champion': 'afc_champion',
        'NFC Champion': 'nfc_champion',
        'Game': 'game',
        'Champion': 'champion',
    })
            .filter(pl.col('season').is_in(seasons_to_retrieve))
            .sort(['season'])
            )
