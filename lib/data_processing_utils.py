import math
import pathlib
from math import nan

import requests
from bs4 import BeautifulSoup

import numpy as np
import polars as pl
import pandas as pd

import nflreadpy as nfl

from lib import preset_selections as presets


def get_player_fantasy_sheet(
        start_season=2016,
        end_season=2025,
        positions_to_retrieve=presets.player_fantasy_positions,
        cols_to_retrieve=presets.player_fantasy_cols
) -> pl.DataFrame:
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

    return get_relevant_player_data.collect()


def get_team_fantasy_sheet(
        start_season=2016,
        end_season=2025,
        cols_to_retrieve=presets.team_fantasy_dst_cols
) -> pl.DataFrame:
    seasons_to_retrieve = list(range(start_season, end_season + 1))

    # Team Data

    # retrieve and export team data for chosen seasons
    get_relevant_team_data = (
        nfl.load_team_stats(
            seasons=seasons_to_retrieve,
            summary_level="reg"
        )[cols_to_retrieve].lazy()
        .sort(['season'])
    )

    return get_relevant_team_data.collect()

def add_id_col(frame : pl.DataFrame) -> pl.DataFrame:

    assert all(col in frame.columns for col in ['player_id', 'team'])

    # add a single id column to act as a combined key with season
    return (frame.with_columns(
        pl.when(pl.col('player_id').str.len_chars() > 0)
        .then(pl.col('player_id'))
        .otherwise(pl.col('team'))
        .alias('id')
    ))



def get_combined_data(
        start_season=2016,
        end_season=2025,
        player_positions_to_retrieve=presets.player_fantasy_positions,
        player_cols_to_retrieve=presets.player_fantasy_cols,
        team_cols_to_retrieve=presets.team_fantasy_dst_cols
) -> pl.DataFrame:
    # Shared variables

    player_data = get_player_fantasy_sheet(
        start_season,
        end_season,
        player_positions_to_retrieve,
        player_cols_to_retrieve
    )
    team_data = get_team_fantasy_sheet(
        start_season,
        end_season,
        team_cols_to_retrieve
    )

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

    combined_frame = add_id_col(combined_frame)

    # print(f"nulls: {combined_frame.null_count().sum_horizontal()}")

    # for column in combined_frame.columns:
    #     print(f"Column: {column} | Dtype: {combined_frame[column].dtype}")

    return combined_frame


# TODO: adjust scoring to avg score and account for season games change (2022?)
def add_fantasy_scoring_inplace(frame):
    # create scoring column

    assert all(col in frame.columns for col in presets.combined_fantasy_columns)

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

    ).clip(lower_bound=0).alias("class_ppr_score")

    frame.insert_column(len(frame.columns), class_scoring_expression)

def add_log_column(frame, column):
    frame.insert_column(
        len(frame.columns),
        pl.when(pl.col(column) > 0)
        .then(np.log1p(pl.col(column)))
        .when(pl.col(column) <= 0)
        .then(0)
        .otherwise(None)
        .alias(f'log1p_{column}')
    )

def shift_column_by_season(
        frame : pl.DataFrame,
        col_name : str,
        shift = -1,
        id_col='id',
        null_replacement=None,
        nan_replacement=nan
) -> pl.DataFrame:
    '''
    Shifts the specified column forward or backward shift number of seasons and puts the result in a new column named future_col_name or past_col_name.
    Requires an ID col to exist in the data to use as a key along with season for matching
    If abs(shift) > 1: The new name will have past{abs(shift)} or future{abs(shift)} to indicate how many times it is shifted.
    :param frame: the dataframe to shift a column in
    :param col_name: the name of the column to copy and shift
    :param shift: The direction and number of seasons to shift by -1 is back 1 season 3 is forward 3 seasons
    :param id_col: the col to use as an id for matching along with season
    :param null_replacement: Optional replacement for None values in the resulting column
    :param nan_replacement: Optional replacement for NaN values in the resulting column
    :return: The original frame with the additional shifted column
    '''

    assert shift != 0

    prefix = 'future'
    if shift > 0:
        prefix = 'past'

    if abs(shift > 1):
        prefix = f'{prefix}{abs(shift)}'

    season_shift = frame.select([
        pl.col(id_col),
        (pl.col("season") + shift),
        pl.col(col_name).alias(f'{prefix}_{col_name}'),
    ]
    )

    result = frame.join(
        season_shift,
        on=[pl.col('id'), pl.col("season")],
        how='left'
    )

    if not math.isnan(nan_replacement):
        result = result.fill_nan(nan_replacement)

    if null_replacement is not None:
        result = result.fill_null(null_replacement)

    return result


def output_result_frame(frame: pl.DataFrame, name: str):
    # output as parquet, csv, and excel
    print(frame.write_csv(pathlib.Path.cwd() / "output" / str(name + ".csv"), include_header=True))
    print(frame.write_excel(pathlib.Path.cwd() / "output" / str(name + ".xlsx"), include_header=True))
    print(frame.write_parquet(pathlib.Path.cwd() / "output" / str(name + ".parquet")))

def output_source_frame(frame: pl.DataFrame, name: str):
    # output as parquet, csv, and excel
    print(frame.write_csv(pathlib.Path.cwd() / "data" / str(name + ".csv"), include_header=True))
    print(frame.write_parquet(pathlib.Path.cwd() / "data" / str(name + ".parquet")))



def get_num_games_season_wikepedia():
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_NFL_seasons')
    return tables[3]

def left_join_from_excel_source(frame: pl.DataFrame, source_name: str):
    source = pl.read_excel(pathlib.Path.cwd() / "data" / str(source_name + ".xlsx"))



if __name__ == '__main__':

    url = 'https://en.wikipedia.org/wiki/List_of_NFL_seasons'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find Wikipedia tables (they have specific classes)
    tables = soup.find_all('table', class_='wikitable')
    print(f"Found {len(tables)} wikitable(s)")

    # Extract first NFL seasons table
    table = tables[2]  # Adjust index as needed
    rows = []

    # for index, table in enumerate(tables):
    #     df_table = pl.from_pandas(pd.read_html(table.text)[0])
    #     print("Index:", index, "Shape:", table.shape)
    #     print(df_table.head())

    for row in table.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td'])]

        if cells:  # Skip empty rows

            rows.append(cells)

    df = pd.DataFrame(rows[1:], columns=rows[0])  # First row = headers
    print(df.head())
    print(df.shape)
