from typing import Sequence

import requests
from bs4 import BeautifulSoup

import numpy as np
import polars as pl
import pandas as pd

import nflreadpy as nfl
from polars import Expr
from polars._typing import JoinStrategy

from lib import preset_selections as presets
from lib import file_utils as futils
from lib import data_fetching_utils as df_utils


def debug_pipeline(pipeline: pl.LazyFrame, stage: str = "final"):
    print(f"\n=== {stage} ===")
    print("PLAN:", pipeline.explain(optimized=True))
    return pipeline


def collect_and_print_profile(pipeline: pl.LazyFrame) -> pl.DataFrame:
    result, profile = pipeline.profile()

    # print a pretty summary of the performance of the operations

    with pl.Config(tbl_rows=-1):
        print(profile.with_columns((pl.col('end') - pl.col('start')).alias('duration_ms')))

    return result


def add_id_col(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    # assert all(col in frame.columns for col in ['player_id', 'team'])

    # add a single id column to act as a combined unique key with season
    return (frame.lazy().with_columns(
        pl.when(pl.col('player_id').str.len_chars() > 0)
        .then(pl.col('player_id'))
        .otherwise(pl.col('team'))
        .alias('id')
    ))


def add_key_col(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return (frame.lazy().with_columns(
        pl.when(pl.col('player_id').str.len_chars() > 0)
        .then(pl.col('player_id') + str(pl.col("season")))
        .otherwise(pl.col('team') + str(pl.col("season")))
        .alias('key')
    ))


def get_combined_data(
        start_season=2016,
        end_season=2025,
        player_positions_to_retrieve=presets.player_fantasy_positions,
        player_cols_to_retrieve=presets.player_fantasy_cols,
        team_cols_to_retrieve=presets.team_fantasy_dst_cols
) -> pl.LazyFrame:
    # Shared variables

    player_data = df_utils.get_player_fantasy_sheet(
        start_season,
        end_season,
        player_positions_to_retrieve,
        player_cols_to_retrieve
    )

    team_data = df_utils.get_team_fantasy_sheet(
        start_season,
        end_season,
        team_cols_to_retrieve
    )

    frames = [player_data, team_data]

    frames = [
        frame.rename({'recent_team': 'team'})
        if 'recent_team' in frame.collect_schema().names()
        else frame
        for frame in frames
    ]

    # combine files
    combined_frame = (pl.concat(frames, how='diagonal_relaxed')
                      .fill_null(strategy='zero'))

    combined_frame = add_id_col(combined_frame)
    combined_frame = add_key_col(combined_frame)

    # print(f"nulls: {combined_frame.null_count().sum_horizontal()}")

    # for column in combined_frame.columns:
    #     print(f"Column: {column} | Dtype: {combined_frame[column].dtype}")

    return combined_frame


def add_ranking_column(
        frame: pl.LazyFrame | pl.DataFrame,
        ranking_col: str,
        new_col_name: str,
        group_cols: list[str] | None = None,
) -> pl.LazyFrame:
    if group_cols is None:
        return frame.lazy().with_columns(
            [
                pl.col(ranking_col)
                .rank('average', descending=True)
                .alias(new_col_name),
            ])

    else:
        return frame.lazy().with_columns(
            [
                pl.col(ranking_col)
                .rank('average', descending=True)
                .over(group_cols)
                .alias(new_col_name),
            ])


def add_position_group_ranking_column(
        frame: pl.LazyFrame | pl.DataFrame,
        group_name: str,
        group: list[str],
        ranking_col: str = 'class_ppr_score',
) -> pl.LazyFrame:
    new_col_name = f'{ranking_col}_{group_name}_ranking'

    return frame.lazy().with_columns([
        pl.when(pl.col("position").is_in(group))
        .then(pl.col("class_ppr_score").rank(method='average', descending=True).over('season'))
        .otherwise(None)
        .alias(new_col_name)
    ])
    # filter and rank
    ranked_group = (frame.lazy()
        .filter(pl.col("position").is_in(group))
        .with_columns([
            pl.col(ranking_col)
            .rank(method="dense", descending=True)
            .alias(new_col_name)
        ])
        .select([key_col, new_col_name])  # Only key + rank
    )

    # join ranking back
    return frame.lazy().join(ranked_group, on=key_col, how='left')


def add_position_group_ranking_columns(
        frame: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    result = frame.lazy()

    for group_name, group in presets.position_groups.items():
        result = add_position_group_ranking_column(
            result, group_name, group
        )

    return result


# TODO: adjust scoring to avg score and account for season games change (2022?)
def add_fantasy_scoring(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    # create scoring column

    # assert all(col in frame.columns for col in presets.combined_fantasy_columns)

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

    return frame.lazy().with_columns(class_scoring_expression)


def add_log_column(frame: pl.LazyFrame | pl.DataFrame, column) -> pl.LazyFrame:
    return frame.lazy().with_columns(
        pl.when(pl.col(column) > 0)
        .then(np.log1p(pl.col(column)))
        .when(pl.col(column) <= 0)
        .then(0)
        .otherwise(None)
        .alias(f'log1p_{column}')
    )


def shift_column_by_season(
        frame: pl.DataFrame | pl.LazyFrame,
        col_name: str,
        shift=-1,
        id_col='id',
        null_replacement=None,
        nan_replacement=None
) -> pl.LazyFrame:
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

    result = (frame.lazy()
    .join(
        frame.lazy().select([
            pl.col(id_col),
            (pl.col("season") + shift),
            pl.col(col_name).alias(f'{prefix}_{col_name}'),
        ]),
        on=[pl.col('id'), pl.col("season")],
        how='left'
    ))

    if nan_replacement is not None:
        result = result.fill_nan(nan_replacement)

    if null_replacement is not None:
        result = result.fill_null(null_replacement)

    return result


def join_from_source(
        frame: pl.LazyFrame | pl.DataFrame,
        file_title: str,
        file_type: futils.FileType,
        columns: Sequence[str] | None = None,
        how: JoinStrategy = 'inner',
        on: str | Expr | Sequence[str | Expr] | None = None,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
) -> pl.LazyFrame:
    if left_on is None and right_on is None and on is None:
        raise ValueError("Must specify either left_on and right_on, or on")

    source = futils.read_source_frame(file_title, file_type).lazy()
    if columns is not None:
        source = source.select(columns)

    return frame.lazy().join(
        source,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
    )


def add_season_aggregate_stats(
        frame: pl.LazyFrame | pl.DataFrame,
        stat_names: Sequence[str] | None = 'num_games',
) -> pl.LazyFrame:
    season_data = df_utils.get_season_aggregate_data(
        presets.min_start_year,
        presets.max_end_year
    ).lazy()

    if stat_names is not None:
        if 'season' not in stat_names:
            stat_names = list(stat_names) + ['season']
        season_data = season_data.select(stat_names)

    return frame.lazy().join(
        season_data,
        on='season',
        how='left'
    )


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
