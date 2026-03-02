import pathlib

import polars as pl


def get_output_path(file_name : str) -> pathlib.Path:
    return pathlib.Path.cwd() / 'datafiles' / file_name

def output_frame(
        frame: pl.DataFrame,
        path: pathlib.Path,
        output_parquet=True,
        output_excel=True,
        output_csv=False,
):
    # output as parquet and csv
    print(frame.write_csv(str(path) + ".csv", include_header=True))
    print(frame.write_excel(str(path) + ".xlsx", include_header=True))
    print(frame.write_parquet(str(path) + ".parquet"))