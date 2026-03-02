import pathlib
from enum import Enum

import polars as pl

class FileType(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"


def get_output_path(file_title: str) -> pathlib.Path:
    return pathlib.Path.cwd() / 'output' / file_title


def get_source_path(file_title: str) -> pathlib.Path:
    return pathlib.Path.cwd() / 'source' / file_title

def output_frame_to_path(
        frame: pl.DataFrame,
        path: pathlib.Path,
        output_parquet=True,
        output_excel=True,
        output_csv=False,
) -> None:
    # output as parquet and csv
    if output_csv: print(frame.write_csv(str(path) + "." + FileType.CSV.value, include_header=True))
    if output_excel: print(frame.write_excel(str(path), include_header=True))
    if output_parquet: print(frame.write_parquet(str(path) + "." + FileType.PARQUET.value))

def output_result_frame(
        frame: pl.DataFrame,
        file_title: str,
        output_parquet=True,
        output_excel=True,
        output_csv=False,
):
    output_frame_to_path(frame, get_output_path(file_title), output_parquet, output_excel, output_csv)

def output_source_frame(
        frame: pl.DataFrame,
        file_title: str,
        output_parquet=True,
        output_excel=True,
        output_csv=False,
) -> None:
    output_frame_to_path(frame, get_source_path(file_title), output_parquet, output_excel, output_csv)


# reading

def read_frame_from_path(
        file_path: pathlib.Path,
        file_type: FileType,
) -> pl.DataFrame:
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be FileType enum")

    match file_type:
        case FileType.CSV:
            return pl.read_csv(file_path)
        case FileType.PARQUET:
            return pl.read_parquet(file_path)
        case FileType.EXCEL:
            return pl.read_excel(file_path)
        case _:
            raise ValueError("Unrecognized file type")

def read_source_frame(
        file_title: str,
        file_type: FileType,
) -> pl.DataFrame:
    return read_frame_from_path(get_source_path(file_title), file_type)

def read_output_frame(
        file_title: str,
        file_type: FileType,
) -> pl.DataFrame:
    return read_frame_from_path(get_source_path(file_title), file_type)