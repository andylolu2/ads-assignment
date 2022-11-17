from pathlib import Path
from typing import Union

import pandas as pd
import pymysql
import pymysql.cursors
import requests

from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def create_connection(database=None) -> pymysql.Connect:
    """Create a database connection to database.

    Depends on values of `db_username`, `db_password`, `db_host`, and
    `db_port` in config.

    Args:
        database (str, optional): database name.

    Returns:
        pymysql.Connect: database connection
    """
    return pymysql.connect(
        user=config["db_username"],
        passwd=config["db_password"],
        host=config["db_host"],
        port=config["db_port"],
        local_infile=1,
        db=database,
        read_timeout=60,
        write_timeout=60,
    )


def download(url: str, local: Union[str, Path], skip_if_exists=True):
    path = Path(local)
    if path.exists() and skip_if_exists:
        return path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path


def upload(file_path: Path, *, table: str, database):
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                    LOAD DATA LOCAL INFILE '{str(file_path)}' INTO TABLE `{table}`
                    FIELDS TERMINATED BY ',' ENCLOSED BY '"'
                    LINES STARTING BY '' TERMINATED BY '\n'
            """
            )
        conn.commit()
    print(f"Uploaded data {file_path}")


def sql_read(sql: str, database=None):
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    return df


def reset(table: str, database=None):
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE `{table}`")
        conn.commit()


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError
