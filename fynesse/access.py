from pathlib import Path
from typing import Union

import osmnx
import pandas as pd
import pymysql
import pymysql.cursors
import requests

from .config import *
from .utils import BBox, set_areas

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
    )


def download(url: str, local: Union[str, Path], skip_if_exists=True):
    """Downloads a file from a url to a local path"""
    path = Path(local)
    if path.exists() and skip_if_exists:
        return path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path


def upload(file_path: Path, *, table: str, database, delimiter=""):
    """Uploads a local file to a given table in database"""
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    LOAD DATA LOCAL INFILE %(file_path)s INTO TABLE %(table)s
                    FIELDS TERMINATED BY ',' ENCLOSED BY %(delimiter)s
                    LINES STARTING BY '' TERMINATED BY '\n'
                """,
                {
                    "file_path": str(file_path),
                    "table": table,
                    "delimiter": delimiter,
                },
            )
        conn.commit()
    print(f"Uploaded data {file_path}")


def sql_read(sql: str, database=None, **kwargs):
    """Performs a sql query and return the results as a pandas DataFrame"""
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, kwargs)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    return df


def reset(table: str, database=None):
    """Clears a given table"""
    with create_connection(database) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE %s", (table,))
        conn.commit()


def osm_pois(bbox: BBox):
    """Access the OpenStreetMap API and download relevant point of interests"""
    gdf = osmnx.geometries_from_bbox(*bbox, tags=config["osm_tags"])
    return set_areas(gdf)
