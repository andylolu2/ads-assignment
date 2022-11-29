from datetime import datetime
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx
import pandas as pd

from . import access
from .config import *
from .utils import BBox, set_areas

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded,  how are outliers encoded? What do columns represent,makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def clean_pp_data(sql: str, database=None, **kwargs):
    df = access.sql_read(sql, database, **kwargs)
    df = df.convert_dtypes()
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"])

    geometry = gpd.points_from_xy(df.longitude, df.latitude)
    gdf = gpd.GeoDataFrame(data=df, geometry=geometry, crs="EPSG:4326")

    return set_areas(gdf)


def query_price(
    date_min: Optional[Union[datetime, str]] = None,
    date_max: Optional[Union[datetime, str]] = None,
    bbox: Optional[BBox] = None,
    town_city: Optional[str] = None,
    district: Optional[str] = None,
    county: Optional[str] = None,
    country: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Queries the properties price database.

    Returns a DataFrame from joining of the property prices data and
    locations data. Optionally filter the results by date range,
    latitude / longitude range, or for particular cities / districts /
    counties / countries.
    """
    where_clause = "TRUE"
    args = {}

    if bbox is not None:
        latitude_max, latitude_min, longitude_max, longitude_min = bbox
    else:
        latitude_max = latitude_min = longitude_max = longitude_min = None

    all_args = [
        ("pp.date_of_transfer", date_min, ">="),
        ("pp.date_of_transfer", date_max, "<="),
        ("pc.latitude", latitude_min, ">="),
        ("pc.latitude", latitude_max, "<="),
        ("pc.longitude", longitude_min, ">="),
        ("pc.longitude", longitude_max, "<="),
        ("pp.town_city", town_city, "="),
        ("pp.district", district, "="),
        ("county", county, "="),
        ("country", country, "="),
    ]

    for i, (field, value, comparator) in enumerate(all_args):
        if value is not None:
            where_clause += f" AND {field} {comparator} %(arg_{field}_{i})s"
            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d")
            args[f"arg_{field}_{i}"] = value

    # Note: Discovered that using pp.postcode index instead of the pp.date
    # index results in significantly faster queries, especially if bbox is small.
    index = "`pp.postcode`" if bbox is not None else "`pp.date`"
    query = f"""
        SELECT 
            pp.price, 
            pp.date_of_transfer, 
            pp.postcode, 
            pp.property_type, 
            pp.new_build_flag, 
            pp.tenure_type, 
            pp.locality, 
            pp.town_city, 
            pp.district, 
            pp.county, 
            pc.country, 
            pc.latitude, 
            pc.longitude, 
            pp.db_id AS pp_db_id, 
            pc.db_id AS pc_db_id
        FROM pp_data AS pp USE INDEX ({index}) INNER JOIN postcode_data as pc 
            ON pp.postcode = pc.postcode
        WHERE {where_clause}
    """

    if limit is not None:
        query += "LIMIT %(limit)s"
        args["limit"] = limit

    gdf = clean_pp_data(query, "property_prices", **args)
    return gdf


def clean_osm_data(*bboxs: BBox) -> gpd.GeoDataFrame:
    # built-in conversion seems to work well, and using <NA> to handle missing values seems reasonable
    gdf = pd.concat([access.osm_pois(bbox) for bbox in bboxs])
    return gdf.convert_dtypes()


def visualise_area(bbox: BBox, ax: plt.Axes):
    graph = osmnx.graph_from_bbox(*bbox)
    nodes, edges = osmnx.graph_to_gdfs(graph)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")


def visualise_osm(bbox: BBox, feature: str, legend: bool, ax: plt.Axes):
    visualise_area(bbox, ax)

    gdf = clean_osm_data(bbox)
    is_num = pd.api.types.is_numeric_dtype(gdf[feature].dtype)
    if is_num:
        gdf[feature] = gdf[feature].astype(float)

    gdf.plot(
        ax=ax,
        column=feature,
        categorical=not is_num,
        cmap="rainbow",
        alpha=0.9,
        markersize=10,
        edgecolor="b",
        linewidth=0.5,
        legend=legend,
        missing_kwds={
            "color": "silver",
            "alpha": 0.4,
            "markersize": 10,
        },
    )

    ax.set_xlim(bbox[3], bbox[2])
    ax.set_ylim(bbox[1], bbox[0])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(feature)


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def verify_pp_data():
    # Note: expensive queries
    gdf = clean_pp_data("SELECT DISTINCT(tenure_type) FROM pp_data", "property_prices")
    assert set(gdf.tenure_type) == {"F", "L", "U"}

    # property type is one of S, D, T, F, O
    gdf = clean_pp_data(
        "SELECT DISTINCT(property_type) FROM pp_data", "property_prices"
    )
    assert set(gdf.property_type) == {"S", "D", "T", "F", "O"}
