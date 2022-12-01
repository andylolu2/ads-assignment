import math
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

BBox = Tuple[float, float, float, float]


def bbox(latitude, longitude, width, height) -> BBox:
    """Creates a bounding box centred at a given point

    BBox is represented as a 4-tuple of north, south, east, west.
    """

    # convert metres to degrees
    height_degree = abs(height / 111111)
    width_degree = abs(width / (111111 * math.cos(math.radians(latitude))))
    return (
        latitude + height_degree / 2,
        latitude - height_degree / 2,
        longitude + width_degree / 2,
        longitude - width_degree / 2,
    )


def to_flat_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(crs=3857)


def to_geographic_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(crs=4326)


def set_areas(gdf: gpd.GeoDataFrame):
    gdf = to_flat_crs(gdf)
    gdf["geometry_area"] = gdf.area
    gdf = to_geographic_crs(gdf)
    return gdf


def spatial_join(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, *args, **kwargs
) -> gpd.GeoDataFrame:
    gdf1 = to_flat_crs(gdf1)
    gdf2 = to_flat_crs(gdf2)
    gdf2["saved_geometry"] = gdf2.geometry
    res = gdf1.sjoin(gdf2, *args, **kwargs)
    res["distance"] = res.centroid.distance(res["saved_geometry"].centroid)
    res = to_geographic_crs(res)
    return res


def split_df(df: pd.DataFrame, prop: float):
    mask = np.random.rand(len(df)) < prop
    df1 = df.iloc[mask]
    df2 = df.iloc[~mask]
    return df1, df2


def get_type(value):
    if isinstance(value, str):
        if value.isdigit():
            return int
        elif value.replace(".", "").isdigit():
            return float
    return type(value)


def predict(results, x):
    pred = results.get_prediction(x.to_numpy(np.float32)).summary_frame(0.95)
    return pred["mean"], pred["obs_ci_lower"], pred["obs_ci_upper"]
