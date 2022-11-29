import math
from typing import Tuple

import geopandas as gpd

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
    res = gdf1.sjoin(gdf2, *args, **kwargs)
    res = to_geographic_crs(res)
    return res
