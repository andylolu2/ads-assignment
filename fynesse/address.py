from datetime import datetime, timedelta
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing_extensions import Literal

from .assess import clean_osm_data, query_price
from .utils import bbox, spatial_join, to_flat_crs, to_geographic_crs

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

OSM_FEATURES = {
    "building": "categorical",
    "amenity": "categorical",
    "capacity": "number",
    "building:levels": "number",
    "shop": "bool",
    "natural": "bool",
    "osm_geo_area": "number",
}

PP_FEATURES = {
    "price": "number",
    "date_of_transfer": "date",
    "property_type": "categorical",
}


def process_features(
    gdf: gpd.GeoDataFrame,
    eval_mapping: Optional[Dict] = None,
):
    cat_features = []
    all_features = {**OSM_FEATURES, **PP_FEATURES}

    eval = eval_mapping is not None
    if eval_mapping is None:
        eval_mapping = dict()

    for feature, dtype in all_features.items():
        if dtype == "bool":
            gdf[feature] = gdf[feature].notna()
        elif dtype == "number":
            values = pd.to_numeric(gdf[feature], errors="coerce")

            if feature in eval_mapping:
                fill_value = eval_mapping[feature]
            else:
                assert not eval
                fill_value = values.median()
                eval_mapping[feature] = fill_value

            values = values.fillna(fill_value)
            gdf[feature] = values
        elif dtype == "categorical":
            cat_features.append(feature)
            if feature in eval_mapping:
                categories = eval_mapping[feature]
            else:
                assert not eval
                counts = gdf[feature].value_counts()
                categories = set((counts[counts >= 20]).index)
                if set(counts.index) != categories:
                    categories.add("<OTHER>")
                eval_mapping[feature] = categories

            def map(value):
                if value not in categories:
                    if "<OTHER>" in categories:
                        return "<OTHER>"
                    else:
                        return pd.NA
                return value

            cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
            gdf[feature] = gdf[feature].map(map, na_action="ignore").astype(cat_type)
        elif dtype == "date":
            t = pd.to_datetime(gdf[feature])
            gdf[feature] = (t - datetime(2005, 1, 1)).dt.days
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

    gdf = pd.get_dummies(gdf, columns=cat_features, prefix=cat_features)
    return gdf, eval_mapping


def full_set(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: str,
    bbox_length: float,
    t_days: float,
    buffer_size: int,
):
    t_delta = timedelta(days=t_days)
    bbox_ = bbox(latitude, longitude, bbox_length, bbox_length)
    date_min, date_max = date - t_delta, date + t_delta

    print("Retrieving property price data...")
    pp_gdf = query_price(bbox=bbox_, date_min=date_min, date_max=date_max)
    pp_gdf = pp_gdf.loc[
        :, pp_gdf.columns.isin({"pp_db_id", "geometry"} | PP_FEATURES.keys())
    ]

    pp_gdf = to_flat_crs(pp_gdf)
    pp_gdf.geometry = pp_gdf.geometry.buffer(buffer_size)
    pp_gdf = to_geographic_crs(pp_gdf)
    pp_gdf["pp_geo"] = pp_gdf.geometry

    geometry = gpd.points_from_xy(x=[longitude], y=[latitude])
    data = {
        "pp_db_id": [-1],
        "price": [None],
        "date_of_transfer": [date],
        "property_type": [property_type],
    }
    pp_gdf_pred = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    mask = np.random.rand(len(pp_gdf)) < 0.8
    pp_gdf_train = pp_gdf.iloc[mask]
    pp_gdf_test = pp_gdf.iloc[~mask]

    print("Downloading OSM data...")
    osm_gdf = clean_osm_data(bbox_)
    osm_gdf = osm_gdf.query("element_type in ('node', 'way')")
    osm_gdf = osm_gdf.loc[:, osm_gdf.columns.isin({"geometry"} | OSM_FEATURES.keys())]
    osm_gdf = to_flat_crs(osm_gdf)
    osm_gdf["osm_geo_area"] = osm_gdf.geometry.area
    osm_gdf = to_geographic_crs(osm_gdf)

    print("Performing spatial join...")
    gdf_train = spatial_join(
        pp_gdf_train, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    )
    gdf_test = spatial_join(
        pp_gdf_test, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    )
    gdf_pred = spatial_join(
        pp_gdf_pred, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    )
    return gdf_train, gdf_test, gdf_pred


def train_test_set(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: Literal["F", "S", "D", "T", "O"],
    bbox_length: float,
    t_days: float,
    buffer_size: int,
):
    gdf_train, gdf_test, gdf_pred = full_set(
        latitude, longitude, date, property_type, bbox_length, t_days, buffer_size
    )
    gdf_train, eval_mapping = process_features(gdf_train)
    gdf_test, _ = process_features(gdf_test, eval_mapping)
    gdf_pred, _ = process_features(gdf_pred, eval_mapping)

    agg_train = gdf_train.dissolve(by="pp_db_id", aggfunc=agg_features)
    agg_train = agg_train.drop(columns=["geometry"])

    agg_test = gdf_test.dissolve(by="pp_db_id", aggfunc=agg_features)
    agg_test = agg_test.drop(columns=["geometry"])

    agg_pred = gdf_pred.dissolve(by="pp_db_id", aggfunc=agg_features)
    agg_pred = agg_pred.drop(columns=["geometry"])

    assert list(agg_train.columns) == list(agg_test.columns) == list(agg_pred.columns)

    features = agg_train.columns != "price"

    train_x = agg_train.loc[:, features].to_numpy(np.float32)
    train_y = agg_train["price"].to_numpy(np.float32)
    test_x = agg_test.loc[:, features].to_numpy(np.float32)
    test_y = agg_test["price"].to_numpy(np.float32)
    pred_x = agg_pred.loc[:, features].to_numpy(np.float32)

    return train_x, train_y, test_x, test_y, pred_x


def agg_features(series: pd.Series):
    feature = str(series.name)
    if feature in OSM_FEATURES:
        agg_method = OSM_FEATURES[feature]
        if agg_method == "bool":
            return series.sum()
        elif agg_method == "number":
            return series.mean()
    elif any(feature.startswith(key) for key in OSM_FEATURES):
        return series.sum()
    elif feature in PP_FEATURES or any(feature.startswith(key) for key in PP_FEATURES):
        return series.iloc[0]


def predict_price(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: Literal["F", "S", "D", "T", "O"],
    t_days=365 * 2,
    bbox_length=2000,
    buffer_size=200,
):
    """Price prediction for UK housing."""

    train_x, train_y, test_x, test_y, pred_x = train_test_set(
        latitude, longitude, date, property_type, t_days, bbox_length, buffer_size
    )

    model = sm.GLM(train_y, train_x, family=sm.families.Poisson())
    results = model.fit()

    train_y_pred = results.get_prediction(train_x).summary_frame(alpha=0.05)
    test_y_pred = results.get_prediction(test_x).summary_frame(alpha=0.05)

    pass
