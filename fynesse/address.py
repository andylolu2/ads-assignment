import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing_extensions import Literal

from fynesse import assess, utils
from fynesse.config import config
from fynesse.utils import BBox


def expanded_pp_data(bbox: BBox, date_min: datetime, date_max: datetime, radius: float):
    pp_gdf = assess.query_price(bbox=bbox, date_min=date_min, date_max=date_max)
    pp_gdf = pp_gdf.loc[
        :, pp_gdf.columns.isin({"pp_db_id", "geometry"} | config["pp_features"].keys())
    ]
    pp_gdf = utils.to_flat_crs(pp_gdf)
    pp_gdf.geometry = pp_gdf.geometry.buffer(radius)
    pp_gdf = utils.to_geographic_crs(pp_gdf)
    return pp_gdf


def sample_pp_data(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: str,
    radius: float,
):
    geometry = gpd.points_from_xy(x=[longitude], y=[latitude])
    data = {
        "pp_db_id": [-1],
        "price": [None],
        "date_of_transfer": [date],
        "property_type": [property_type],
    }
    sample = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
    sample = utils.to_flat_crs(sample)
    sample.geometry = sample.geometry.buffer(radius)
    sample = utils.to_geographic_crs(sample)
    return sample


def filtered_osm_data(bbox: BBox, features: set):
    osm_gdf = assess.clean_osm_data(bbox)
    # exclude elements of type "relation"
    # osm_gdf = osm_gdf.query("element_type in ('node', 'way')")
    osm_gdf = osm_gdf.loc[:, osm_gdf.columns.isin({"geometry"} | features)]
    osm_gdf = utils.to_flat_crs(osm_gdf)
    osm_gdf["osm_geo_area"] = osm_gdf.geometry.area
    osm_gdf = utils.to_geographic_crs(osm_gdf)
    osm_gdf = osm_gdf.reset_index(drop=True)
    return osm_gdf


def encode_categorial(features: pd.Series, cutoff: float, categories=None):
    if categories is None:  # training time
        counts = features.value_counts(normalize=True)
        cutoff_label = (counts.cumsum() > cutoff).idxmax()
        cutoff_index = counts.index.get_loc(cutoff_label)
        categories = set((counts.iloc[: cutoff_index + 1]).index)

        # add <OTHER> category if categories does not cover all categories
        if set(counts.index) - categories:
            categories.add("<OTHER>")

    def map(value):
        if value not in categories:
            if "<OTHER>" in categories:
                return "<OTHER>"
            else:
                # Rare case. Example where this could happen:
                # A new property type appears in eval / inference time
                return pd.NA
        return value

    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    return features.map(map, na_action="ignore").astype(cat_type), categories


def encode_numerical(features: pd.Series, fill_value=None):
    values = pd.to_numeric(features, errors="coerce")

    if fill_value is None:
        fill_value = values.median()

    return values.fillna(fill_value), fill_value


def encode_features(
    gdf_: gpd.GeoDataFrame,
    categorical_cutoff: float,
    eval_mapping: Optional[Dict] = None,
):
    gdf = gdf_.copy()
    cat_features = []
    all_features = {**config["osm_features"], **config["pp_features"]}
    if eval_mapping is None:
        eval_mapping = dict()

    for feature, type_ in all_features.items():
        if feature not in gdf:
            continue
        elif type_ == "number":
            if feature in eval_mapping:
                fill_value = eval_mapping[feature]
                gdf[feature], _ = encode_numerical(gdf[feature], fill_value)
            else:
                gdf[feature], fill_value = encode_numerical(gdf[feature])
                eval_mapping[feature] = fill_value
        elif type_ == "categorical":
            cat_features.append(feature)
            if feature in eval_mapping:
                categories = eval_mapping[feature]
                gdf[feature], _ = encode_categorial(
                    gdf[feature], categorical_cutoff, categories
                )
            else:
                gdf[feature], categories = encode_categorial(
                    gdf[feature], categorical_cutoff
                )
                eval_mapping[feature] = categories
        elif type_ == "date":
            t = pd.to_datetime(gdf[feature])
            gdf[feature] = np.log((t - datetime(2005, 1, 1)).dt.days)
        else:
            raise ValueError(f"Invalid dtype: {type_}")

    gdf = pd.get_dummies(gdf, columns=cat_features, prefix=cat_features)
    return gdf, eval_mapping


def agg_features(df: pd.DataFrame):
    distance = df["distance"]
    osm_features = config["osm_features"]
    pp_features = config["pp_features"]
    result = {}
    for feature in df:
        if feature in osm_features or any(
            feature.startswith(key) for key in osm_features
        ):
            not_na = df[feature].notna()
            inv_dist = 1 / distance[not_na]
            weights = inv_dist / inv_dist.sum()
            result[feature] = (df[feature][not_na] * weights).sum()
        elif feature in pp_features or any(
            feature.startswith(key) for key in pp_features
        ):
            result[feature] = df[feature].iloc[0]
    return pd.Series(result)


def full_set(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: str,
    bbox_length: float,
    t_days: float,
    radius: float,
):
    t_delta = timedelta(days=t_days)
    bbox_ = utils.bbox(latitude, longitude, bbox_length, bbox_length)
    date_min, date_max = date - t_delta, date + t_delta

    print("Retrieving property price data...")
    pp_gdf = expanded_pp_data(bbox_, date_min, date_max, radius)
    # Construct sample pp point for inference later
    pp_gdf_pred = sample_pp_data(latitude, longitude, date, property_type, radius)
    # split pp data in train and test
    pp_gdf_train, pp_gdf_test = utils.split_df(pp_gdf, prop=0.8)

    print("Downloading OSM data...")
    osm_gdf = filtered_osm_data(bbox_, set(config["osm_features"].keys()))

    print("Performing spatial join...")
    gdf_train = utils.spatial_join(
        pp_gdf_train, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    ).drop(columns="index_osm")
    gdf_test = utils.spatial_join(
        pp_gdf_test, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    ).drop(columns="index_osm")
    gdf_pred = utils.spatial_join(
        pp_gdf_pred, osm_gdf, how="left", lsuffix="pp", rsuffix="osm"
    ).drop(columns="index_osm")
    return gdf_train, gdf_test, gdf_pred


def train_test_set(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: Literal["F", "S", "D", "T", "O"],
    bbox_length: float,
    t_days: float,
    radius: int,
    categorical_cutoff: float,
):
    gdf_train, gdf_test, gdf_pred = full_set(
        latitude, longitude, date, property_type, bbox_length, t_days, radius
    )
    gdf_train, eval_mapping = encode_features(gdf_train, categorical_cutoff)
    gdf_test, _ = encode_features(gdf_test, categorical_cutoff, eval_mapping)
    gdf_pred, _ = encode_features(gdf_pred, categorical_cutoff, eval_mapping)

    agg_train = gdf_train.groupby("pp_db_id").apply(agg_features)
    agg_test = gdf_test.groupby("pp_db_id").apply(agg_features)
    agg_pred = gdf_pred.groupby("pp_db_id").apply(agg_features)

    assert list(agg_train.columns) == list(agg_test.columns) == list(agg_pred.columns)

    features = agg_train.columns != "price"

    train_x = pd.DataFrame(agg_train.loc[:, features])
    train_y = pd.DataFrame(agg_train["price"])
    test_x = pd.DataFrame(agg_test.loc[:, features])
    test_y = pd.DataFrame(agg_test["price"])
    pred_x = pd.DataFrame(agg_pred.loc[:, features])

    return train_x, train_y, test_x, test_y, pred_x


def evaluate_mse(results, x, y):
    y_pred_mean, _, _ = utils.predict(results, x)
    y_np = np.log10(y.to_numpy(np.float32)[:, 0])
    mse = np.mean((y_np - y_pred_mean) ** 2)
    return mse


def predict_price(
    latitude: float,
    longitude: float,
    date: datetime,
    property_type: Literal["F", "S", "D", "T", "O"],
    t_days=365 * 2,
    bbox_length=2000,
    radius=200,
    categorical_cutoff=0.95,
):
    train_x, train_y, test_x, test_y, pred_x = train_test_set(
        latitude,
        longitude,
        date,
        property_type,
        t_days,
        bbox_length,
        radius,
        categorical_cutoff,
    )

    train_size, feature_size = train_x.shape
    print(f"Training with {train_size} data points with {feature_size} features.")
    if train_size < 100:
        warnings.warn(f"Training set only has size {train_size}.")
    if feature_size < 10:
        warnings.warn(f"Training set only has {feature_size} features.")

    x = train_x.to_numpy(np.float32)
    y = np.log10(train_y.to_numpy(np.float32))
    results = sm.OLS(y, x).fit()

    train_mse = evaluate_mse(results, train_x, train_y)
    test_mse = evaluate_mse(results, test_x, test_y)
    if test_mse > 4 * train_mse:
        warnings.warn(
            "Performance on test set is significantly worse than that in training set."
            + f"Predictions might be unreliable. Train MSE: {train_mse}, test MSE: {test_mse}"
        )

    pred_y = utils.predict(results, pred_x)

    return results, train_x, train_y, test_x, test_y, pred_x, pred_y
