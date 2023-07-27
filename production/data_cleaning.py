"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)
from scripts import binned_selling_price

#############
# google_search_data_df = load_dataset(context, "raw/google_search_data")
# product_manufacturer_list_df = load_dataset(context, "raw/product_manufacturer_list")
# sales_data_df = load_dataset(context, "raw/sales_data")
# theme_list_df = load_dataset(context, "raw/theme_list")
# theme_product_list_df = load_dataset(context, "raw/theme_product_list")
# social_media_data_df = load_dataset(context, "raw/social_media_data")
###############


# to standardize the date format in every dataset
def standardize_date(date_str):
    # List of possible date formats
    date_formats = ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y-%m-%d"]

    # Iterate through the formats and try to parse the date
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
        except ValueError:
            pass

    # If no valid format is found, return None or the original string, depending on your preference
    return None


@register_processor("data-cleaning", "google_search")
def clean_google_search_table(context, params):
    input_dataset = "raw/google_search_data"
    output_dataset = "cleaned/google_search_data_df_clean"

    # load dataset
    google_search_data_df = load_dataset(context, input_dataset)

    google_search_data_df_clean = (
        google_search_data_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .drop_duplicates()
        .clean_names(case_type="snake")
    )
    google_search_data_df_clean = google_search_data_df_clean.groupby(
        ["date", "claim_id", "platform", "year_new", "week_number"], as_index=False
    )["search_volume"].sum()
    # year_new to year in data comparison
    google_search_data_df_clean["date"] = pd.to_datetime(
        google_search_data_df_clean["date"]
    )
    google_search_data_df_clean["year"] = google_search_data_df_clean["date"].dt.year
    # google_search_data_df_clean.head()

    # save the dataset
    save_dataset(context, google_search_data_df_clean, output_dataset)

    return google_search_data_df_clean


@register_processor("data-cleaning", "sales")
def clean_sales_table(context, params):
    input_dataset = "raw/sales_data"
    output_dataset = "cleaned/sales_data_df_clean"

    # load dataset
    sales_data_df = load_dataset(context, input_dataset)

    sales_data_df_clean = (
        sales_data_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .drop_duplicates()
        .change_type(["sales_dollars_value"], np.int64)
        .clean_names(case_type="snake")
    )
    sales_data_df_clean["system_calendar_key_n"] = pd.to_datetime(
        sales_data_df_clean["system_calendar_key_n"], format="%Y%m%d"
    )

    # save the dataset
    save_dataset(context, sales_data_df_clean, output_dataset)
    return sales_data_df_clean


@register_processor("data-cleaning", "social_media")
def clean_social_media_table(context, params):
    input_dataset = "raw/social_media_data"
    output_dataset = "cleaned/social_media_data_df_clean"

    # load dataset
    social_media_data_df = load_dataset(context, input_dataset)

    social_media_data_df_clean = (
        social_media_data_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .drop_duplicates()
        .clean_names(case_type="snake")
        .dropna(
            subset=["theme_id"], how="any"
        )  # there are some nan values , those are removed
        .change_type(["theme_id"], np.int64)
    )
    social_media_data_df_clean.dropna(subset=["theme_id"], inplace=True)
    social_media_data_df_clean["published_date"] = social_media_data_df_clean[
        "published_date"
    ].apply(standardize_date)
    # for a theme_id and a published id there are 2 rows in many cases
    social_media_data_df_clean = social_media_data_df_clean.groupby(
        ["theme_id", "published_date"], as_index=False
    )["total_post"].sum()
    social_media_data_df_clean["published_date"] = pd.to_datetime(
        social_media_data_df_clean["published_date"]
    )
    social_media_data_df_clean["year"] = social_media_data_df_clean[
        "published_date"
    ].dt.year
    # social_media_data_df_clean.head()

    # save the dataset
    save_dataset(context, social_media_data_df_clean, output_dataset)
    return social_media_data_df_clean


@register_processor("data-cleaning", "product_manufacturer")
def clean_product_manufacturer_table(context, params):
    input_dataset = "raw/product_manufacturer_list"
    output_dataset = "cleaned/prod_df_clean"

    # load dataset
    product_manufacturer_list_df = load_dataset(context, input_dataset)

    # product_manufacturer_list_df=product_manufacturer_list_df[['PRODUCT_ID','Vendor']]
    prod_df_clean = (
        product_manufacturer_list_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .transform_columns(["Vendor"], string_cleaning, elementwise=False)
        .replace({"": np.NaN})
        # ensure that the key column does not have duplicate records
        .remove_duplicate_rows(col_names=["PRODUCT_ID"], keep_first=True)
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type="snake")
    )
    # prod_df_clean.head()
    # save the dataset
    save_dataset(context, prod_df_clean, output_dataset)
    return prod_df_clean


@register_processor("data-cleaning", "theme_list")
def clean_theme_list_table(context, params):
    input_dataset = "raw/theme_list"
    output_dataset = "cleaned/theme_list_df_clean"

    # load dataset
    theme_list_df = load_dataset(context, input_dataset)

    theme_list_df_clean = (
        theme_list_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .drop_duplicates()
        .clean_names(case_type="snake")
    )
    # save the dataset
    save_dataset(context, theme_list_df_clean, output_dataset)
    return theme_list_df_clean


@register_processor("data-cleaning", "theme_product")
def clean_theme_product_table(context, params):
    input_dataset = "raw/theme_product_list"
    output_dataset = "cleaned/theme_product_list_df_clean"

    # load dataset
    theme_product_list_df = load_dataset(context, input_dataset)

    theme_product_list_df_clean = (
        theme_product_list_df.copy()
        # set dtypes : nothing to do here
        .passthrough()
        .drop_duplicates()
        .clean_names(case_type="snake")
    )

    # save the dataset
    save_dataset(context, theme_product_list_df_clean, output_dataset)
    return theme_product_list_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    output_train_features = "train/features"
    output_train_target = "train/target"
    output_test_features = "test/features"
    output_test_target = "test/target"

    # load dataset
    google_search_data_df_clean = load_dataset(
        context, "cleaned/google_search_data_df_clean"
    )
    prod_df_clean = load_dataset(context, "cleaned/prod_df_clean")
    sales_data_df_clean = load_dataset(context, "cleaned/sales_data_df_clean")
    theme_list_df_clean = load_dataset(context, "cleaned/theme_list_df_clean")
    theme_product_list_df_clean = load_dataset(
        context, "cleaned/theme_product_list_df_clean"
    )
    social_media_data_df_clean = load_dataset(
        context, "cleaned/social_media_data_df_clean"
    )

    # To get common theme ids
    media_merged = pd.merge(
        social_media_data_df_clean,
        theme_list_df_clean,
        how="left",
        left_on="theme_id",
        right_on="claim_id",
    )
    g_search_merged = pd.merge(
        google_search_data_df_clean, theme_list_df_clean, how="left", on="claim_id"
    )
    sales_merged = pd.merge(
        sales_data_df_clean, theme_product_list_df_clean, how="left", on="product_id"
    )
    sales_merged = pd.merge(
        sales_merged, theme_list_df_clean, how="left", on="claim_id"
    )
    common_themes = (
        set(media_merged["claim_name"])
        & set(g_search_merged["claim_name"])
        & set(sales_merged["claim_name"])
    )

    media_filtered = media_merged[media_merged["claim_name"].isin(common_themes)]
    g_search_filtered = g_search_merged[
        g_search_merged["date"] >= media_filtered["published_date"].min()
    ]
    g_search_filtered = g_search_filtered[
        g_search_filtered["claim_name"].isin(common_themes)
    ]
    sales_filtered = sales_merged[sales_merged["claim_name"].isin(common_themes)]

    # Social media
    # add 3 weeks to social media date
    media_filtered["published_date"] = media_filtered[
        "published_date"
    ] + datetime.timedelta(days=21)

    # generate year and week features with delayed dates
    media_filtered["week"] = media_filtered["published_date"].apply(
        lambda x: x.isocalendar()[1]
    )
    media_filtered["year"] = media_filtered["published_date"].dt.year

    # groupby to obtain weekly granularity
    media_weekly = media_filtered.groupby(
        ["theme_id", "year", "week"], as_index=False
    ).agg(total_post=("total_post", "sum"))

    # Google search
    # add 1 week to google search dates
    g_search_filtered["date"] = g_search_filtered["date"] + datetime.timedelta(days=7)

    # generate year and week features with delayed dates
    g_search_filtered["week"] = g_search_filtered["date"].apply(
        lambda x: x.isocalendar()[1]
    )
    g_search_filtered["year"] = g_search_filtered["date"].dt.year

    # groupby to obtain weekly granularity
    g_search_weekly = g_search_filtered.groupby(
        ["claim_id", "year", "week"], as_index=False
    ).agg(search_volume=("search_volume", "sum"))

    # sales
    # merge with product dataset to get the vendor column
    sales_filtered = pd.merge(
        sales_filtered, prod_df_clean, on="product_id", how="inner", validate="m:1"
    )

    # filter to obtain only the client data (vendor A)
    sales_filtered_A = sales_filtered[sales_filtered["vendor"] == "A"]
    sales_filtered_A.head()
    # generate year and week features
    sales_filtered_A["week"] = sales_filtered_A["calender_key"].apply(
        lambda x: x.isocalendar()[1]
    )
    sales_filtered_A["year"] = sales_filtered_A["calender_key"].dt.year
    # groupby to obtain weekly granularity
    sales_weekly = sales_filtered_A.groupby(
        ["claim_id", "year", "week"], as_index=False
    ).agg(
        sales_dollars_value=("sales_dollars_value", "sum"),
        sales_units_value=("sales_units_value", "sum"),
        sales_lbs_value=("sales_lbs_value", "sum"),
    )

    # merging social media and google search data
    social_google = pd.merge(
        media_weekly,
        g_search_weekly,
        how="inner",
        left_on=["theme_id", "year", "week"],
        right_on=["claim_id", "year", "week"],
        validate="1:1",
    )
    social_google.drop(["theme_id"], axis=1, inplace=True)

    social_google_sales = pd.merge(
        social_google,
        sales_weekly,
        how="inner",
        on=["claim_id", "year", "week"],
        validate="1:1",
    )

    social_google_sales["weight_per_unit"] = (
        social_google_sales["sales_lbs_value"]
        / social_google_sales["sales_units_value"]
    )
    social_google_sales["weight_per_unit"] = social_google_sales[
        "weight_per_unit"
    ].round(2)
    social_google_sales.drop(
        ["sales_units_value", "sales_lbs_value"], axis=1, inplace=True
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=context.random_seed
    )
    train, test = custom_train_test_split(
        social_google_sales, splitter, by=binned_selling_price
    )

    target_col = "sales_dollars_value"

    train_X, train_y = (
        train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    test_X, test_y = (
        test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
