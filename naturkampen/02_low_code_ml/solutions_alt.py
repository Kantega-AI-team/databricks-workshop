# Databricks notebook source
from typing import List

import pandas as pd
import statsmodels.formula.api as sm
from pyspark.sql import DataFrame


def custom_data_preparation(
    table_name: str, categorical_features: List[str], numeric_features: List[str]
) -> DataFrame:
    """
    Not strictly necassary, just makes it easier to view and understand
    how categorical encoding works
    """
    df = spark.read.table(table_name).toPandas()

    new_numerical_columns = []
    for category in categorical_features:
        original_array = df[category].unique()
        dummies = pd.get_dummies(df[category], drop_first=True)
        new_numerical_columns = new_numerical_columns + list(dummies.columns)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[category], inplace=True)
        base = [item for item in original_array if item not in df.columns]

    sdf = spark.createDataFrame(df)
    for category in new_numerical_columns + numeric_features:
        metadata_dict = sdf.schema[category].metadata
        metadata_dict["spark.contentAnnotation.semanticType"] = "numeric"
        sdf = sdf.withMetadata(category, metadata_dict)
    return sdf


df = custom_data_preparation(
    "naturkampen_2",
    categorical_features=["county", "mayors_party"],
    numeric_features=["area", "population"],
)

for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(" ", "_"))
pdf = df.toPandas()


mod = sm.ols(
    formula="rank~area+population+INNLANDET+MØRE_OG_ROMSDAL+NORDLAND+ROGALAND+TROMS_OG_FINNMARK+TRØNDELAG+VESTFOLD_OG_TELEMARK+VESTLAND+VIKEN+Ap+Frp+H+KrF+MDG+SV+Sp+V",
    data=pdf,
)
res = mod.fit()
print(res.summary())
