# Databricks notebook source
# MAGIC %md ## Low code ML
# MAGIC
# COMMAND ----------

from typing import List

import pandas as pd
from databricks import automl
from pyspark.sql import DataFrame

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", False)


def custom_data_preparation(
    table_name: str, categorical_features: List[str], numeric_features: List[str]
) -> DataFrame:
    """
    Not strictly necassary, just makes it easier to view and understand
    how categorical encoding works
    """
    df = spark.read.table(table_name).drop("mayor", "_c0").toPandas().dropna()
    df["party"] = df["party"].apply(lambda x: x.replace(" ", "").replace("\n", ""))

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
    "naturkampen",
    categorical_features=["county", "party"],
    numeric_features=["area", "population"],
)

display(df.sample(0.1))
# COMMAND ----------

# MAGIC %md Selve modelltreningen foregår i neste celle. Vi bruker [databricks automl](https://docs.databricks.com/applications/machine-learning/automl.html?_ga=2.221852319.625249080.1662571032-1699436349.1656921042) som trener på forskjellige modeller fra scikit-learn, XGBoost og LightGBM. Vi har satt `timeout_minutes=5` som betyr at vi gir auto-ML høyst 5 minutter på å trene og tune ulike modeller.
# MAGIC
# MAGIC Mens du venter på kjøringen kan du sannsynligvis starte på oppgavene under.

# COMMAND ----------

summary = automl.regress(
    dataset=df,
    target_col="rank",
    exclude_cols=["name"],
    primary_metric="mae",
    timeout_minutes=5,
)
