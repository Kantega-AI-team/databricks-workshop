# Databricks notebook source
from typing import List

import pandas as pd
from pyspark.sql import DataFrame

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", False)


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


# COMMAND ----------

# MAGIC %md #### Oppgave 1: Kan du gjøre en prediksjon med en av modellene?
# MAGIC
# MAGIC Hvordan tolker du prediksjonen?
# MAGIC
# MAGIC **TIPS**: Se på eksperimentnotebooken som har laget modellen

# COMMAND ----------

import mlflow

logged_model = "runs:/6ff4489f99f846d0ad0ca38f6700ba66/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

sample = df.sample(0.1).limit(5).toPandas()
sample["predictions"] = loaded_model.predict(sample)

# COMMAND ----------

sample

# COMMAND ----------

# MAGIC %md #### Oppgave 2: Kan du forklare modellen du brukte for prediksjon?
# MAGIC
# MAGIC Hvilke inputvariabler kan forklare mest av naturkampen-plasseringen?
# MAGIC
# MAGIC **TIPS**: Se på eksperimentnotebooken

# COMMAND ----------

# MAGIC %md Se command 19 i [eksperimentnotebooken](https://adb-2582450973867059.19.azuredatabricks.net/?o=2582450973867059#notebook/3901909671451995/command/3901909671452043 )
