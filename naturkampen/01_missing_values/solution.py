# Databricks notebook source
import matplotlib.pyplot as plt
import pandas as pd
from numpy import absolute, mean
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = spark.read.table("naturkampen_1")
pdf = df.toPandas()
pdf.drop(columns=["_c0", "name"], inplace=True)

target = "cabin_construction"
numeric_features = ["area", "population", "eco_river_water", "unnamed_metric"]
categorical_features = ["county"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values using mean
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regression", LinearRegression())]
)


X, y = pdf[[col for col in pdf.columns if col not in [target, "name"]]], pdf[target]
model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
scores = absolute(scores)

s_mean = mean(scores)
print("Mean MAE: %.3f" % (s_mean))

# COMMAND ----------
