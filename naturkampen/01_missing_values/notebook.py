# Databricks notebook source
# MAGIC %md ## Manglende data - Hva gjør vi da?
# MAGIC
# MAGIC Du er Data Scientist for bedriften *Grønnere enn Grønnest*, og har fått beskjed om å lage en maskinlæringsmodell for å predikere hyttebygging i norske kommuner.
# MAGIC
# MAGIC Modellen du skal bygge skal være basert på fylke, kommunestørrelse, populasjon og økologisk tillstand i elver og vann. I tillegg finnes en variabel du ikke helt har forstått hva er for noe, og som attpåtil mangler informasjon for noen kommuner.
# MAGIC
# MAGIC Et utdrag fra datasettet kan du se ved å kjøre kommandoen under:

# COMMAND ----------

df = spark.read.table("naturkampen_1")
display(df.sample(0.2))

# COMMAND ----------

# MAGIC %md #### Oppgave 1: Hvordan vil du gå frem for å håndtere disse manglende dataene?
# MAGIC Har du flere forslag er det fritt fram å kommme med alle

# COMMAND ----------

# MAGIC %md
# MAGIC **Svar**:
# MAGIC
# MAGIC *Fyll inn ditt svar her*

# COMMAND ----------

# MAGIC %md #### Oppgave 2: Hva tror du konsekvensene er ved taktikken du har valgt?

# COMMAND ----------

# MAGIC %md
# MAGIC **Svar**:
# MAGIC
# MAGIC *Fyll inn ditt svar her*

# COMMAND ----------

# MAGIC %md Nå er det på tide å gå i gang med selve kodingen. Vi har allerede gjort oss klar til å trene en enkel maskinlæringsmodell, men først må vi altså rydde opp i dataene. Dette er (delvis) din jobb!

# COMMAND ----------

""" Vi begynner med å importere et par nyttige klasser og funksjoner fra numpy, 
pandas og scikit-learn, for så og konvertere spark-dataframen til en Pandas dataframe. 
Det er ikke så farlig om du ikke forstår hva som skjer her! """

import matplotlib.pyplot as plt
import pandas as pd
from numpy import absolute, mean
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer  # Her ligger et lite hint...
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Se på dataene
pdf = df.toPandas()
pdf.head()

# COMMAND ----------

# MAGIC %md #### Oppgave 3 - alternativ A:
# MAGIC Håndtere de manglende verdiene ved å filtrere de bort. Om du heller vil gjøre noe annet, gå til alternativ B

# COMMAND ----------

pdf.drop(columns=["_c0", "name"], inplace=True)
# TODO: pdf.dropna( ... ) fyll ut nødvendige kommandoer her

# COMMAND ----------

# MAGIC %md #### Oppgave 3 - alternativ B:
# MAGIC Håndtere de manglende verdiene ved å sette en verdi på en eller annen måte

# COMMAND ----------

target = "cabin_construction"
numeric_features = ["area", "population", "eco_river_water", "unnamed_metric"]
categorical_features = ["county"]


numeric_transformer = Pipeline(
    # TODO: Hvis du prøver alternativ B fyller du inn et steg her. Ellers ender du opp med en ubrukelig modell..!
    steps=[("scaler", StandardScaler())]
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


# COMMAND ----------

# MAGIC %md #### Trene modellen, vi bruker kryssvalidering for å evaluere performance

# COMMAND ----------

X, y = pdf[[col for col in pdf.columns if col not in [target, "name"]]], pdf[target]
model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)
scores = absolute(scores)


# Gjennomsnittlig absoluttfeil:
s_mean = mean(scores)
print("Mean MAE: %.3f" % (s_mean))

# COMMAND ----------

# MAGIC %md #### Bonusoppgaver
# MAGIC
# MAGIC - Kan du forklare hva som skjer på linje 3 og 4 i forrige celle?
# MAGIC - Kan du si noe om hvor god denne modellen egentlig er?
# MAGIC - Gå til naturkampen.no og se om du finner ut hvilken indikator "unnamed_metric" egentlig er. Er du fortsatt fornøyd med taktikken du valgte?
