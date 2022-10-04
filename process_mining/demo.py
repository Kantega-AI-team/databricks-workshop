# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# data
import pandas as pd

# filters
from pm4py import (
    filter_directly_follows_relation,
    filter_start_activities,
    filter_variants_top_k,
)

# process mining
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils

# visualization
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# COMMAND ----------

# MAGIC %md
# MAGIC # Laster inn data

# COMMAND ----------

df = spark.read.table("process_mining").toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Konverterer pandas df til en event log

# COMMAND ----------

# MAGIC %md
# MAGIC For å lage en event log som er lesbar for pm4py-pakken må man gjøre noen transformasjoner og sorteringer. Timestamp må konverteres og sorteres i riktig rekkefølge og nøkkelvariabler må få få nye navn. I dette eksempelet har vi valgt å erstatte originale variabelnavn med de som pakken krever, men man kan også lagre disse som nye variabler hvis man ikke ønsker å erstatte navnene direkte

# COMMAND ----------

log = dataframe_utils.convert_timestamp_columns_in_df(df)
log = log.rename(
    columns={
        "timestamp": "time:timestamp",
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "resource": "org:resource",
    }
)
log = log.sort_values("time:timestamp")

log = log_converter.apply(log)

# COMMAND ----------

log

# COMMAND ----------

# MAGIC %md
# MAGIC # Process mining maps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Directly-Follows Graph
# MAGIC Denne type graf viser alle mulige noder og kanter

# COMMAND ----------

dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(
    dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY
)
dfg_visualization.view(gviz)

# COMMAND ----------

# MAGIC %md
# MAGIC Det er også mulig å se gjennomsnittlig tid mellom to eventer

# COMMAND ----------

dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
gviz = dfg_visualization.apply(
    dfg, log=log, variant=dfg_visualization.Variants.PERFORMANCE
)
dfg_visualization.view(gviz)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Heuristics Miner

# COMMAND ----------

# MAGIC %md
# MAGIC Heuristic Miner er en av de vanligste algoritmene man kan bruke i Process Mining. Den håndterer støy og klarer å vise den viktigste adferden som er representert i en event log. Plottet nedenfor er laget uten noen form for filtrering og kan ses på som en forenklet representasjon av plottene ovenfor.

# COMMAND ----------

heu_net = heuristics_miner.apply_heu(log)
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ulike typer filtrering med Heuristic Miner

# COMMAND ----------

# MAGIC %md
# MAGIC Plottet nedenfor viser en variant hvor vi kun ser på de 3 mest vanlige "prosessene". For å endre antall prosesser man ønsker å se på, kan 3-tallet i første kodelinje erstattes.

# COMMAND ----------

most_common_log = filter_variants_top_k(log, 3)
heu_net = heuristics_miner.apply_heu(most_common_log)
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)

# COMMAND ----------

# MAGIC %md
# MAGIC I den første grafen med Heuristic Miner ser vi at av 10.000 caser er det 200 som starter med aktiviteten "Payment". For å se kun disse casene kan vi filtrere på start activities. Det er også mulig å filtrere på end activities også. For å gjøre dette må man bytte ut "filter_start_activities" med "filter_end_activities" i første kodelinje.

# COMMAND ----------

starting_point_log = filter_start_activities(log, {"Payment"})
heu_net = heuristics_miner.apply_heu(starting_point_log)
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)

# COMMAND ----------

# MAGIC %md
# MAGIC Det er også mulig å filtrere på aktiviteter som enten direkte eller indirekte følger hverandre. I dette eksempelet har vi valgt å filtrere på caser hvor aktiviteten "Send Appeal to Prefecture" følger direkte etter "Add penalty". Det er også mulig å gjøre en indirekte filtrering hvor vi kun sier at "Send Appeal to Prefecture" skal være etter "Add penalty", men nødvendigvis ikke rett etter. Da byttes "filter_directly_follows_relation" i første kodelinje ut med "filter_eventually_follows_relation"

# COMMAND ----------

directly_follow_log = filter_directly_follows_relation(
    log, [("Add penalty", "Send Appeal to Prefecture")]
)
heu_net = heuristics_miner.apply_heu(directly_follow_log)
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)
