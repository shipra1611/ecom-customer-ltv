#!/bin/sh
# Expects data/raw/online_retail_II.csv (UCI Online Retail II, https://archive.ics.uci.edu/dataset/502)
set -e
mkdir -p data outputs tableau
cd src
python 01_prepare_real_data.py
python 02_load_sql.py
python 03_rfm_kmeans.py
python 04_cohorts.py
python 05_ltv_xgboost.py
