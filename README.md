# Bikelution
Data and source code for the paper "Bikelution: Federated Gradient-Boosting for Scalable Shared Micro-Mobility Demand Forecasting"

## Table of Contents
- [Overview](#overview)
- [Prerequisites & installation](#prerequisites--installation)
- [Data preparation](#data-preparation)
- [Model training & inference](#model-training--inference)  
- [Contributors](#contributors)
- [Acknowledgement](#acknowledgement)


# Overview  
This repository contains the code required to train the **Bikelution** model using a federated learning (FL) approach.  
For the centralized learning variant, refer to the Shared‑Mobility repository: <https://github.com/DataStories-UniPi/Shared-Mobility>.


# Prerequisites & installation
1. Python 3.10
1. System libraries – git, curl, wget (for dataset download)
1. Python packages (cf., requirements.txt)
```bash
pip install -r requirements.txt
```

# Dataset Preparation  
The repository expects the bike‑sharing datasets (e.g., *NYC*, *Chicago*, *Barcelona*) stored as Parquet files with the following layout:

```
data/
 └─ <dataset_name>/h6_w168_multi/
      └─ <split>=train|validation|test   # Parquet file containing tree predictions
```

To generate these files, run the preprocessing pipeline from the Shared‑Mobility project. The datasets are publicly available at <https://citibikenyc.com/system-data> (NYC dataset), <https://divvybikes.com/system-data> (Chicago dataset), and <https://doi.org/10.5281/zenodo.17650616> (Barcelona dataset). To create the individual Parquet files for each client (i.e., bike station) use the following Python script:

```bash
# Create a sliced (per-client) Parquet view for the desired split (e.g., train)
python 1-dataset-federation.py --dataset citi --slice train
```


# Model training & inference 

For the centralized variant, refer to the [Shared‑Mobility](https://github.com/DataStories-UniPi/Shared-Mobility) repository. For the federated training of Bikelution, run the following Python script:

```bash
# Launch Bikelution FL simulation
python 2-launch-fl-simulation.py \
    --federation citi \
    --n_estimators 37 \
    --bs 64 \
    --parquet_bs 4096 \
    --num_rounds 15 \
    --local_epochs 10 \
    --early_stop \
    --patience 5 \
    --mu 0.125 \
    --conv_channels 32 \
    --dropout_rate 0.13 \
    --fraction_fit 0.25 \
    --fraction_eval 0.25
```

The simulation logs per‑client files under `data/logs/` and saves model checkpoints in `data/pth/`. To obtain forecasts from the global FL model, execute the code in notebook `bikelution-inference.ipynb`. Refer to the [Shared‑Mobility](https://github.com/DataStories-UniPi/Shared-Mobility) repository for inference on the centralized model. Use notebook `bikelution-metrics.ipynb` to generate the comparison tables.


# Contributors
- Antonis Tziorvas; Department of Informatics, University of Piraeus
- Andreas Tritsarolis; Department of Informatics, University of Piraeus
- Yannis Theodoridis; Department of Informatics, University of Piraeus


# Acknowledgement
This work was supported in part by the EU Horizon Framework Programme under Grant Agreement No. 101093051 (EMERALDS; <https://www.emeralds-horizon.eu/>) and EU Horizon Europe R\&I Programme under Grant Agreement No. 101070416 (Green.Dat.AI; <https://greendatai.eu>).