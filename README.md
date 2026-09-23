# OceanMind: A multi-agent AI system for ocean diagnosis

**Contents**

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Use CMEMS for Global Ocean Diagnosis](#use-cmems-for-global-ocean-diagnosis)
- [Example Queries](#example-queries)
- [Benchmarks](#benchmarks)
- [Citation and License](#citation-and-license)
- [Development](#development)

## Architecture

OceanMind is a natural-language workspace for analyzing time-dependent,
three-dimensional ocean data. It combines LLM-guided planning, executable
ocean-analysis tools, and an interactive map/chat interface to produce maps,
time series, statistics, and evidence-based interpretations.

- Natural-language analysis over Zarr and NetCDF datasets.
- Reusable workflows for spatial, temporal, vertical, event, and diagnostic
  analysis.
- Transparent plans, intermediate results, and interactive visualizations.
- A Next.js frontend backed by FastAPI and ocean-domain Python tools.

[![OceanMind interactive analysis and system architecture](assets/oceanmind-architecture.png)](assets/oceanmind-architecture.pdf)

## Quick Start

### Step 1: Install dependencies

Python 3.10 and Node.js 20 are recommended:

```bash
conda create -n ocean python=3.10 nodejs=20 -y
conda activate ocean
conda install -c conda-forge cartopy netcdf4 dask zarr numcodecs gsw -y
python -m pip install -U pip
python -m pip install -e ".[zarr]"
```

### Step 2: Configure environment

Copy the example configuration and set your API credentials in `.env`:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY="your_api_key"
OPENAI_BASE_URL="https://api.deepseek.com"
OPENAI_MODEL="deepseek-v4-pro"
```

Set `data_path` in `configs/dataset_config.yaml` to the directory containing
your ocean dataset. The other options in `.env.example` are optional.

### Step 3: Start

Linux and macOS:

```bash
bash start-server.sh
```

Windows:

```powershell
.\start-server.bat
```

The launcher prepares the frontend on first use and starts both services. Open
[http://localhost:3000](http://localhost:3000).

## Use CMEMS for Global Ocean Diagnosis

Convert downloaded CMEMS NetCDF files to OceanMind's per-variable Zarr stores:

```bash
python data/convert_cmems_to_oceanmind.py /path/to/cmems_nc /path/to/cmems_zarr
```

Point `configs/dataset_config.yaml` at the converted data:

```yaml
name: CMEMS
data_path: /path/to/cmems_zarr
backend: zarr
zarr_store_pattern: "CMEMS_{variable}.zarr"
```

Update the dataset metadata, then restart OceanMind:

```bash
python data/backfill_dataset_depths.py configs/dataset_config.yaml
```

## Example Queries

```text
Show the mean sea-surface temperature over the South China Sea from 2014 to 2022.

Plot January 2018 bottom salinity over 113E-124E and 13.5N-24.5N.

Find areas where bottom oxygen was below 60 mmol/m3 during summer 2020.

Compute the normal volume transport across a transect from 116E,18N to 121E,22N.
```

Adjust the region and dates to match your active dataset.

## Benchmarks

The benchmark record and evaluation protocol are available at
[https://doi.org/10.5281/zenodo.21189249](https://doi.org/10.5281/zenodo.21189249).

## Citation and License

If you use OceanMind in research or demonstrations, cite the OceanMind
manuscript and the underlying ocean datasets. A formal citation will be added
when available.

This repository does not currently include a license.

## Development

Please feel free to contact Fan Zhang ([mafzhang@ust.hk](mailto:mafzhang@ust.hk)),
Prof. Can Yang ([macyang@ust.hk](mailto:macyang@ust.hk)), or Prof. Jianping Gan
([magan@ust.hk](mailto:magan@ust.hk)) if any inquiries.
