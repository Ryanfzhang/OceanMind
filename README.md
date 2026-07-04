# OceanMind

OceanMind is a natural-language ocean data analysis workspace. It combines an
LLM planner, executable ocean-domain tools, large-array data access, and a
Next.js map/chat interface so users can ask scientific questions over gridded
ocean datasets and receive maps, time series, statistics, and interpretation.

## Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Example Queries](#example-queries)
- [Built-in Skills](#built-in-skills)
- [Architecture](#architecture)

## Quick Start

### 1. Install the backend

Use Python 3.10. The scientific stack is easiest to install through Conda.

```bash
conda create -n ocean python=3.10 -y
conda activate ocean
conda install -c conda-forge cartopy netcdf4 dask zarr numcodecs gsw -y
python -m pip install -U pip
python -m pip install -e ".[zarr]"
```


### 2. Install the frontend

The frontend requires Node.js and npm. If `npm` is not available in your shell,
install Node.js from Conda first:

```bash
conda install -c conda-forge nodejs=20 -y
```

This installs both `node` and `npm`.

```bash
cd apps/web
npm install
cd ../..
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with an OpenAI-compatible LLM endpoint:

```env
OPENAI_API_KEY="your_api_key"
OPENAI_BASE_URL="https://api.deepseek.com"
OPENAI_MODEL="deepseek-v4-pro"
QUERY_ROUTER_MODEL="deepseek-v4-flash"
PLANNER_MODEL="deepseek-v4-pro"
PLANNER_SELECTOR_MODEL="deepseek-v4-flash"
WEB_ANSWER_MODEL="deepseek-v4-pro"
RESULT_SYNTHESIZER_MODEL="deepseek-v4-pro"
```

`BACKEND_API_BASE_URL` controls the frontend proxy target and defaults to
`http://127.0.0.1:8000`.

### 4. Start the backend

```bash
conda activate ocean
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Start the frontend

```bash
cd apps/web
npm run dev
```

Open `http://localhost:3000`.

## Configuration

### Dataset

The active dataset is configured in `configs/dataset_config.yaml`. The default
configuration points to a Zarr-backed ocean dataset:

```yaml
data_path: your_path_to_dataset
backend: zarr
zarr_store_pattern: "{name}_{variable}.zarr"
```

Update `data_path` if your dataset is stored elsewhere. The standard variables
used by the current workflows are:

```text
temp, salt, u, v, chlorophyll, oxygen
```

Inspect the active dataset after starting the backend:

```bash
curl http://127.0.0.1:8000/dataset
```

### Use CMEMS data

OceanMind can use CMEMS NetCDF downloads after converting them to the
per-variable Zarr layout used by the backend.

```bash
python data/convert_cmems_to_oceanmind.py /path/to/raw_cmems_nc /path/to/oceanmind_zarr --overwrite
```

Then point `configs/dataset_config.yaml` at the converted directory:

```yaml
name: CMEMS
data_path: /path/to/oceanmind_zarr
backend: zarr
zarr_store_pattern: "CMEMS_{variable}.zarr"
```

Backfill the dataset metadata, then restart the backend:

```bash
python data/backfill_dataset_depths.py configs/dataset_config.yaml
curl http://127.0.0.1:8000/dataset
```

## Example Queries

Spatial fields:

```text
Show me the SST mean from 2014 to 2022 over the South China Sea.
```

```text
Plot January 2018 bottom salinity over 113E-124E and 13.5N-24.5N.
```

Time series and trends:

```text
Analyze the summer sea-temperature trend in the South China Sea from 2011 to 2022.
```

```text
Compute a regional mean oxygen time series for 110E-120E, 18N-23N from 2015 to 2020.
```

Events and mechanisms:

```text
Show areas in the South China Sea where bottom dissolved oxygen is below 60 mmol/m3 during summer 2020.
```

```text
Compare chlorophyll bloom events and low-oxygen events near the Pearl River Estuary.
```

Transects and dynamics:

```text
Compute the normal volume transport across a transect from 116E,18N to 121E,22N.
```

```text
Map relative vorticity for surface currents in the northern South China Sea.
```

## Built-in Skills

OceanMind selects analysis workflows from `skills/*/SKILL.md`. Each skill
describes when it applies, which tool calls are allowed, required parameters,
artifact naming, and interpretation rules.

Representative skill groups:

- Dataset information and field visualization.
- Spatial fields, climatology, anomalies, histograms, and EOF analysis.
- Regional time series, resampling, trends, regression, and lag correlation.
- Profiles, sections, Hovmoller diagrams, layer means, and vertical integrals.
- Derived fields including density, vorticity, stratification, and diagnostics.
- Event workflows for heatwaves, hypoxia, blooms, upwelling, fronts, eddies,
  jets, meanders, and eutrophication.
- Transport, budget, water-mass, and mechanism-ranking workflows.
- Environmental health assessment, evidence synthesis, and policy-oriented
  recommendation workflows.

## Architecture

```text
Next.js workspace
    |
    |  /api/query/stream proxy
    v
FastAPI backend
    |
    |-- query router and memory
    |-- harness planner / skill planner
    |-- executable task graph
    |-- tool executor
    |
    v
domain/ocean tools
    |
    |-- data access and preprocessing
    |-- spatial, time-series, vertical, event, and diagnostic analysis
    |-- visualization and result payload generation
    |
    v
Zarr / NetCDF ocean datasets
```

Important directories:

```text
apps/api/             FastAPI backend and streaming query endpoints
apps/web/             Next.js frontend workspace
configs/              Dataset and water-mass configuration
domain/ocean/         Ocean analysis, diagnostics, events, and visualization
packages/             Agent runtime, planner, memory, LLM gateway, tool loading
skills/               Workflow manuals used by the planner
tests/                Backend and frontend-facing behavior tests
```


## Citation and License

Citation information is not yet included in this repository. If you use
OceanMind in research or demos, cite this repository and any underlying ocean
datasets used in your analysis.

No license file is currently included. Add a project license before public
redistribution.
