# OceanMind

OceanMind is a natural-language ocean data analysis workspace. It combines an
LLM planner, executable ocean-domain tools, large-array data access, and a
Next.js map/chat interface so users can ask scientific questions over gridded
ocean datasets and receive maps, time series, statistics, and interpretation.

## Contents

- [Quick Start](#quick-start)
- [One-command server and Windows boot startup](#one-command-server-and-windows-boot-startup)
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

## One-command server and Windows boot startup

The launcher runs both services from the correct working directories, loads the
root `.env`, waits for backend readiness, and restarts both processes after a
child exits (10-second backoff). Production mode is the default: no reload,
one backend worker, and a prebuilt Next.js frontend. It does not install packages
or build assets on every reboot.

### First-time preparation (Windows, Linux or macOS)

Finish the installation and dataset configuration above. Build the frontend once:

```text
conda activate ocean
cd apps/web
npm ci
npm run build
cd ../..
```

After frontend changes, stop the running server and run the build again.
A successful preflight is not a test of dataset access or LLM credentials.

Windows (Anaconda Prompt or PowerShell after activating the environment):

```powershell
.\start-server.bat check
.\start-server.bat
```

Linux/macOS:

```bash
conda activate ocean
bash start-server.sh check
bash start-server.sh
```

Open http://localhost:3000. Keep this terminal open. Stop with Ctrl+C, or from
another terminal in the repository:

```text
python server.py stop
```

The stop command targets only this repository's launcher. It cannot stop servers
started manually with uvicorn/npm. Avoid running the launcher alongside those.
On Windows the managed child processes are terminated, so stop only when no
analysis is running. Linux/macOS receives SIGTERM before a bounded forced stop.

If Python is not on PATH, set `OCEANMIND_PYTHON` to the full path to the
installed environment's Python before running the wrapper. You can also call
that Python directly with `server.py`. Use
`--node "C:\path\to\node.exe"` if Node is not on PATH.
`--dev` is available for local development; do not use it for a boot task.

Logs rotate at 10 MB with three backups per service:

- `logs/server/supervisor.log`: startup, exit and restart events.
- `logs/server/backend.log`: backend errors and requests.
- `logs/server/frontend.log`: Next.js output.

Treat logs as private: application output can contain queries and provider errors.
The launcher does not log the contents of `.env`.

### Windows: start after reboot, even before login

Use Windows Task Scheduler, not the Startup folder (which runs only after login).
The installer registers a boot trigger with a 45-second delay, no execution-time
limit, no parallel duplicate task instances, and retry on supervisor failure.
It does not install a Windows Service, change firewall rules, or reboot the server.

1. In your working Conda environment, obtain the exact executable paths:

```powershell
conda activate ocean
python -c "import sys; print(sys.executable)"
(Get-Command node).Source
```

2. Open **PowerShell as Administrator**, enter the repository directory, and use
your actual paths (the following paths are examples):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-windows-startup.ps1 -PythonPath "C:\Miniconda3\envs\ocean\python.exe" -NodePath "C:\Program Files\nodejs\node.exe"
Start-ScheduledTask -TaskName OceanMind
```

The prompt asks for a Windows account password, not a PIN. Choose an account with
read access to the repository, Python/Node installations and datasets, and write
access to logs, outputs, caches and runtime state. The account must be allowed to
log on as a batch job. The task runs with limited privileges, not as SYSTEM.
Credentials are registered with Windows Task Scheduler, not stored in the scripts.
If the account password or installation paths change, update/reinstall the task.
The installation checks the invoking environment; verify permissions under the
chosen task account separately.

Do not start the manual launcher while the scheduled instance is running.
Check task state and recent result:

```powershell
Get-ScheduledTask -TaskName OceanMind
Get-ScheduledTaskInfo -TaskName OceanMind
Get-Content .\logs\server\backend.log -Tail 50
```

Verify http://localhost:3000, run a small dataset query, then perform a planned
reboot when no analysis is running and verify access **before interactive login**.
A reboot loses in-flight analyses; this feature restores the server, not the job.

For temporary maintenance, disable future startup and request a managed stop:

```powershell
Disable-ScheduledTask -TaskName OceanMind
& "C:\Miniconda3\envs\ocean\python.exe" .\server.py stop
```

Wait until the task is no longer running before editing/rebuilding. Re-enable
and start it using `Enable-ScheduledTask` and `Start-ScheduledTask`.
To uninstall, stop it as above, then:

```powershell
Unregister-ScheduledTask -TaskName OceanMind -Confirm
```

Avoid forcibly ending only the supervisor in Task Manager/Task Scheduler: child
processes may survive. Use the managed stop command. Network drives mapped in an
interactive session may not exist at boot; use local storage or a UNC path with
appropriate account permissions. Dataset/network availability can delay startup.

### Remote access

Defaults are loopback-only. For a trusted LAN/VPN deployment, use
`--web-host 0.0.0.0` on the launcher, or `-WebHost 0.0.0.0` when installing
the task. Browse to http://SERVER-IP:3000, not http://0.0.0.0:3000.
The backend stays on 127.0.0.1; the frontend proxy is configured automatically.
Custom ports use `--web-port` / `--api-port` (installer: `-WebPort` / `-ApiPort`).

Do not expose this unauthenticated analysis interface directly to the public
Internet. Restrict inbound access to trusted addresses, or use an authenticated
HTTPS reverse proxy/VPN. Opening firewall ports is an explicit administrator step.

Task Scheduler reference:
https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/register-scheduledtask

### Launch with a public domain

Pass the external URL separately from the local listen address:

```powershell
.\start-server.bat --public-url "https://oceanmind.wavyocean.hkust.edu.hk/"
```

```bash
bash start-server.sh --public-url "https://oceanmind.wavyocean.hkust.edu.hk/"
```

The argument validates and displays the public origin and passes it to child
processes as `OCEANMIND_PUBLIC_URL`. The current frontend uses same-origin API
requests; it needs no hard-coded domain or rebuild when this URL changes.
**This option is not a DNS, certificate or reverse-proxy installer.**
It does not change the listen address or verify that the public site is reachable.

For HTTPS access, your existing DNS must point to the ingress/proxy and that
proxy must terminate TLS for this domain and forward requests to the Next.js
frontend (normally `http://127.0.0.1:3000` on the same host), not directly to
the Python backend. Forward all paths, including `/api/*`; preserve streaming
responses (disable response buffering) and allow long analysis requests.
Keep the proxy itself configured for startup after reboot.

If the proxy runs on another machine, add `--web-host 0.0.0.0` and allow the
frontend port only from that proxy's trusted address. Keep authentication/access
restrictions at the proxy; the public URL argument does not add authentication.

To persist the URL in a newly installed Windows startup task:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-windows-startup.ps1 -PythonPath "C:\Miniconda3\envs\ocean\python.exe" -NodePath "C:\Program Files\nodejs\node.exe" -PublicUrl "https://oceanmind.wavyocean.hkust.edu.hk/"
```

Use your actual executable paths. If the OceanMind task already exists, stop
and unregister it using the maintenance instructions above before reinstalling;
the installer intentionally refuses to overwrite an existing task.

### Managed Nginx HTTPS (Windows)

To have the launcher also run Nginx, add `--nginx`, `--tls-cert` and `--tls-key`.
This is different from `--public-url` alone: it generates and tests an Nginx
configuration, starts the proxy, and restarts the managed stack if a process exits.

1. Download and unzip the official Windows package from https://nginx.org/en/download.html
   (for example to `C:\nginx`). Do not separately start nginx.exe.
2. Obtain a valid PEM full-chain certificate and PEM private key for
   `oceanmind.wavyocean.hkust.edu.hk`. Store them outside Git, with restricted
   file permissions. The key must not require an interactive password.
   This script does not issue or renew certificates.
3. After installing dependencies and building the frontend, run from the repository:

```powershell
.\start-server.bat --public-url "https://oceanmind.wavyocean.hkust.edu.hk/" --nginx "C:\nginx\nginx.exe" --tls-cert "C:\certs\fullchain.pem" --tls-key "C:\certs\privkey.pem"
```

Traffic follows: domain HTTPS:443 -> Nginx -> 127.0.0.1:3000 -> 127.0.0.1:8000.
Port 80 redirects to HTTPS. Both 80 and 443 must be free, and network/firewall
policy must permit your intended users. No firewall or DNS changes are performed.
Keep web-host at its loopback default. Check public access from another computer.
The template is at `nginx.conf.template`; the generated config and isolated
Nginx prefix are under `.run/nginx/`. Existing Nginx configuration is not overwritten.
Nginx stderr goes to rotating `logs/server/nginx.log`; access logging is disabled.

To include Nginx in the Windows boot task, run in administrator PowerShell
(replace executable and certificate paths with your real ones):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-windows-startup.ps1 -PythonPath "C:\Miniconda3\envs\ocean\python.exe" -NodePath "C:\Program Files\nodejs\node.exe" -PublicUrl "https://oceanmind.wavyocean.hkust.edu.hk/" -NginxPath "C:\nginx\nginx.exe" -TlsCert "C:\certs\fullchain.pem" -TlsKey "C:\certs\privkey.pem"
Start-ScheduledTask -TaskName OceanMind
```

Stop the manual instance before starting the task. For an existing task, stop and
unregister it first as described above. The task account must read the certificate
and private key. After certificate renewal, perform a managed stop/start to reload.
On Linux/macOS, binding ports 80/443 requires appropriate privileges; prefer a
separate OS-managed Nginx service rather than running the whole analysis app as root.

This is transport encryption, not user authentication. Before public use, restrict
access via your institutional network/firewall or add authentication at the proxy.
Windows Nginx has documented scalability limitations; see
https://nginx.org/en/docs/windows.html and
https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering.
