"""Build a short-window live API suite from the frozen benchmark questions.

The original queries.json and its gold chains stay intact. Live runs have a
different purpose: exercise the deployed agent and UI contract once per task.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHORTER_WINDOWS = (
    (r"during summers from 2018 to 2020", "during June 2019 and June 2020"),
    (r"during spring seasons from 2018 to 2020", "during March 2019 and March 2020"),
    (r"January to March 2015", "January 1 to February 15, 2015"),
    (r"January to June 2015", "January 1 to February 28, 2015"),
    (r"2015-01-01 to 2015-03-31", "2015-01-01 to 2015-02-15"),
    (r"during the summer of 2018", "during June 2018"),
    (r"during summer 2015", "during June 2015"),
    (r"from 2017 to 2020", "from 2019 to 2020"),
    (r"from 2018 to 2020", "from 2019 to 2020"),
    (r"2018-2020", "2019-2020"),
)

ADDED = (
    ("water_mass_mechanism", "water_mass_identification", "In the northern South China Sea (113–117°E, 19–22°N), identify upper-300 m water masses from temperature and salinity during June 1–15, 2019. Explain which mixing or transport mechanisms the available data can support."),
    ("water_mass_mechanism", "plume_attribution", "What explains the low-salinity surface water near the Pearl River mouth (113–116°E, 20–23°N) during June 1–15, 2019? Compare freshwater influence and current-driven transport, and state what cannot be established without river discharge data."),
    ("water_mass_mechanism", "unsupported_nutrient_evidence", "Can nitrate supply explain the chlorophyll pattern near the South China Sea shelf break (115–118°E, 18–21°N) during June 1–15, 2019? Use available variables where possible and explicitly identify any missing evidence."),
    ("environmental_policy", "evidence_based_hypoxia_policy", "Draft a monitoring and mitigation plan for summer bottom-water hypoxia in the Bohai Sea (118–121°E, 38–40°N). Use August 1–15, 2019 CMOMS evidence where useful; distinguish supported findings from policy assumptions."),
    ("environmental_policy", "marine_protected_area_policy", "Design a transparent decision framework for a proposed marine protected area in the northern South China Sea. Which ecological and oceanographic indicators, stakeholders, and trade-offs should officials assess?"),
    ("general", "web_fact_lookup", "中国海最深的地方在哪里，大约有多深？请说明你采用的‘中国海’范围并给出可靠来源。"),
    ("general", "direct_ocean_explanation", "What is the difference between an eddy and an ocean front? Give a concise explanation without loading a dataset."),
    ("novel_code", "custom_centroid", "For June 1–7, 2019, calculate the daily longitude and latitude of the top-decile surface-chlorophyll centroid in 113–117°E, 19–22°N. Show its day-to-day displacement and the code used; this custom centroid need not match a named skill."),
    ("novel_code", "custom_vertical_contrast", "Define a daily vertical temperature-contrast index as the regional mean at the surface minus the regional mean at 100 m in 118–121°E, 38–40°N for January 1–15, 2019. Write the calculation, plot or tabulate the index, and interpret missing depths honestly."),
    ("novel_code", "custom_lag_correlation", "At 124°E, 30°N from March 1–20, 2019, compute lagged correlations between surface chlorophyll and temperature for lags −3 through +3 days using your own short Python routine. Explain the sign convention and why correlation does not establish causation."),
)


def shorten(query: str) -> str:
    for pattern, replacement in SHORTER_WINDOWS:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query


def build() -> list[dict]:
    original = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    if len(original) != 240:
        raise ValueError("Frozen benchmark suite changed; review the live suite")
    cases = []
    seen_types = set()
    for item in original:
        task_type = item["expected_skill"]
        if task_type in seen_types:
            continue
        seen_types.add(task_type)
        source_id = item["query_id"]
        question = shorten(item["query"])
        cases.append({
            "case_id": len(cases) + 1,
            "source_query_id": source_id,
            "category": item["suite"],
            "task_type": task_type,
            "query": question,
            "original_query": item["query"],
            "time_shortened": question != item["query"],
        })
    for category, task_type, question in ADDED:
        cases.append({
            "case_id": len(cases) + 1,
            "source_query_id": None,
            "category": category,
            "task_type": task_type,
            "query": question,
            "original_query": None,
            "time_shortened": False,
        })
    return cases


if __name__ == "__main__":
    output = HERE / "live_queries.json"
    cases = build()
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} live questions to {output}")
