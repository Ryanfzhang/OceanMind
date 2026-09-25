"""Build a short-window live API suite from the frozen benchmark questions.

The original queries.json and its gold chains stay intact. Live runs have a
different purpose: exercise the deployed agent and UI contract once per task.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REMOVED_DUPLICATES = {152: 149}  # Same transport task, transect, and dates.
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
    ("water_mass_mechanism", "In the northern South China Sea (113–117°E, 19–22°N), identify upper-300 m water masses from temperature and salinity during June 1–15, 2019. Explain which mixing or transport mechanisms the available data can support."),
    ("water_mass_mechanism", "What explains the low-salinity surface water near the Pearl River mouth (113–116°E, 20–23°N) during June 1–15, 2019? Compare freshwater influence and current-driven transport, and state what cannot be established without river discharge data."),
    ("water_mass_mechanism", "Compare the upper-200 m temperature–salinity structure east of Taiwan (121–125°E, 22–26°N) in January 1–15 and July 1–15, 2019. What evidence would support Kuroshio intrusion versus seasonal surface heating?"),
    ("water_mass_mechanism", "Why might bottom oxygen be low in the Bohai Sea (118–121°E, 38–40°N) during August 1–15, 2019? Examine stratification, currents, and surface chlorophyll, and separate association from causation."),
    ("water_mass_mechanism", "During June 1–15, 2019, do temperature and salinity anomalies on the East China Sea shelf (123–127°E, 28–32°N) suggest a changed water mass or local surface forcing? Show the diagnostic evidence and its limits."),
    ("water_mass_mechanism", "Can nitrate supply explain the chlorophyll pattern near the South China Sea shelf break (115–118°E, 18–21°N) during June 1–15, 2019? Use available variables where possible and explicitly identify any missing evidence."),
    ("environmental_policy", "Draft a monitoring and mitigation plan for summer bottom-water hypoxia in the Bohai Sea (118–121°E, 38–40°N). Use August 1–15, 2019 CMOMS evidence where useful; distinguish supported findings from policy assumptions."),
    ("environmental_policy", "For recurring algal blooms off the Yangtze estuary (121–124°E, 29–33°N), propose a practical early-warning and response policy. Analyze March 1–15, 2019 chlorophyll and physical conditions, and explain what additional observations are needed."),
    ("environmental_policy", "A coastal authority wants to reduce marine heatwave impacts near Hong Kong (113–116°E, 20–23°N). Recommend monitoring triggers and response actions using June 1–15, 2019 temperature as an example, with uncertainty clearly stated."),
    ("environmental_policy", "What evidence should be required before restricting nutrient discharges to address low oxygen in the Bohai Sea? Explain monitoring design, causal limitations, and how the policy should be evaluated."),
    ("environmental_policy", "Design a transparent decision framework for a proposed marine protected area in the northern South China Sea. Which ecological and oceanographic indicators, stakeholders, and trade-offs should officials assess?"),
    ("environmental_policy", "Compare two policy options for East China Sea coastal algal blooms: stronger nutrient controls and expanded real-time monitoring. Give a phased recommendation, measurable success criteria, and key uncertainties."),
    ("general", "中国海最深的地方在哪里，大约有多深？请说明你采用的‘中国海’范围并给出可靠来源。"),
    ("general", "什么是海洋热浪？常用定义中的持续时间和温度阈值是什么？"),
    ("general", "What is the difference between an eddy and an ocean front? Give a concise explanation without loading a dataset."),
    ("general", "What are the main seas bordering mainland China, and how do their typical depths differ? Cite reliable sources."),
    ("general", "为什么海水越深不一定越冷？请解释温跃层和水团的作用。"),
    ("general", "What can a temperature–salinity diagram reveal about water masses, and what can it not establish on its own?"),
    ("novel_code", "For June 1–7, 2019, calculate the daily longitude and latitude of the top-decile surface-chlorophyll centroid in 113–117°E, 19–22°N. Show its day-to-day displacement and the code used; this custom centroid need not match a named skill."),
    ("novel_code", "Define a daily vertical temperature-contrast index as the regional mean at the surface minus the regional mean at 100 m in 118–121°E, 38–40°N for January 1–15, 2019. Write the calculation, plot or tabulate the index, and interpret missing depths honestly."),
    ("novel_code", "In 123–125°E, 29–31°N during March 1–15, 2019, calculate a custom daily chlorophyll patchiness index: spatial standard deviation divided by spatial mean, with safe handling of zero or missing means. Report the series and code."),
    ("novel_code", "At 115°E, 20°N in June 1–20, 2019, find the first day of three consecutive days when surface temperature exceeds the 75th percentile of that 20-day series. Write ordinary Python for the run-length rule and state if no day qualifies."),
    ("novel_code", "For 118–121°E, 38–40°N during August 1–15, 2019, create a daily bottom-oxygen spatial inequality score using the Gini coefficient across valid grid cells. Implement the statistic in code and show how missing values were handled."),
    ("novel_code", "At 124°E, 30°N from March 1–20, 2019, compute lagged correlations between surface chlorophyll and temperature for lags −3 through +3 days using your own short Python routine. Explain the sign convention and why correlation does not establish causation."),
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
    for item in original:
        source_id = item["query_id"]
        if source_id in REMOVED_DUPLICATES:
            continue
        question = shorten(item["query"])
        cases.append({
            "case_id": len(cases) + 1,
            "source_query_id": source_id,
            "category": item["suite"],
            "query": question,
            "original_query": item["query"],
            "time_shortened": question != item["query"],
        })
    for category, question in ADDED:
        cases.append({
            "case_id": len(cases) + 1,
            "source_query_id": None,
            "category": category,
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
