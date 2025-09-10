# Formula Kite Data Analysis  

This repository contains the datasets and analyses from the **Formula Kite testing campaign** held in **Port Camargue (France), June 6–11, 2025**.  
The project focuses on two main aspects of performance:  
- Straight runs (upwind & downwind speed testing)  
- Maneuvers (tacks & gybes)  

---

## Repository Structure  
# FormulaKiteDataAnalysis — Repository Structure (synthetic view)

> Purpose: provide a clear overview, without unnecessary repetition, to understand the general architecture and data flow.

```
FormulaKiteDataAnalysis/
├── README.md
├── TREE.md  ← (this document)
│
├── Data_Sailnjord/              # processed datasets ready for Python analysis
│   ├── Maneuvers/
│   │   ├── 08_06/
│   │   │   ├── Gian/ ── 08_06_Run{1..5}/ (SenseBoard.csv)
│   │   │   ├── Karl/ ── 08_06_Run{1..6}/ (Karl Maeder.csv)
│   │   │   ├── senseboard_log/ (SenseBoard_log_modified_250608.xlsx)
│   │   │   └── Interview and equipment/ (Interview *.xlsx)
│   │   └── 11_06/
│   │       ├── Gian/ ── 11_06_Run{1..5}/ (Gian Stragiotti.csv | SenseBoard.csv)
│   │       ├── Karl/ ── 11_06_Run{1..6}/ (Karl Maeder.csv)
│   │       ├── senseboard_log/ (…250611…)
│   │       └── Interview and equipment/ (…250611…)
│   │
│   └── Straight_lines/
│       ├── 06_06/ ── 06_06_Run{1..8}/ (Gian|Karl|SenseBoard.csv) + interviews/logs
│       ├── 07_06/ ── 07_06_Run{1..10}/ (Gian|Karl|SenseBoard.csv) + interviews/logs
│       ├── 09_06/ ── 09_06_Run{1..11}/ (Karl|SenseBoard.csv) + interviews/logs
│       └── 10_06/ ── 10_06_Run{1..10}/ (Gian|Karl|SenseBoard.csv) + interviews/logs
│
├── Maneuvers/
│   ├── analysis notebooks & scripts (addsenseboarddata.ipynb, MainCOG.ipynb,
│   │   Report_*_*.ipynb, cog_analysis.py, report_fct.py, Report_with_eval.py…)
│   ├── aggregated data (all_data*.csv, summary*.json)
│   ├── old/ (legacy notebooks)
│   └── __pycache__/
│
├── Straight Run/
│   ├── analysis notebooks (MainReport.ipynb, analysis*.ipynb, *ttest.ipynb…)
│   ├── scripts (analysis.py, cog_analysis.py, report_fct.py, merge_all.ipynb…)
│   ├── aggregated data (all_data*.csv, summary*.json)
│   ├── archives/ (rendered reports, html/pdf, older versions)
│   └── __pycache__/
│
├── Test Kite Port Camargue/     # pre-Python processed materials
│   ├── campaign documents (equipment lists, protocols, pilot logs)
│   ├── Raw data/ organized by Day1..Day6
│   │   ├── SenseBoard bin/ (raw .bin files)
│   │   ├── SenseBoard post-processing/ (scripts + _imu_log_*.csv, test_forces_*.csv)
│   │   ├── Vakaros csv/ (raw exports)
│   │   ├── Vakaros post-processing/ (Cells Renamed/*, Lines/*.xlsx)
│   │   └── Wind and Marks/ (wind & marks logs *.csv)
│   └── Detrending filter/ (scripts + correction spreadsheets)
│
└── __pycache__/
```
