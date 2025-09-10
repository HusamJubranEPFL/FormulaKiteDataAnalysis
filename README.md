# Formula Kite Data Analysis  

This repository contains the datasets and analyses from the **Formula Kite testing campaign** held in **Port Camargue (France), June 6–11, 2025**.  
The project focuses on two main aspects of performance:  
- Straight runs (upwind & downwind speed testing)  
- Maneuvers (tacks & gybes)  

The work combines raw field data, systematic filtering, and advanced statistical modeling to provide insights into kitefoil performance.  

---

## Repository Structure  

- `Data_Sailnjord/`  
  - `Maneuvers/` — maneuver runs organized by date (`08_06`, `11_06`), athlete (`Gian`, `Karl`), and run (`Run1…`).  
    Contains `SenseBoard.csv` files per run, plus equipment interviews and SenseBoard logs.  
  - `Straight_lines/` — straight-line runs (`06_06` to `10_06`) with the same structure (athletes, runs, SenseBoard, equipment interviews, and logs).  

- `Maneuvers/`  
  Analysis code & notebooks for maneuver studies: `MainCOG.ipynb`, `Report_*_Tack/Jibe.ipynb`, `cog_analysis.py`, `report_fct.py`, `runner.ipynb`.  
  Includes consolidated datasets (`all_data.csv`, `all_data_enriched.csv`, `summary.json`).  
  Subfolders: `old/` (previous versions), `__pycache__/` (ignore).  

- `Straight Run/`  
  Analysis code & notebooks for straight runs: `analysis.ipynb/.py`, `MainReport.ipynb`, `cog_analysis.py`, `report_fct.py`, statistical tests (`mast_ttest.ipynb`, `weight_ttest.ipynb`).  
  Includes consolidated datasets (`all_data.csv`, `all_data_enriched.csv`, `summary.json`, `summary_enriched.json`).  
  Subfolders: `archives/` (historic reports & notebooks), `__pycache__/` (ignore).  

- `Test Kite Port Camargue/`  
  Pre-Python processed materials from the testing campaign (documentation, logs, and intermediate files prepared before the Python analysis began).  

- `__pycache__/`  
  Auto-generated Python cache files (not relevant for analysis).  

---

## Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your_username>/<your_repo>.git
   cd <your_repo>
