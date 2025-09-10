# Formula Kite Data Analysis  

This repository contains the datasets and analyses from the **Formula Kite testing campaign** held in **Port Camargue (France), June 6–11, 2025**.  
The project focuses on two main aspects of performance:  
- Straight runs (upwind & downwind speed testing)  
- Maneuvers (tacks & gybes)  

---

## Repository Structure  

```text
FormulaKiteDataAnalysis
│   README.md
│   TREE.md
│   __pycache__/                 # auto-generated cache files (ignore)
│
├── Data_Sailnjord/              # processed datasets ready for Python analysis
│   ├── Maneuvers/               # maneuver runs
│   │   ├── 08_06/               # date
│   │   │   ├── Gian/Run*/       # athlete + runs (SenseBoard.csv, GPS)
│   │   │   ├── Karl/Run*/       # athlete + runs
│   │   │   ├── Interview and equipment/  # equipment metadata (.xlsx)
│   │   │   └── senseboard_log/           # SenseBoard logs
│   │   └── 11_06/ (same structure)
│   │
│   └── Straight_lines/          # straight runs
│       ├── 06_06/               # date
│       │   ├── Run*/            # runs (Gian, Karl, SenseBoard)
│       │   ├── Interview and equipment/
│       │   └── senseboard_log/
│       ├── 07_06/
│       ├── 09_06/
│       └── 10_06/
│
├── Maneuvers/                   # analysis of maneuvers
│   ├── *.ipynb / *.py           # analysis notebooks & scripts
│   ├── all_data*.csv            # consolidated datasets
│   ├── summary.json
│   ├── old/                     # previous versions
│   └── __pycache__/             # auto-generated (ignore)
│
├── Straight Run/                # analysis of straight runs
│   ├── *.ipynb / *.py           # analysis notebooks & scripts
│   ├── all_data*.csv            # consolidated datasets
│   ├── summary*.json
│   ├── archives/                # historic reports & notebooks
│   └── __pycache__/             # auto-generated (ignore)
│
└── Test Kite Port Camargue/     # pre-Python processed materials
    ├── Equipment files, logs, protocols
    └── Raw data/                # raw campaign datasets (SenseBoard, Vakaros, Wind/Marks)

## Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your_username>/<your_repo>.git
   cd <your_repo>
