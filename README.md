# OpenBankingAPI
This repository contains Python scripts and data files for scraping, compiling, and analyzing Open Banking API performance data.

## Repository Contents
- `webscrap.py` – Python script for web scraping HTML data  
- `results.py` – Python script for processing compiled data  
- `Monthly_data.zip` – Zipped folder containing monthly HTML data files  
- `datacompile.xlsx` – Excel file used for storing compiled data  

## Requirements
- Python 3.8 or higher  
- Required packages:  
  - pandas  
  - numpy  
  - beautifulsoup4  
  - requests  
  - openpyxl  

Install dependencies with:  
```bash
pip install -r requirements.txt
``` 

## Usage
### 1. Web Scraping
- Unzip Monthly_data.zip.
- Place all HTML files in the same folder as webscrap.py.
- Run the scraper:
```bash
python webscrap.py
```

### 2. Processing Results
- Ensure datacompile.xlsx is in the same folder as results.py.
- Run the results script:
```bash
python results.py
```

## Notes
Keep all required files in the same working directory for scripts to function correctly.
This project is for research and educational purposes only.
