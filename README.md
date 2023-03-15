# GPS_Final_Project
Final project for AA272
Teammates: Renee Nguyen, Pol Francesch Huc, Jacob Azoulay

## File structure
- data: contains all data files logged and processed.
- GNSS_2_ECEF: contains all files pertinent to our attempt at getting the ECEF position history directly from the raw GNSS measurements.

## Environment Setup
This environment setup assumes you have installed Python (version >= 3.8) on your machine. 
```
python -m venv gps_final_project
```
Windows:
```
gps_final_project\Scripts\activate
```
MacOS/Linux:
```
source gps_final_project/bin/activate
```

Installing libaries:
```
pip install -r ./requirements.txt
```

## How to run
Please ensure you have installed the appropiate python libraries.

```
python fileCrawl.py
python analyze_fix.py
```
