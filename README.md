Weather-dataset-visualisation-code


This project is a single-file FastAPI application that allows you to upload datasets (CSV, Excel, JSON, Parquet) and automatically performs lightweight Exploratory Data Analysis (EDA). When a file is uploaded, the app intelligently parses it, detects column types (numeric, datetime, categorical, boolean, text, geo), generates profiling details such as missing values and summary statistics, and provides recommended visualizations based on the data. It also includes a built-in HTML + Plotly interface, so no separate frontend is required. To run the app, simply install the required Python packages, execute `python app_enhanced.py`, and open `http://localhost:8080` in your browser to start uploading and analyzing datasets.
