"""
Enhanced single-file FastAPI EDA app

Features:
- Supports CSV, TSV, XLSX (multiple sheets), JSON (ndjson & nested), Parquet
- Automatic type detection: numeric, datetime, categorical, boolean, text, geo (lat/lon), nested JSON flattening
- Profiling and recommended visualizations per dataset
- Interactive Plotly frontend (embedded) similar to original app

Install requirements:
    pip install fastapi uvicorn pandas numpy python-multipart plotly openpyxl pyarrow
    # duckdb is optional for very large files: pip install duckdb

Run:
    python app_enhanced.py

Open:
    http://localhost:8000
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import pandas as pd
import numpy as np
import io
import json
from typing import List, Dict, Any
import tempfile
import os

app = FastAPI(title="Generic EDA Web Tool — Single File")

# ---------- Helpers ----------

def safe_read_csv(s: str, sep=',', nrows=None):
    try:
        return pd.read_csv(io.StringIO(s), nrows=nrows)
    except Exception:
        # try with python engine
        return pd.read_csv(io.StringIO(s), engine='python', sep=sep, nrows=nrows)


def parse_uploaded_file(contents: bytes, filename: str, sample_rows: int = 1000):
    """Return a dict of {name: DataFrame} — for single-sheet returns {'sheet1': df}
    Tries to handle csv, tsv, xlsx, json (ndjson or json array), parquet.
    """
    name = filename.lower()
    tmp = None
    try:
        if name.endswith('.csv') or name.endswith('.txt'):
            s = None
            try:
                s = contents.decode('utf-8')
            except UnicodeDecodeError:
                s = contents.decode('latin1', errors='ignore')
            df = safe_read_csv(s, sep=',', nrows=sample_rows)
            return {os.path.splitext(filename)[0]: df}

        elif name.endswith('.tsv') or '\t' in contents.decode('utf-8', errors='ignore')[:200]:
            s = contents.decode('utf-8', errors='ignore')
            df = safe_read_csv(s, sep='\t', nrows=sample_rows)
            return {os.path.splitext(filename)[0]: df}

        elif name.endswith('.xlsx') or name.endswith('.xls'):
            # write to temp file and use pandas.read_excel with sheet_name=None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            tmp.write(contents); tmp.close()
            sheets = pd.read_excel(tmp.name, sheet_name=None)
            # limit rows per sheet for safety
            sheets = {k: v.head(sample_rows) for k,v in sheets.items()}
            return sheets

        elif name.endswith('.json'):
            # try ndjson first
            s = None
            try:
                s = contents.decode('utf-8')
            except Exception:
                s = contents.decode('latin1', errors='ignore')
            lines = [l for l in s.splitlines() if l.strip()]
            if len(lines) > 1 and all(l.strip().startswith('{') for l in lines[:5]):
                # ndjson
                records = [json.loads(l) for l in lines[:sample_rows]]
                df = pd.json_normalize(records)
                return {os.path.splitext(filename)[0]: df}
            else:
                # try full json array
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        df = pd.json_normalize(obj[:sample_rows])
                        return {os.path.splitext(filename)[0]: df}
                    elif isinstance(obj, dict):
                        # single object -> flatten
                        df = pd.json_normalize([obj])
                        return {os.path.splitext(filename)[0]: df}
                except Exception:
                    pass
                # fallback to csv parse
                df = safe_read_csv(s, sep=',', nrows=sample_rows)
                return {os.path.splitext(filename)[0]: df}

        elif name.endswith('.parquet'):
            try:
                # pandas will use pyarrow/fastparquet
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
                tmp.write(contents); tmp.close()
                df = pd.read_parquet(tmp.name)
                return {os.path.splitext(filename)[0]: df.head(sample_rows)}
            except Exception:
                # try reading with duckdb from buffer
                raise

        else:
            # unknown extension — attempt csv then json
            try:
                s = contents.decode('utf-8')
            except Exception:
                s = contents.decode('latin1', errors='ignore')
            # try csv
            try:
                df = safe_read_csv(s, sep=',', nrows=sample_rows)
                return {os.path.splitext(filename)[0]: df}
            except Exception:
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        df = pd.json_normalize(obj[:sample_rows])
                        return {os.path.splitext(filename)[0]: df}
                except Exception:
                    pass
            raise ValueError('Unsupported file or failed to parse')
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping column -> detected type: numeric, datetime, categorical, boolean, text, geo"""
    types = {}
    for col in df.columns:
        ser = df[col]
        # drop nulls for detection
        s = ser.dropna()
        if s.empty:
            types[col] = 'unknown'
            continue
        # booleans
        if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()) <= {0,1,True,False}:
            types[col] = 'boolean'; continue
        # numeric
        if pd.api.types.is_numeric_dtype(s):
            types[col] = 'numeric'; continue
        # datetime
        try:
            parsed = pd.to_datetime(s.sample(min(len(s), min(100, len(s)))), errors='coerce')
            if parsed.notnull().sum() >= max(1, int(0.6 * len(parsed))):
                types[col] = 'datetime'; continue
        except Exception:
            pass
        # geo detection: lat/lon pairs
        if col.lower() in ('lat','latitude') or col.lower() in ('lon','lng','longitude'):
            types[col] = 'geo'; continue
        # small cardinality -> categorical
        unique_frac = s.nunique(dropna=True) / max(1, len(s))
        if unique_frac < 0.05 and s.nunique() < 200:
            types[col] = 'categorical'; continue
        # text/varchar
        if pd.api.types.is_string_dtype(s):
            types[col] = 'text'; continue
        # fallback
        types[col] = 'unknown'
    return types


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a lightweight profile — per-column stats and dataset-level notes"""
    profile = {}
    total_rows = len(df)
    profile['total_rows'] = int(total_rows)
    profile['total_columns'] = int(df.shape[1])
    col_types = detect_column_types(df)
    profile['columns'] = {}
    for col in df.columns:
        ser = df[col]
        non_null = ser.dropna()
        col_prof = {
            'type': col_types.get(col, 'unknown'),
            'count': int(len(ser)),
            'missing': int(ser.isnull().sum()),
            'missing_pct': float(ser.isnull().mean()),
            'unique': int(ser.nunique(dropna=True)),
        }
        t = col_prof['type']
        if t == 'numeric':
            try:
                s = pd.to_numeric(non_null, errors='coerce').dropna()
                if not s.empty:
                    col_prof.update({
                        'mean': float(s.mean()),
                        'median': float(s.median()),
                        'std': float(s.std(ddof=0)),
                        'min': float(s.min()),
                        'max': float(s.max())
                    })
            except Exception:
                pass
        elif t == 'datetime':
            try:
                s = pd.to_datetime(non_null, errors='coerce').dropna()
                if not s.empty:
                    col_prof.update({
                        'min': str(s.min()),
                        'max': str(s.max())
                    })
            except Exception:
                pass
        elif t in ('categorical','text','boolean'):
            top = non_null.astype(str).value_counts().head(5).to_dict()
            col_prof['top_values'] = {k: int(v) for k,v in top.items()}
        profile['columns'][col] = col_prof
    # quick notes
    notes = []
    if total_rows > 200000:
        notes.append('Large dataset — profiling was sampled and some operations limited for performance')
    profile['notes'] = notes
    return profile


def recommend_visualizations(df: pd.DataFrame, col_types: Dict[str,str]) -> List[Dict[str,Any]]:
    """Return a list of recommended visualizations based on detected types and cardinalities"""
    recs = []
    # detect time series candidates
    time_cols = [c for c,t in col_types.items() if t == 'datetime']
    numeric_cols = [c for c,t in col_types.items() if t == 'numeric']
    cat_cols = [c for c,t in col_types.items() if t == 'categorical']
    geo_cols = [c for c,t in col_types.items() if t == 'geo']

    if time_cols and numeric_cols:
        recs.append({'type':'line', 'x': time_cols[0], 'y': numeric_cols[:3], 'reason':'Datetime + numeric columns -> time series'})
    if numeric_cols:
        for n in numeric_cols[:6]:
            recs.append({'type':'histogram', 'column': n, 'reason':'Numeric distribution'})
    if len(numeric_cols) >= 2:
        recs.append({'type':'scatter', 'x': numeric_cols[0], 'y': numeric_cols[1], 'reason':'Two numeric columns -> scatter'})
    if cat_cols:
        for c in cat_cols[:6]:
            recs.append({'type':'bar', 'column': c, 'reason':'Categorical counts'})
    if geo_cols:
        recs.append({'type':'map', 'lat': [c for c in geo_cols if 'lat' in c.lower()][0] if any('lat' in c.lower() for c in geo_cols) else geo_cols[0],
                    'lon': [c for c in geo_cols if 'lon' in c.lower()][0] if any('lon' in c.lower() for c in geo_cols) else None,
                    'reason':'Geo columns -> scatter map'})
    # correlation heatmap if many numerics
    if len(numeric_cols) >= 2:
        recs.append({'type':'heatmap', 'columns': numeric_cols[:12], 'reason':'Correlation/heatmap for numeric columns'})
    return recs

# ---------- Routes ----------

@app.get('/', response_class=HTMLResponse)
async def index():
    # Simple embedded frontend — similar to original app but uses recommendations returned by backend
    html = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Generic EDA</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{font-family:Inter,system-ui; background:#071129;color:#e6eef8;padding:18px} .card{background:rgba(255,255,255,0.03);padding:12px;border-radius:8px;margin-bottom:12px}</style>
</head>
<body>
  <h2>Generic EDA — Upload a dataset</h2>
  <div class="card">
    <input id="file" type="file" />
    <button id="btn">Upload & Analyze</button>
    <label>Sample rows for preview: <input id="sample" type="number" value="500" style="width:90px" /></label>
    <div id="status"></div>
  </div>
  <div id="out"></div>
<script>
const fileEl = document.getElementById('file');
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const out = document.getElementById('out');

btn.onclick = async ()=>{
  if(!fileEl.files.length){ alert('pick a file'); return; }
  const f = fileEl.files[0];
  const form = new FormData();
  form.append('file', f);
  form.append('sample_rows', document.getElementById('sample').value || '500');
  status.textContent = 'Uploading...';
  const resp = await fetch('/upload', {method:'POST', body: form});
  if(!resp.ok){ status.textContent = 'Server error: ' + await resp.text(); return; }
  const json = await resp.json();
  status.textContent = 'Done.';
  renderResults(json);
}

function renderResults(j){
  out.innerHTML = '';
  const meta = document.createElement('div'); meta.className='card';
  meta.innerHTML = `<b>${j.filename}</b> — ${j.total_rows} rows × ${j.headers.length} columns\n<pre>${JSON.stringify(j.profile,null,2)}</pre>`;
  out.appendChild(meta);
  const recs = document.createElement('div'); recs.className='card'; recs.innerHTML = '<h4>Recommendations</h4>' + '<pre>' + JSON.stringify(j.recommendations,null,2) + '</pre>';
  out.appendChild(recs);
}
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)

@app.post('/upload')
async def upload(file: UploadFile = File(...), sample_rows: int = Form(500)):
    if not file.filename:
        return JSONResponse({'error':'no file provided'}, status_code=400)
    contents = await file.read()
    try:
        sheets = parse_uploaded_file(contents, file.filename, sample_rows=sample_rows)
    except Exception as e:
        return JSONResponse({'error': f'Failed to parse file: {str(e)}'}, status_code=400)

    # For multi-sheet files, pick first sheet as primary
    primary_name = list(sheets.keys())[0]
    df = sheets[primary_name].copy()
    # coerce columns to strings
    df.columns = [str(c) for c in df.columns]

    # lightweight coercions: try numeric conversion where many values look numeric
    numeric_cols = []
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notnull().sum() >= max(1, int(0.5 * len(coerced))):
            df[col] = coerced
            numeric_cols.append(col)
    # try parse datetimes
    for col in df.columns:
        if col in numeric_cols: continue
        coerced = pd.to_datetime(df[col], errors='coerce')
        if coerced.notnull().sum() >= max(1, int(0.5 * len(coerced))):
            df[col] = coerced

    profile = profile_dataframe(df)
    col_types = detect_column_types(df)
    recs = recommend_visualizations(df, col_types)

    # Prepare preview data (convert to json-friendly)
    preview = df.head(sample_rows).where(pd.notnull(df), None).to_dict(orient='records')

    response = {
        'filename': file.filename,
        'headers': list(df.columns),
        'data_preview': preview,
        'total_rows': int(len(df)),
        'profile': profile,
        'column_types': col_types,
        'recommendations': recs
    }
    return JSONResponse(response)

if __name__ == '__main__':
    print('Starting enhanced EDA server on http://localhost:8080')
    uvicorn.run('app_enhanced:app', host='0.0.0.0', port=8080, reload=False)


