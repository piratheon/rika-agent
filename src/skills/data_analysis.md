# skill: data_analysis
description: Analyse CSV, JSON, or tabular data with pandas and produce charts or summaries.
tools: run_python, write_file, send_file

## When to use
- Summarising tabular datasets
- Producing statistics, aggregations, pivot tables
- Generating matplotlib/plotly charts

## Usage pattern
```python
run_python(code="""
import pandas as pd

df = pd.read_csv('data.csv')
print(df.describe())
print(df.head())

# Export summary
df.groupby('category')['value'].sum().to_csv('summary.csv')
""")
```

## Notes
- Install: `pip install pandas matplotlib plotly openpyxl`
- For large files (>50 MB), use chunked reading: `pd.read_csv(..., chunksize=10000)`
- Save charts to workspace with `plt.savefig('chart.png')` then use send_file
