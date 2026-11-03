# Dashboard guide

Start it with `kikolens start` and open <http://localhost:8501>. Pick a file, a link or a database in the sidebar; every section works on the same dataset, and cleaning steps carry over to the rest.

The interface follows the light or dark theme from Streamlit's settings menu (top right).

## Sections

### Overview
Rows, columns, quality score, missing cells and memory use, a table with the type and health of every column, and a preview of the data.

![Overview](images/overview-light.png)

### Explore
Pick a chart type and the columns to plot. Filters (numeric range, date range, category) narrow the rows first. Scatter and density charts sample 20,000 points so they stay responsive.

![Explore](images/explore-light.png)

### Quality and cleaning
Lists everything wrong with the data, ordered by how much it matters:

* missing values and columns that are mostly empty
* duplicated rows
* constant columns
* text columns that are really numbers, dates or booleans
* outliers (IQR rule)

**Auto-clean** applies only the safe fixes: trim whitespace, drop duplicates, convert mistyped columns, drop constant columns. Everything else is an explicit operation you choose. **Undo** reverts the last step and **Reset** returns to the file as loaded. The original file is never modified.

![Quality and cleaning](images/quality-light.png)

### Patterns
Correlations above 0.85, columns with extreme values, categories that dominate, and probable identifier columns. A button asks the local language model for a written summary (needs Ollama).

### Statistics
A summary table, the distribution of any column, and the correlation matrix with the ten strongest relationships.

### Ask
Type a question and a local model writes the SQL; you can edit it before it runs. The *SQL* tab runs your own queries. The data is the table `df`. Only read-only queries are accepted.

### Models
AutoML picks an algorithm from the size and shape of the data and explains why. You can also choose the algorithm or import a model you trained earlier. Results include metrics, feature importance, a row explanation and a what-if simulator.

### Neural network
A multilayer perceptron you configure: hidden layers (presets or your own, e.g. `128, 64, 32`), activation, optimizer, learning rate, L2 regularization, maximum epochs, early stopping and the held-out share for testing. After training you get metrics, the network diagram, the training curve, permutation importance and, if a classical model was trained for the same target, a side-by-side comparison.

![Neural network](images/neural-light.png)

### Time series
Choose a date column and a numeric column. The page resamples to a regular period, estimates trend and seasonality, and forecasts ahead with a 95% range. The forecast is tested on the last part of the data and compared with repeating the last value; the page says plainly when it does not beat that baseline.

![Time series](images/timeseries-light.png)

### Segments
k-means clustering on the numeric columns, with an optional chart of inertia and silhouette scores to help choose the number of groups. Download the data with the segment label attached.

### Report
Builds one self-contained HTML page with quality, statistics, correlations, patterns, any models trained in this session and, optionally, commentary from the local model. Open it in a browser and print to save as PDF.

## Dark theme

![Dark theme](images/neural-dark.png)
