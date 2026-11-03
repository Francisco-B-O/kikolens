# Dashboard guide

Start it with `kikolens start` and open <http://localhost:8501>. Pick a file, a link or a database in the sidebar; every section works on the same dataset, and the cleaning and transformation steps carry over to the rest.

The sidebar has two levels: an **area** (Data, Analysis, Models, Share) and, under it, the sections of that area. The interface follows the light or dark theme from Streamlit's settings menu (top right).

## Data

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

**Auto-clean** applies only the safe fixes: trim whitespace, drop duplicates, convert mistyped columns, drop constant columns. Everything else is an explicit operation you choose. **Undo** reverts the last action (Auto-clean counts as one) and **Reset** returns to the file as loaded. The original file is never modified.

![Quality and cleaning](images/quality-light.png)

### Transform
Six tabs of operations, each recorded as a step:

| Tab | Operations |
| :--- | :--- |
| **Rows** | Filter by a condition (`==`, `>`, `in`, `contains`, `is empty`...), sort, sample a fraction or a number of rows. |
| **Columns** | Create a column from a formula, combine or split text, rename, replace values. |
| **Features** | Date parts (year, month, weekday, weekend, days since start), bins by quantile or equal width, scaling (standard, min-max, robust), log, rank, one-hot encoding. |
| **Sequences** | Lagged values and rolling statistics, ordered by a column and optionally within a group (for example per customer). |
| **Reshape** | Group and summarize, pivot, stack columns. |
| **Combine** | Upload a second file and join it on a key (left, inner, outer, right) or append its rows. |

Formulas accept column names, numbers, `+ - * / // % **`, comparisons and the functions `log`, `log1p`, `log10`, `sqrt`, `abs`, `exp`, `round`, `floor`, `ceil`, `min`, `max`. Put a name with spaces between backticks: ``log1p(`unit price`)``. Division by zero gives an empty value, not an error. Anything else is rejected, so a recipe from someone else cannot run code.

![Transform](images/transform-light.png)

### Patterns
Correlations above 0.85, columns with extreme values, categories that dominate, and probable identifier columns. A button asks the local language model for a written summary (needs Ollama).

### Statistics
A summary table, the distribution of any column, and the correlation matrix with the ten strongest relationships.

### Ask
Type a question and a local model writes the SQL; you can edit it before it runs. The *SQL* tab runs your own queries. The data is the table `df`. Only read-only queries are accepted.

## Analysis

### Experiments
Five tabs that answer "is this real or chance?" in plain language. Each result shows the test used, the p-value, the **effect size** with a label (negligible, small, medium, large) and a conclusion that also says what you cannot conclude.

* **Compare groups.** Pick a number and a column that splits the rows. Two groups: Welch's t-test, or Mann-Whitney when the data are clearly not normal. Three or more: ANOVA with Tukey comparisons, or Kruskal-Wallis with Holm-adjusted comparisons. The page tells you which was chosen and why.
* **Two categories.** Chi-squared test (Fisher's exact for small 2x2 tables) with Cramér's V and a map of where the table departs from chance.
* **Correlation.** Pearson and Spearman with a confidence interval, and a warning when outliers or curvature distort the picture.
* **A/B test.** Control against treatment on a rate (two-proportion z-test and the probability that the treatment is better) or an average (Welch). Includes a planner that tells you how many rows each group needs before you start.
* **What relates to...** Tests every column against a target and corrects for running many tests at once (Benjamini-Hochberg), so a handful of random "hits" are not reported as findings.

![Experiments](images/experiments-light.png)

### Anomalies
Choose columns, a method and the share of rows to flag. Methods: Isolation Forest (general purpose), Local Outlier Factor (odd rows inside dense groups), Robust z-score (one column at a time, easiest to explain) and Mahalanobis distance (accounts for correlated columns). Every flagged row comes with the reason, such as "income is 8.2 robust SDs above typical". You can download the rows or remove them (undoable; removal is not stored in the recipe, because the rows are specific to this data).

![Anomalies](images/anomalies-light.png)

### Dimensions
* **PCA.** How many components capture 90% of the variation, what each is made of, and a scatter of the rows on any two components, colored by a column.
* **t-SNE.** A 2D map where similar rows sit together (a sample of 3,000 rows is used on larger data).
* **Redundant columns.** The variance inflation factor of each column: above 5 it is largely explained by the others, above 10 it is nearly a duplicate.

![Dimensions](images/dimensions-light.png)

### Compare data
See what changed between two datasets: the first part of the data against the rest, the original file against your working data, or another file against the open one. Each column gets a verdict (stable, moderate, drift) from the population stability index (PSI), a KS or chi-squared test, and notes such as new categories or a changed share of empty values. Dates and columns that are different in every row (identifiers) are shown as *info*, since they are expected to change. Text that holds numbers or dates (`1,200`, `2024-03-01`) is read as such. Columns that exist in only one dataset, or changed from numbers to text, are called out. Use it to check whether new data still looks like what a model was trained on.

![Compare data](images/compare-light.png)

### Segments
k-means clustering on the numeric columns, with an optional chart of inertia and silhouette scores to help choose the number of groups. Download the data with the segment label attached.

## Models

### Models
AutoML picks an algorithm from the size and shape of the data and explains why. You can also choose the algorithm or import a model you trained earlier. Results include metrics, feature importance, a row explanation and a what-if simulator.

### Model lab
The careful way to choose a model. Pick the target, the columns to leave out and the models to compare; the lab keeps a share of rows aside, scores every model with cross-validation on the rest, tunes the best one with a randomized search, and only then measures it once on the held-out rows. Along the way it:

* warns about **data leaks**: columns that predict the target suspiciously well on their own
* warns about **imbalanced classes** and can weight rarer classes
* draws diagnostics: predicted against actual and residuals (regression), confusion matrix, ROC, precision-recall and calibration (classification), and a learning curve
* explains the model with permutation importance, partial dependence for any feature, and the effect of each feature on one chosen row

![Model lab](images/lab-leaderboard-light.png)

![Model lab diagnostics](images/lab-diagnostics-light.png)

### Neural network
A multilayer perceptron you configure: hidden layers (presets or your own, e.g. `128, 64, 32`), activation, optimizer, learning rate, L2 regularization, maximum epochs, early stopping and the held-out share for testing. After training you get metrics, the network diagram, the training curve, permutation importance and, if a classical model was trained for the same target, a side-by-side comparison.

![Neural network](images/neural-light.png)

### Time series
Choose a date column and a numeric column. The page resamples to a regular period, estimates trend and seasonality, and forecasts ahead with a 95% range. The forecast is tested on the last part of the data and compared with repeating the last value; the page says plainly when it does not beat that baseline.

![Time series](images/timeseries-light.png)

### Predict
Use a model on data it has not seen. Choose a model trained in this session, one from your saved models, or open a model file, then:

* **Predict a file.** Use the open dataset or upload another file. Before predicting, the page lists problems with the input: missing columns, values that could not be read, categories never seen in training, numbers far outside the training range. Regression gives a likely range for each row; classification gives a confidence, and rows under 60% are counted so you know where to look first. If the file is raw, tick the box to repeat the cleaning steps used for training.
* **Predict one row.** A form built from what the model expects, with the training range of each input.
* **Save and share.** Keep the model in your saved models (saving the same name again creates a new version), download the package, or download a Markdown model card with its metrics, most influential features and expected inputs.

![Predict](images/predict-light.png)

## Share

### Report
Builds one self-contained HTML page with quality, statistics, correlations, patterns, any models trained in this session and, optionally, commentary from the local model. Open it in a browser and print to save as PDF.

### Project
* **Save.** Packs the original data, the recipe, extra datasets, your notes and, if you tick it, the trained models, into one `.kikolens` file.
* **Open.** Restores all of that and replays the recipe, so you land exactly where you stopped. Data are stored as Parquet and never run code; models inside are loaded only when you confirm you trust the file.
* **Recipe.** Every cleaning and transformation step, downloadable as JSON or as a Python script. Upload a recipe and a new file with the same columns to get it processed the same way.

## Dark theme

![Dark theme](images/neural-dark.png)
