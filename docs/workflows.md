# Workflows

Four end-to-end paths that use the sample files in `examples/`. Each can be done in the dashboard or from the terminal.

## 1. From a messy file to a model you can reuse

**Goal:** clean `sales_demo.csv`, find a good model for `revenue`, and keep it.

| Dashboard | Terminal |
| :--- | :--- |
| *Quality and cleaning*: press **Auto-clean** (trims text, drops the 7 duplicate rows, turns `date` into dates and `unit_price` from text into numbers). | `kikolens clean examples/sales_demo.csv --output clean.csv --save-recipe recipe.json` |
| *Model lab*: choose `revenue`, leave out `units`, run it. Read the leaderboard, then the diagnostics. | `kikolens lab clean.csv revenue --exclude units --register sales-model` |
| *Predict*: **Save to my models**. | (already saved by `--register`) |

Why leave out `units`? In this synthetic data it is almost a copy of `revenue`, so a model that includes it just reads the answer off that column and looks excellent (R² around 0.9 from one column). Leaving it out forces the model to learn from the date, region, channel, price and discount, which is the question that matters if you want to predict revenue before the units are known. The lab's leakage check looks for columns that are *nearly perfect* copies of the target; a column like `units` that is highly informative but not identical is for you to judge.

When a CSV is read, dates and numbers written as text (`2023-01-01`, `1,200`) are recognized and used as dates and numbers when modeling; the lab tells you which columns it read that way. The recipe and the model package keep the cleaning steps and the columns the model expects, so you do not need to remember either.

## 2. Score a new quarter

**Goal:** apply the same cleaning to `sales_new_quarter.csv`, predict its revenue, and check whether the model can still be trusted.

```bash
kikolens apply examples/sales_new_quarter.csv recipe.json --output new_clean.csv
kikolens score new_clean.csv sales-model --output predictions.csv
kikolens drift clean.csv new_clean.csv --output drift.csv
```

In the dashboard: *Predict* with **Predict a file** and the *Another file* option (tick *Apply the cleaning steps used for training first* to skip the first command), then *Compare data* with **Another file against the open one**.

What you will see:

* `score` warns that the dates are later than anything in training and that many `unit_price` values are above the training range.
* `drift` reports `channel`, `region` and `unit_price` as drifted (more online sales, a bigger West share, higher prices), and `revenue` as well because it trends upward. It exits with code `2`.
* The predictions still run, but they are less accurate: in our run the average error on the new quarter was about twice the error on the held-out test, and only a little over half of the actual values fell inside the predicted range that should hold about 90% of the time. That gap is exactly what the warnings and the drift report are telling you. Retrain with recent data before relying on them.

## 3. Understand what drives a result

**Goal:** find out which factors are really associated with revenue, without fooling yourself.

1. *Experiments* → **What relates to...** → target `revenue` (terminal: `kikolens screen clean.csv revenue`). All columns are tested together and corrected for chance. In the sample data only `units` is related (rank correlation 0.95); `unit_price`, `region` and `discount` are not, even though each would have looked "interesting" in a chart.
2. *Experiments* → **Compare groups**: revenue by `region`. The ANOVA gives p ≈ 0.27 and an effect size of 0.01 ("negligible"): the regional averages differ only as much as chance would produce with this many rows, and the page says that this does not prove there is no difference.
3. *Dimensions* → **Redundant columns**: `revenue` and `units` both get a VIF around 10.8 ("severe"), meaning each is mostly explained by the other. Do not read one as an explanation of the other.
4. *Model lab* → **Explain**: permutation importance and the partial dependence of the top feature, for the model with `units` left out.

## 4. Enrich, test and ship

**Goal:** join manager targets to the sales, compare channels with an A/B test, and share the result.

1. *Transform* → **Combine**: upload `regions.csv`, join on `region` (left). Each row now carries its `manager` and `target`.
2. *Transform* → **Columns** → **Formula**: `share_of_target` = `revenue / target`.
3. *Experiments* → **A/B test**: group `channel`, control `Store`, outcome `revenue` (terminal: `kikolens ab clean.csv channel revenue --control Store`). In the sample data Online averages 1,888 against 1,881 for Store (+0.4%, 95% interval −41 to +55, p ≈ 0.78): no reliable difference. The page also tells you that detecting a gap this small would take on the order of 24,000 rows per group, which is the honest answer to "should we invest in this channel because of this data?".
4. *Anomalies*: review the 2% most unusual days (`kikolens anomalies clean.csv --share 2`); each comes with the reason it was flagged. Remove them only if they are errors, not real events.
5. *Project* → **Save**: write your notes, tick *Include trained models* and download the `.kikolens` file. *Report* builds the HTML page for others.

The steps in 1 and 2 are recorded in the recipe; the join needs `regions.csv` supplied again when you replay it (`kikolens apply data.csv recipe.json --output out.csv -d examples/regions.csv`).

## Monitoring from a script or CI

`kikolens drift` returns exit code `2` when the data changed a lot, `0` otherwise. A nightly job can compare the latest export with the data the model was trained on:

```bash
kikolens drift training.csv latest.csv || echo "Data changed: review before trusting predictions"
```
