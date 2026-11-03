# How the methods work

## Preparing the data
The classical models, the neural network and the lab share the same ideas:

1. Rows with a missing target are dropped.
2. Date columns become three numbers: days since 1970, month and weekday.
3. Text columns are one-hot encoded; in AutoML and the neural network those with 50 or more distinct values are dropped, in the lab up to 30 categories are kept per column (rarer ones are grouped), columns with a different value in every row are dropped as identifiers, and so are text columns with more distinct values than half the rows.
4. Missing numbers are filled with the median, then everything is standardized.

The target is a **regression** target when it is numeric with more than 20 distinct values, and a **classification** target otherwise. A date target, or a classification target with more than 50 classes, is rejected with an explanation.

Twenty percent of the rows are held out; every reported metric is measured on them.

## AutoML
In automatic mode the choice depends on the data: ridge regression for small regression problems, lasso when there are many columns for the rows available, random forest otherwise; logistic regression for small classification problems and gradient boosting for larger ones. The page shows the rationale. You can override it with random forest, gradient boosting, decision tree, ridge/logistic or lasso/KNN.

## Model lab
The lab exists to answer "which model should I trust?" without fooling yourself.

1. **Split first.** A share of the rows (20% by default) is set aside and not touched until the very end.
2. **Preprocessing lives inside each model.** Imputers, scalers and encoders are fitted only on the training part of each cross-validation fold, so nothing from the validation rows leaks into training.
3. **Compare.** Every selected model is scored with k-fold cross-validation (stratified for classification). A baseline (predict the mean, or the most common class) is always in the table: a model that does not beat it has learned nothing.
4. **Tune the best.** A randomized search over sensible ranges. The tuned version replaces the default only if it scores better in cross-validation.
5. **Test once.** The chosen model is scored on the held-out rows. Those are the numbers to report.
6. **Check for problems.**
   * *Leakage:* a numeric column whose rank correlation with the target is 0.98 or more is flagged, and so is any column that, on its own and with a shallow decision tree, predicts the target almost perfectly (cross-validated score of 0.98 or more, and clearly better than the baseline). Such columns are often a copy of the answer or something only known afterwards.
   * *Imbalance:* when the rarest class is under 10% of the rows accuracy becomes misleading, so the lab recommends balanced accuracy or F1 and offers class weighting.
   * *Learning curve:* shows whether more data would help.

Primary metrics: R-squared for regression, weighted F1 for classification. Regression also reports RMSE and MAE; classification adds accuracy, balanced accuracy, ROC AUC and, for two classes, the Brier score.

Importance is measured by permutation on the held-out rows. Partial dependence shows the average prediction as one feature changes. The per-row explanation replaces one feature at a time with a typical value and reports how much the prediction moves.

## Neural network
A multilayer perceptron from scikit-learn (`MLPRegressor` or `MLPClassifier`).

* **Regression targets are standardized inside the model** and predictions come back in the original units. Without this a network cannot learn a target such as revenue in the thousands.
* **Early stopping** holds out 10% of the training rows and stops when the validation score stops improving. It is unavailable with the `lbfgs` optimizer.
* **Importance** is measured by permutation: shuffle one column on the test rows and see how much the score drops. This works for any model.
* The architecture chart draws at most 12 nodes per layer; the legend shows the real size.

Rules of thumb: start with the *Medium* preset; if training and validation curves diverge, raise the L2 regularization or shrink the network; if both are poor, give it more epochs or a larger network. Networks need at least 20 rows and usually beat classical models only on larger, non-linear data.

## Time series
1. The date column is parsed, sorted and aggregated to a regular period (mean by default). Gaps are interpolated.
2. **Decomposition** splits the series into trend (centered moving average), a repeating seasonal pattern and a remainder. The seasonal period comes from the frequency (7 for daily, 12 for monthly, and so on) or, failing that, from the autocorrelation.
3. **Forecast** fits a linear trend plus Fourier terms for the seasonality. The range shown is 95% and widens with distance.
4. **Backtest.** The last stretch of the data is hidden, forecast, and compared with the truth and with a naive baseline that repeats the last value. If the model does not beat the baseline the dashboard warns you.

This is a deliberately simple model. It handles trend and one seasonal cycle well; it does not know about holidays, promotions or sudden changes.

## Statistical tests
Every test returns the p-value, an effect size and a sentence. A small p-value says a difference is unlikely to be chance; the effect size says whether it is big enough to matter, and with lots of rows even trivial differences become "significant".

| Question | Test chosen | Effect size |
| :--- | :--- | :--- |
| Do 2 groups differ in a number? | Welch's t-test if the groups look normal (or have 30+ rows), otherwise Mann-Whitney U | Cohen's d, or rank-biserial r |
| Do 3+ groups differ? | One-way ANOVA with Tukey comparisons if normal with similar spread, otherwise Kruskal-Wallis with Holm-adjusted pairwise Mann-Whitney | eta squared, or epsilon squared |
| Are two categories related? | Chi-squared; Fisher's exact for 2x2 tables with small expected counts | Cramér's V |
| Are two numbers correlated? | Pearson, with a Fisher-z confidence interval, next to Spearman | r |
| Did the treatment change a rate? | Two-proportion z-test, 95% interval for the difference, and the probability that the treatment is better from Beta posteriors | absolute and relative lift |
| Did it change an average? | Welch's t-test | difference and lift |

Normality is checked with Shapiro-Wilk on small groups; equal spread with Levene. The thresholds for the effect labels follow common conventions (for example Cohen's d: 0.2 small, 0.5 medium, 0.8 large).

**Sample size.** For rates, the number of rows per group is `(z_a * sqrt(2 p̄ (1 - p̄)) + z_b * sqrt(p1 (1 - p1) + p2 (1 - p2)))^2 / (p2 - p1)^2` at 5% significance and the power you choose; for averages it is `2 ((z_a + z_b) * sd / delta)^2`.

**Screening many columns.** Testing 50 columns at the 5% level produces about two false hits by chance. The screen therefore reports Benjamini-Hochberg adjusted p-values and marks a column as related only if the adjusted value is below the threshold. Each column is compared with the target using the test that fits the two types (Spearman, Kruskal-Wallis or chi-squared), and numbers with 10 or fewer distinct values are treated as categories.

## Anomalies
Isolation Forest, Local Outlier Factor and Mahalanobis distance first scale the numeric columns with the median and interquartile range; empty values are filled with the median. The share you choose is the share of rows flagged (the most unusual ones).

* **Isolation Forest** isolates rows with random splits; unusual rows need fewer splits.
* **Local Outlier Factor** compares the density around a row with that of its neighbours.
* **Robust z-score** takes the largest distance from the median, in units of the median absolute deviation, over the columns.
* **Mahalanobis distance** measures distance from the centre while accounting for correlations between columns.

For every flagged row, the explanation lists up to three columns whose robust z-score is at least 2.

## Dimensions
**PCA** runs on standardized columns; components for 90% is the smallest number whose cumulative explained variation reaches 90%. **t-SNE** keeps neighbours close but distorts global distances, so use it to spot groups, not to measure separation. The **VIF** of a column is `1 / (1 - R²)` of regressing it on the others, computed from the inverse of the correlation matrix.

## Comparing datasets
Numeric columns are cut at the deciles of the reference; categorical ones use their categories. The **PSI** is the sum over bins of `(current share - reference share) * ln(current share / reference share)`. Below 0.1 is stable, 0.1 to 0.25 moderate and above 0.25 a clear shift. Numbers with 12 or fewer distinct values are compared as categories. Text that holds numbers or dates is converted first, and dates and columns that are different in almost every row (identifiers) are listed as information rather than judged. A Kolmogorov-Smirnov test (numbers) or chi-squared test (categories) is shown beside the PSI, and a change of 10 points or more in the share of empty values downgrades a stable column to moderate. With large samples the tests flag tiny changes, which is why the verdict uses the PSI.

## Clustering
k-means on the standardized numeric columns, shown on the first two principal components. The inertia and silhouette chart helps pick `k`; the silhouette peak is a hint, not a rule.

## Recipes
A recipe is a list of `{operation, parameters}` steps. The cleaning operations (fill missing, convert type, clip outliers, drop duplicates...) and the transformations share one registry, so any of them can be replayed. Applying a recipe to a new file runs the steps in order and stops at the first one that cannot apply, naming it. Operations that depend on the data itself (such as removing flagged anomalies) are not recorded. Joins refer to the extra dataset by name; supply it when replaying (`-d name=path` in the CLI).

## Saved models
A model package holds the trained model together with:

* the **input schema**: the original columns the model needs, their types, training ranges or categories, and share of empty values
* the **classes** and the **test metrics**
* for lab regressions, the 5th and 95th percentile of the test errors, which become the likely range of each prediction
* optionally the **recipe**, so a raw file can be cleaned the same way before predicting

When scoring, the data are coerced to the training types (text like `1,200` becomes a number), missing columns are filled with empty values and reported, and the model's own imputation handles gaps. Predictions are only as trustworthy as the resemblance between the new data and the training data, which is what the warnings and the drift comparison measure.

Packages and `.pkl` files are Python pickles. Load only files you created yourself, because unpickling can run code.
