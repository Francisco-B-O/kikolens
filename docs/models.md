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

The default list is twelve models. Five more are available on request: an **SVM** (with the target scaled for regression), **AdaBoost**, **Bagging**, a **voting** ensemble (random forest, histogram gradient boosting and a linear model, averaging predictions or probabilities) and a **stacking** ensemble (the same three, combined by a linear model trained on their cross-validated predictions). If PyTorch is installed, a **PyTorch MLP** joins them: Adam, dropout, weight decay and early stopping on 15% of the training rows, with the same preprocessing as every other model.

Primary metrics: R-squared for regression, weighted F1 for classification. Regression also reports RMSE and MAE; classification adds accuracy, balanced accuracy, ROC AUC and, for two classes, the Brier score.

Importance is measured by permutation on the held-out rows. Partial dependence shows the average prediction as one feature changes. The per-row explanation replaces one feature at a time with a typical value and reports how much the prediction moves.

## Uncertainty (conformal prediction)
Split conformal prediction. Half of the held-out test rows calibrate: for regression the score of a row is the absolute error, for classification one minus the probability given to the true class. The threshold is the `ceil((n + 1) * coverage) / n` quantile of those scores, and it gives a range `prediction ± threshold` or the set of classes whose probability is at least `1 - threshold`. The guarantee, that new rows fall inside about `coverage` of the time, holds without assuming normal errors but only if new data resemble the calibration rows. The other half of the test rows measures the coverage actually obtained.

## Counterfactuals
For one row, the search tries every value of one variable (15 quantiles of a number, up to 30 categories), then pairs of the six most influential variables, and keeps the changes that give the wanted answer. They are ranked by how many variables change and then by the total standardised distance. Variables you mark as fixed never move. The result says what the model would do, not what would happen in the world.

## Fairness
For each group of a column: the share of rows that receive the favourable prediction (selection rate), the true positive rate, the false positive rate and the accuracy. The warnings use the four-fifths guideline (the lowest selection rate divided by the highest, below 0.8) and gaps of more than 10 points in the true or false positive rate. Groups under 10 rows are left out.

## Feature search
Candidates are built from up to eight numeric columns: a logarithm (`log1p`, for non-negative skewed columns), a square, and the product, the two ratios, the sum and the difference of every pair. The 24 most rank-correlated with the target are scored by cross-validated histogram gradient boosting, and added greedily while each improves the score by at least the chosen threshold. Because the evaluation is cross-validated and the formulas are applied with the safe expression interpreter, the gain is not an artefact of the training rows and the result replays on new data.

## Survival
The Kaplan-Meier estimate multiplies, at each event time, the share of those still at risk that did not have the event; the band is the log-log Greenwood interval. The log-rank test compares observed with expected events in each group at every event time, using the hypergeometric variance. It assumes that censoring is unrelated to the outcome.

## Privacy
Detection samples up to 2,000 values per column and flags it when at least 60% pass the check for a pattern. The checks are not just shapes: card numbers must pass the Luhn algorithm, IBANs the mod-97 check, and Spanish DNI/NIE numbers must carry the right letter, so ordinary long numbers are not reported. Columns named like personal data (`name`, `email`, `phone`, `birth`...) are flagged by name, with the reason shown. Masking keeps the last characters; fingerprints are `SHA-256(salt + value)` cut to 12 hex characters by default. A salted fingerprint cannot be reversed, but anyone who has the salt can test whether a known value is present, which is why the recipe that holds it should stay private. k-anonymity groups the rows by the combination of descriptive columns and reports the size of each group; rows in a group of one are unique and the easiest to re-identify.

## Matching records
Names are lowercased, stripped of accents and punctuation and their words sorted, so "García, Ana" and "ana garcia" are equal. Candidates for each row are the rows of the other table that share a rare word (a word in more than 20% of the rows is ignored); an exact match short-circuits the search. Each candidate gets the larger of the character similarity (Ratcliff-Obershelp) and a blend of it with the share of words in common. The best partner above the threshold is kept, and a match is marked ambiguous when the runner-up scores within 0.03 of it.

## Regression
Linear regression is solved by least squares; the standard errors are the classical ones, `sigma^2 (X'X)^-1`, with `t` intervals and p-values on the residual degrees of freedom. Categories become indicators with the commonest value as the reference; dates become days since the first. The page reports R-squared and adjusted R-squared, the F test of the whole model, the variance inflation factor of each term, the Durbin-Watson statistic (about 2 when neighbouring errors are unrelated), a Breusch-Pagan test (regressing the squared errors on the terms; a small p-value means their spread changes with the fit) and a Shapiro-Wilk test of the errors. Logistic regression is fitted by Newton-Raphson on standardized columns, so the Wald intervals and p-values come from the inverse information matrix; coefficients are also shown as odds ratios, with McFadden's pseudo R-squared, the likelihood-ratio test, AUC and the accuracy against always guessing the commonest. Rows with an empty value in any used column are left out and counted.

## Model health
The model scores the rows whose real result is known and the errors are grouped by period. For a number the reference is the mean absolute error measured on the held-out rows when the model was built, and a period is flagged when its error is at least 1.5 times that; for a category it is the accuracy, flagged when 10 points lower. Periods with fewer than 20 rows are reported but not judged. A trend warning appears when the rank correlation (Spearman) between time and the metric of the judged periods is significant (p below 0.05) in the direction of getting worse, with at least four periods.

## Unusual periods in a series
The expected value of each period is a centred rolling median (the trend) plus the median of what is left at each position of the cycle (the seasonal pattern, with the period guessed from the autocorrelation). The residual is scaled with the median absolute deviation, which a few spikes cannot inflate, and a period is flagged when the scaled value is beyond the sensitivity (3.5 by default). `min_gap` merges runs of neighbouring flags into one.

## Comparing saved models
Both models score the same rows. For a number, the absolute errors of the two are compared with a Wilcoxon signed-rank test; for a category, the rows where only one is right are compared with an exact binomial test (the McNemar idea). The first model by mean absolute error (or weighted F1) is the champion, and the page says whether its lead is reliable at the 5% level or could be chance.

## Why it changed
Two periods are compared. For a total or a count, each value of a column contributes the difference of its own total, so the contributions add up to the change exactly. For an average, the change is `sum over values of (w_after - w_before) * (m_before - M_before)` (mix: the value weighs more or less) plus `sum of w_after * (m_after - m_before)` (rate: its average moved), where `w` is the share of rows and `m` the average of the value; the two add up to the change exactly. A value that exists in only one period has all its effect in the mix. Columns are ranked by the share of the total absolute contribution held by their single biggest value.

## Customers
*Retention*: a cohort is the period of a user's first activity, and the cell for age `a` is the share of that cohort active in period `cohort + a`; ages beyond the last observed period are blank. *RFM*: recency (days since the last activity), frequency and total spend are ranked into five equal groups; the segments are rules over the three scores. *Funnel*: a user is counted at a stage only if they appear at it and at every earlier one. *Association rules*: Apriori finds the sets of items present in at least `support` of the baskets; for each, `confidence` is `P(right | left)` and `lift` is `confidence / P(right)`.

## Impact
*Before and after a date*: the expected series is a ridge regression of the measure on the control series fitted before the change, or, without controls, the best forecast model of the earlier trend. The uncertainty is taken from the error the same method made on earlier windows of the same length, where nothing happened, and widened with a t distribution because there are few such windows. *Difference in differences*: the coefficient of treated × after in a linear regression of the outcome, with its t interval; it assumes both groups would have moved alike without the treatment.

## Bayesian A/B
Rates use a Beta posterior (uniform prior unless set). Averages use the t distribution of the mean of each group. The probability that the variant is better is the share of 200,000 draws where it beats the control, and the expected loss is the average shortfall when the choice is wrong.

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

## Forecasting models
Five models share one interface, and each can be tested on its own:

* **Naive** repeats the last value; **seasonal naive** repeats the last full cycle.
* **Trend + seasonality** is the linear trend plus Fourier terms described above.
* **Holt-Winters** is the additive version with a damped trend: a level, a trend and one seasonal profile, updated at each step; its smoothing parameters are found by minimizing the one-step squared error. With fewer than two cycles of data it drops the seasonality.
* **Lag boosting** removes a linear trend, then fits a gradient-boosting model on the previous values (1, 2, 3, 7, 14 and one season back), a rolling mean and deviation, the weekday, month, day of year and a flag for the special dates you give. It forecasts recursively, feeding its own predictions back as lags.

**Choosing.** The series is cut at several points (up to three); at each, every model is trained on what came before and forecasts the next stretch (as long as the horizon, up to a quarter of the series). The model with the lowest mean absolute error wins, and the table also gives RMSE and MASE (error relative to repeating the previous value). The final model is refitted on all the data. **Range.** The interval is not a formula: it is the 5th to 95th percentile of the real errors the winning model made at each step ahead in those tests, and it cannot narrow as the horizon grows. With few test points it borrows from all steps. **Changes.** The series is split recursively where fitting two straight lines instead of one reduces the squared error by more than a penalty proportional to the noise (estimated from second differences) times the logarithm of the length, so noise alone produces no change points. A change is a *level shift* when the jump exceeds three times the noise and a *trend change* otherwise.

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

## Missing data, spellings and rules
* **Are gaps random?** For each column with gaps, KikoLens tests whether the other columns differ between the rows where it is empty and the rows where it is present (Mann-Whitney for numbers, chi-squared for categories). A gap is called dependent when the best p-value is below 0.001 and the effect is at least 0.2.
* **Smart fill.** KNN uses the 5 nearest rows (distance-weighted) on standardized numeric columns. Iterative uses Bayesian ridge regression, cycling over the columns. Both work on numbers only and round columns that hold whole numbers. By group takes the median (or most common value) within each group, falling back to the overall value.
* **Similar spellings.** Values are reduced to a key (lowercase, without accents or punctuation); keys are then merged when their similarity ratio is at least the threshold (0.85 by default), and every variant becomes the most frequent form. A change of two letters in a short word is below 0.85: lower the threshold to catch it.
* **Near-duplicates.** Each row becomes text; rows are compared with character 2 to 4-grams (TF-IDF, cosine similarity), looking at each row's nearest neighbours. Identical rows are excluded (they are ordinary duplicates).
* **Rules** are evaluated per row. Empty values break only the *not empty* rule. A pattern must match the whole value.

## Anomalies
Isolation Forest, Local Outlier Factor and Mahalanobis distance first scale the numeric columns with the median and interquartile range; empty values are filled with the median. The share you choose is the share of rows flagged (the most unusual ones).

* **Isolation Forest** isolates rows with random splits; unusual rows need fewer splits.
* **Local Outlier Factor** compares the density around a row with that of its neighbours.
* **Robust z-score** takes the largest distance from the median, in units of the median absolute deviation, over the columns.
* **Mahalanobis distance** measures distance from the centre while accounting for correlations between columns.

For every flagged row, the explanation lists up to three columns whose robust z-score is at least 2.

## Text
Words are lowercase sequences of letters (at least two), with stop words removed (English and Spanish; the language is detected by counting stop words). *Terms* counts in how many texts each term appears, next to its mean TF-IDF weight. *By group* ranks terms by the smoothed log odds of being in the group versus elsewhere. *Themes* uses TF-IDF vectors (reduced with a truncated SVD only when there are more than 400 terms), normalizes them and runs k-means; each theme shows its centroid's top terms and the text closest to its centre. The *classifier* is TF-IDF on words and pairs of words with a class-balanced logistic regression, scored with cross-validated predictions against the baseline of always predicting the commonest class. *Tone* counts words from a small positive and negative list. The *summary* sends a random sample of at most 40 texts, each cut to 400 characters, to the model running in Ollama.

## Dimensions
**PCA** runs on standardized columns; components for 90% is the smallest number whose cumulative explained variation reaches 90%. **t-SNE** keeps neighbours close but distorts global distances, so use it to spot groups, not to measure separation. The **VIF** of a column is `1 / (1 - R²)` of regressing it on the others, computed from the inverse of the correlation matrix.

## Comparing datasets
Numeric columns are cut at the deciles of the reference; categorical ones use their categories. The **PSI** is the sum over bins of `(current share - reference share) * ln(current share / reference share)`. Below 0.1 is stable, 0.1 to 0.25 moderate and above 0.25 a clear shift. Numbers with 12 or fewer distinct values are compared as categories. Text that holds numbers or dates is converted first, and dates and columns that are different in almost every row (identifiers) are listed as information rather than judged. A Kolmogorov-Smirnov test (numbers) or chi-squared test (categories) is shown beside the PSI, and a change of 10 points or more in the share of empty values downgrades a stable column to moderate. With large samples the tests flag tiny changes, which is why the verdict uses the PSI.

## Explaining a model
* **Shapley values.** For one row, KikoLens draws many random orders of the features. In each order it starts from a random training row and swaps in the features of the row being explained one at a time, recording how much the prediction moves at each step. The average move per feature is its Shapley value; together with the starting value they add up to the prediction exactly. For classifiers the value is in probability of the class you choose. The global version averages the absolute values over a sample of test rows. The cost grows with the number of features, so it is limited to 40.
* **Error analysis.** The error is the absolute error (regression) or whether the prediction was wrong (classification). Each category, and each quartile of a number, is compared with the overall error; a segment is reported only if it has at least 8 rows, its error is more than 15% above average and the gap is at least two standard errors, so noise does not produce stories.
* **Threshold and costs.** For a two-class model the probability threshold is swept from 0.05 to 0.95. The cost is `false alarms × cost of a false alarm + misses × cost of a miss`; the cheapest threshold is usually not 0.5 when one mistake costs more than the other.
* **Comparing two models.** Both are trained on the same rows and tested on the same held-out rows. For regression the absolute errors are compared with a Wilcoxon signed-rank test; for classification, the rows only one of them gets right are compared with a binomial (McNemar-style) test.

## Clustering
k-means on the standardized numeric columns, shown on the first two principal components. The inertia and silhouette chart helps pick `k`; the silhouette peak is a hint, not a rule.

## Recipes
A recipe is a list of `{operation, parameters}` steps. The cleaning operations (fill missing, convert type, clip outliers, drop duplicates...) and the transformations share one registry, so any of them can be replayed. Applying a recipe to a new file runs the steps in order and stops at the first one that cannot apply, naming it. Operations that depend on the data itself (such as removing flagged anomalies) are not recorded. Joins refer to the extra dataset by name; supply it when replaying (`-d name=path` in the CLI).

## Monitoring
A monitor stores its baseline as Parquet (after the recipe, if any), its settings and one JSON file per check. A check applies the recipe, reads dates and numbers stored as text, then compares against the baseline with the same machinery as *Comparing datasets*, scores with the model if one is set, and evaluates the rules. Alerts are *high* (drift at the high level, columns missing, a type change, a broken rule, an unreadable source), *medium* (a quality drop, which becomes high when it is twice the threshold; moderate drift when you asked for it; a change in rows; model inputs out of range, except missing columns, which are high; a prediction shift) or *low* (new columns). The status of a check is `alert` if any alert is high, `warning` if there are only lower ones, `failed` if the data could not be read and `ok` otherwise.

## Saved models
A model package holds the trained model together with:

* the **input schema**: the original columns the model needs, their types, training ranges or categories, and share of empty values
* the **classes** and the **test metrics**
* for lab regressions, the 5th and 95th percentile of the test errors, which become the likely range of each prediction
* optionally the **recipe**, so a raw file can be cleaned the same way before predicting

When scoring, the data are coerced to the training types (text like `1,200` becomes a number), missing columns are filled with empty values and reported, and the model's own imputation handles gaps. Predictions are only as trustworthy as the resemblance between the new data and the training data, which is what the warnings and the drift comparison measure.

Packages and `.pkl` files are Python pickles. Load only files you created yourself, because unpickling can run code.
