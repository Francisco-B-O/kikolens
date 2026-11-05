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

## Distribution fit
Each candidate is fitted by maximum likelihood with SciPy (the location is fixed at zero for distributions defined on positive values) and scored with AIC, `2k - 2 log L`, where `k` counts the fitted parameters; the lowest is best and a difference under 2 is a tie. A distribution is tried only if the data can belong to it: positive values for lognormal, gamma and Weibull, non-negative for exponential, between 0 and 1 for beta. The Kolmogorov-Smirnov test uses parameters estimated from the same data, which makes its p-value too generous, so it is shown as a guide. Normality uses Shapiro-Wilk on up to 5,000 values.

## Bayesian optimisation
A Gaussian process (Matern 5/2 kernel plus noise) is fitted to the cross-validated scores seen so far, over the search space mapped to the unit cube (log scale for parameters such as a learning rate, integer and categorical parameters handled). The next combination is the one with the highest expected improvement among 2,000 random candidates. A quarter of the attempts (at least 3) are random exploration first. A combination that fails is recorded as such and the search goes on.

## Sequential tests
The mSPRT mixes the likelihood ratio over a normal prior for the effect, `N(0, tau^2)`: `Lambda_n = sqrt(s2 / (s2 + n tau^2)) * exp(n^2 tau^2 D^2 / (2 s2 (s2 + n tau^2)))`, where `D` is the running mean of the pairwise differences and `s2` their variance. It is a martingale under no effect, so by Ville's inequality `P(Lambda ever >= 1/alpha) <= alpha` however often it is checked; the test stops when it crosses `1/alpha`. The always-valid p-value is `min(1, 1 / max Lambda)`, and the interval is the set of effects the same test would not reject. `tau` defaults to 20% of the standard deviation of the differences; validity does not depend on it, only how fast it detects. Observations are paired in arrival order, the k-th of each group. An effect estimated at the moment of stopping tends to be exaggerated, as with any early stop. The bandit is Thompson sampling with Beta posteriors, allocating visitors in batches.

## Forecast with drivers and Granger tests
The Granger F-test compares a regression of the effect on its own past with one that adds the past of the cause, for every lag from 1 to the maximum, with a Bonferroni correction because several lags are tried; by default it works on first differences so that a trend shared by both columns is not mistaken for a relation. The forecast uses gradient boosting on lags of the target and of the drivers (plus day of week and month); it is tested with four rolling-origin folds of one-step-ahead predictions, with and without the drivers, and a Wilcoxon test on the paired absolute errors decides whether they help. Future values of the drivers are assumed equal to their last observed value, or to the scenario factors you give.

## Feature selection
Mutual information uses the original columns (categories as codes). The L1 model is a Lasso (or an L1 logistic regression) on the standardised preprocessed matrix, with each column's coefficients added across its indicator columns. Permutation importance comes from a histogram gradient boosting on a held-out 30%; the random forest uses its impurity importances. Ranks are averaged. The score of the best k columns is cross-validated with the same booster, and the recommended k is the smallest whose mean score is within one standard error of the best.

## Segmentation
The Gaussian mixture is fitted for 1 to `k_max` components (full covariances if there are at least 20 rows per column, otherwise diagonal) and the lowest BIC wins; a row's confidence is its highest membership probability. The hierarchical method uses SciPy linkage and cuts the tree into the requested number of groups (above 2,500 rows it groups a sample and assigns the rest to the nearest group centre). DBSCAN's radius is where the sorted distance to the 5th neighbour bends most (largest second difference of the smoothed curve). Each group is described by the two columns whose group mean is furthest from the overall mean in standard deviations (shown only if beyond 0.4).

## Recommender
Item-item cosine similarity on the person-by-item matrix, multiplied by `common / (common + shrinkage)` so that pairs sharing few people count less. A person's score for an item is the sum of its similarity to the items they have used; seen items are excluded and unknown people get the most popular items. Evaluation hides one interaction for each person with at least three, refits without it and counts how often the hidden item is in the top k, against the top k most popular items the person has not used.

## Synthetic data
A Gaussian copula. Each column becomes normal scores through its ranks (a category occupies a probability interval and a point inside it is drawn); the correlation matrix of the scores is repaired to be positive semi-definite and a multivariate normal is sampled; each coordinate goes back through the column's empirical quantiles (numbers and dates) or its category frequencies. Identifiers are regenerated, long free text is skipped, and each column's share of empty values is reproduced independently. The report gives, per column, the Kolmogorov-Smirnov distance (numbers) or total variation distance (categories); the mean gap between Spearman correlations; the share of synthetic rows that exactly equal a real row; the share closer to a real row than a fifth of the usual distance between two real rows; and the train-on-synthetic, test-on-real utility. Relations that are not captured by pairwise correlation (for example an interaction of three columns) are not reproduced.

## The analyst
The language model never writes or runs code. Three pieces keep that true.

* **A client** (`kikolens/ai/client.py`) that talks to Ollama and asks for JSON. The answer is extracted from whatever text surrounds it and passed through a validator; if the validator raises, its message is appended to the prompt and the model tries again (twice by default). After that the error is shown to the user.
* **A chart specification** (`kikolens/ai/chartspec.py`): kind (bar, line, area, scatter, histogram, box, pie, heatmap), columns, aggregation, time grain, filters and top N. Every column is checked against the data (case-insensitively), every filter value is coerced to the column's type, and an impossible chart (a sum of a text column, filters that leave no rows) is refused with the valid options listed. The chart is drawn by KikoLens' own functions.
* **A tool registry** (`kikolens/ai/analyst.py`) with thirteen tools, each with typed arguments (a column, a list of columns, a number, one of a few words). A call is accepted only if the tool exists, the arguments are the ones it takes and every column is real. A tool that fails is recorded and the loop continues; repeating a call ends the loop; at most five steps run. The final answer is written from the short summaries of the steps, not from raw rows, and the steps can be exported as the equivalent calls of the Python API.

Prompts contain the column names, types, ranges, and, unless switched off, up to eight values of each categorical column and nothing else; no row of data is sent (text classification is the exception, since reading the texts is the job). Everything stays on this computer because the model does.

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

## Causal inference
The effect page estimates the average effect with regression adjustment, inverse-propensity weighting, nearest-neighbour matching on the propensity (average effect on the treated) and an augmented inverse-propensity (doubly robust) estimator, which stays consistent if either the propensity model or the outcome models are right. Propensities and outcome models are cross-fitted (five folds) so overfitting does not leak into the estimate, and propensities are trimmed. Ranges come from a bootstrap, except for the doubly robust estimator, which uses its influence function. Covariate balance is the standardised mean difference before and after weighting, and the E-value follows VanderWeele and Ding. Uplift uses a T-learner with cross-fitting, evaluated with the Qini curve. Discontinuity fits a local polynomial on each side with a triangular kernel and a rule-of-thumb bandwidth, and tests the density around the cutoff with a binomial test. Synthetic control finds non-negative weights summing to one that match the pre-intervention path, and its p-value is the share of placebo units whose post/pre fit ratio is at least as large. Instrumental variables use two-stage least squares with heteroskedasticity-robust errors.

## Simulation and optimisation
Inputs are drawn through a Gaussian copula: correlated normals are turned into uniforms and then into each distribution by its quantile function, so any mix of distributions can be correlated, and a requested correlation matrix that is not possible is replaced by the nearest valid one. Resampled columns keep their Spearman correlation. The result is evaluated with the safe expression interpreter, never with `eval`. The tornado shows the rank correlation of each input with the result and its approximate share of the variance (squared correlation over their sum). Value at risk is the 5th percentile (or the 95th when lower is better) and expected shortfall the mean beyond it. Linear programs use HiGHS through SciPy (`linprog`, or `milp` when some variables are integer); the worth of one more unit is the dual value of each limit, available only for continuous programs. Non-linear formulas use SLSQP from several random starting points and keep the best feasible result.

A problem file for `kikolens lp` looks like `{"objective": {"x": 3, "y": 5}, "constraints": [{"coefs": {"x": 1}, "op": "<=", "rhs": 4}], "bounds": {"y": [0, 10]}, "maximize": true, "integer": ["x"]}`.

## Marketing mix and price elasticity
Each channel is transformed by a geometric adstock (`a_t = x_t + decay * a_{t-1}`) and a Hill curve (`a^k / (a^k + h^k)`), and the outcome is explained by non-negative coefficients on those transforms plus a free baseline, a linear trend, Fourier terms for the seasons that fit in the series and standardised controls (non-negative least squares, with free columns entered as plus and minus). Decay (0 to 0.8), half-saturation (a multiple of the average spend) and shape (1 or 2) are searched channel by channel, twice, minimising the error on the last 20 % of the periods. A t statistic from heteroskedasticity-robust errors flags channels indistinguishable from zero. The budget split maximises the sum of steady-state responses (spend divided by one minus the decay, then the Hill curve) with SLSQP from several starting points, keeping the total fixed and each channel within a range of today's spend. This is an observational model: channels that were switched on together, or in response to sales, cannot be separated, so use its answer to choose what to test.

Elasticity is the coefficient of log price in a regression of log quantity, with HC1 errors. With an instrument it becomes two-stage least squares with the first-stage F statistic reported. With a unit cost and constant elasticity `e < -1`, the margin-maximising price is `cost * e / (1 + e)`.

## Classical time series
ADF regresses the first difference on the lagged level, a constant (and trend) and lagged differences, with the number of lags chosen by AIC; its p-value uses MacKinnon's approximation, checked against the published 1 %, 5 % and 10 % critical values. KPSS uses the partial sums of the residuals over a Newey-West long-run variance and interpolates its p-value in the published table (so it is bounded between 0.01 and 0.10). Both are simulated in the test suite to hold their size and have power. STL follows Cleveland et al.: the seasonal subseries are smoothed by local linear regression, a low-pass filter removes what leaks into the season, the trend is smoothed, and optional bisquare robustness weights are recomputed over several outer rounds; strengths are `1 - var(remainder) / var(component + remainder)` on a remainder clipped at four robust standard deviations, so one outlier does not dilute them.

ARIMA models `phi(B) Phi(B^m) (1 - B)^d (1 - B^m)^D y = theta(B) Theta(B^m) e` with the polynomials multiplied out and the residuals computed by a linear filter. Parameters are found by conditional least squares (the first residuals, which depend on values before the sample, are left out), with stationarity and invertibility guaranteed by a partial-autocorrelation parametrisation; this is close to but not the same as exact maximum likelihood, and very short series or roots near the unit circle show the difference. Forecast ranges use the psi weights of the model and assume normal errors. The automatic search takes the differences from KPSS (and the seasonal difference from the strength of the season), tries `p, q` up to 3 with `p + q <= 4` and `P, Q` in {0, 1}, and compares AICc on one common sample, because candidates with more autoregressive terms would otherwise drop more initial residuals and look better for free.

## Model governance
Records live in `~/.kikolens/governance` (one JSON per model, plus `audit.jsonl`) and refer to the model by its file name in the registry. At registration the SHA-256 of the model file is stored; promotion and stage references (`sales@production`) refuse a file whose hash has changed, because it is no longer the model that was documented and validated. A validation scores the model on rows with known results with the same metrics as the model lab and compares one of them (MAE, RMSE, R2 or accuracy) with the training value; because both numbers are noisy, the validation fails only if even the optimistic end of a bootstrap 95 % range of the new metric is worse than the tolerated fraction (20 % by default). The comparison with the production model uses the champion and challenger test of `kikolens models compare` and blocks only a challenger that is reliably worse. Each audit entry stores the hash of the previous one together with its own content, so changing, removing or inserting a line breaks every hash after it; this reveals tampering to anyone who verifies, but cannot prevent someone with write access to the folder from rewriting the whole log.

## Text mining
Topics use TF-IDF with NMF (or counts with LDA) over words that appear in at least two texts and in at most 60 % of them. Coherence is the UMass measure: for each topic's ten leading words, the mean of `log((D(a, b) + 1) / D(b))` over pairs, where `D` counts the texts containing the word(s); it is a guide rather than a verdict, and the automatic choice often lands on a coarser split than a person would. Near duplicates compare TF-IDF vectors of 3 to 4 letter fragments by cosine similarity, in blocks so that 20,000 texts stay manageable, and join texts above the threshold transitively. RAKE splits texts at stop words and punctuation and scores each word by its degree over its frequency. Collocations are ranked by normalised pointwise mutual information. Readability counts syllables from vowel groups, so scores are approximate. Pattern extraction uses regular expressions for common formats.

## Network analysis
Links are summed into a sparse adjacency matrix (repeated links add their weights; self-loops and rows with a missing end are ignored and reported). PageRank is a power iteration with damping 0.85 that spreads dangling nodes evenly. Betweenness is Brandes' algorithm over hops, exact up to a work budget and otherwise estimated from a random sample of starting nodes and rescaled. Communities come from Louvain modularity optimisation, relabelled from largest to smallest; weak modularity (below 0.3) is flagged. Distances are measured inside the largest connected piece from up to 60 starting nodes, and the longest distance is a double-sweep lower bound. Suggested links use Adamic-Adar: shared neighbours weighted by one over the log of their degree. In the tests PageRank, betweenness, clustering, assortativity and modularity agree with networkx to 1e-8 when it is installed.

## Geography
Distances use the haversine formula on a sphere of radius 6371.0088 km (about 0.3 % off the true ellipsoid). Nearest-centre queries and neighbour searches use a ball tree on that metric. Zones come from DBSCAN with the haversine distance. Moran's I uses the k nearest neighbours of each point as binary links, row-standardised, and its p-value comes from shuffling the values (999 times by default); the expected value without a pattern is -1/(n-1). Getis-Ord Gi* sums each point and its neighbours, centres it on the global mean and divides by the standard deviation expected from the number of neighbours; the p-values of all points are corrected with Benjamini-Hochberg, so "hot spot" means significant after accounting for the number of places tested. Grid cells are `cell_km` tall and widened in longitude by the cosine of the mean latitude. A single projection of the sphere is used for all maps, so shapes far from the centre of a large area are distorted.

## Experiment checks
The sample ratio test is a chi-square goodness-of-fit between the rows per group and the planned split; it alarms at p < 0.001 (the usual convention, because it is run on every test), and flags p between 0.001 and 0.05 as slightly off. Effects use Welch's t interval for numbers and a normal interval on the difference of proportions for 0/1 outcomes. CUPED replaces the outcome Y with `Y - theta (X - mean(X))`, where `theta = cov(Y, X) / var(X)` is estimated on all rows pooled so that it does not depend on the group; the variance falls by the squared correlation with X, which the page reports next to the observed narrowing of the interval, and a balance test on X flags assignment problems. Segments are compared with the rest of the rows through the difference of the two effects (a z test on the difference of differences), and the p-values of all segments are corrected with Benjamini-Hochberg. The smallest detectable effect is `(z(0.975) + z(0.8)) * sqrt(2 sd^2 / n)` for the smaller group, at 80 % power.

## Process control
Limits for individual values use the average moving range divided by d2 = 1.128; for subgroup means, the pooled within-subgroup standard deviation divided by c4 and by the square root of each subgroup size (reproducing the published A3 = 1.427 for subgroups of five); for counts, the binomial or Poisson standard deviation of each sample. The rules follow Western Electric and Nelson. Their false-alarm rates on a stable normal process were measured by simulation (rule 1: 0.27 %, 2: 0.19 %, 3: 0.40 %, nine in a row: 0.30 %, trend: 0.27 %, alternating: 0.42 %), which is why only rules 1 and 4 are on by default: all six together would flag about 2 % of the points of a perfectly stable process. Capability treats Cp and Cpk as `(USL - LSL) / 6 sigma` and `min(USL - mean, mean - LSL) / 3 sigma` with sigma from moving ranges, Pp and Ppk with the overall standard deviation, and the share out of specification from the normal distribution next to the observed share. CUSUM accumulates `max(0, S + z - k)` in each direction with `k = 0.5` and `h = 5` standard deviations (average run length about 10 for a one-standard-deviation shift, about 465 when nothing changes); EWMA uses time-varying limits `L sigma sqrt(lambda / (2 - lambda) (1 - (1 - lambda)^(2 i)))`.

## Inventory
The safety stock is `z * sqrt(L * sd_d^2 + d^2 * sd_L^2)` for a cycle service level (the chance of no stockout between an order and its delivery), and the reorder point is `d * L + safety stock`; the fill rate (fraction of demand served from stock) is `1 - sd_L * G(z) / Q` with `G` the standard normal loss function. The newsvendor quantity is the demand quantile at `(price - cost) / (price - salvage)`; its expected profit, leftover and shortage use the normal loss function and agree with a two-million-case simulation in the tests. The economic order quantity is `sqrt(2 D S / H)`. ABC cumulates value after sorting items; XYZ uses the coefficient of variation of demand per period; the demand class follows Syntetos and Boylan: average gap between demands above 1.32 periods is intermittent, and squared coefficient of variation of the demand sizes above 0.49 is erratic (lumpy when both). The policy simulation runs many copies of a (reorder point, order quantity) rule at once with backorders or lost sales, normal daily demand (never negative, drawn per day, because drawing it per small step and cutting it at zero inflates the mean), and normal delivery times rounded to days. It checks stock in eight steps a day: checking only once a day delays each order by half a day of demand on average and makes the real service clearly worse than the formula, so a coarse simulation is not a fair test of the formula. Formulas assume independent, roughly normal demand; for intermittent items the simulation is the better guide.

## Attribution
Each person's contacts are ordered by time and cut at their first conversion; contacts after it do not count. First, last, linear, time-decay (`0.5^(days before the conversion / half-life)`, normalised) and position-based rules distribute the value of each conversion. The Markov model estimates the transition probabilities between start, channels, conversion and a no-conversion end from all paths (including those that did not convert); for each channel, every transition into it is redirected to the no-conversion end and the drop in the absorption probability of conversion is its removal effect; credit is proportional to removal effects. Shapley values use the worth of a set of channels as the conversions of the paths whose channels all belong to the set, and average each channel's marginal contribution over all orders; it is exact up to ten channels (smaller ones are grouped as "other"). On a worked example in the tests, the Markov shares (1/3 and 2/3) and Shapley shares (1/2 and 1/2) equal the values computed by hand.

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
