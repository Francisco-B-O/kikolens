# How the models work

## Preparing the data
All models share one preprocessing step:

1. Rows with a missing target are dropped.
2. Date columns become three numbers: days since 1970, month and weekday.
3. Text columns with fewer than 50 distinct values are one-hot encoded; the rest are dropped.
4. Missing numbers are filled with the median, then everything is standardized.

The target is a **regression** target when it is numeric with more than 20 distinct values, and a **classification** target otherwise. A date target, or a classification target with more than 50 classes, is rejected with an explanation.

Twenty percent of the rows are held out; every reported metric is measured on them.

## AutoML
In automatic mode the choice depends on the data: ridge regression for small regression problems, lasso when there are many columns for the rows available, random forest otherwise; logistic regression for small classification problems and gradient boosting for larger ones. The page shows the rationale. You can override it with random forest, gradient boosting, decision tree, ridge/logistic or lasso/KNN.

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

## Clustering
k-means on the standardized numeric columns, shown on the first two principal components. The inertia and silhouette chart helps pick `k`; the silhouette peak is a hint, not a rule.

## Saved models
`.pkl` files are Python pickles. Load only files you created yourself, because unpickling can run code.
