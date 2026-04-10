import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

class AutoMLExplainer:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = "unknown"
        self.selected_model_name = ""
        self.model = None
        self.feature_importance = {}
        self.selection_rationale = []
        self.training_columns = []
        self.categorical_mappings = {}
        self.metrics = {}
        self.plots = {}

    def analyze_task(self, manual_type: str = None):
        target = self.df[self.target_col]
        n_samples, n_features = len(self.df), len(self.df.columns) - 1
        self.task_type = "regression" if pd.api.types.is_numeric_dtype(target.dtype) and target.nunique() > 20 else "classification"

        # Intelligence Mapping
        if manual_type and manual_type != "💎 Auto":
            models = {
                "Random Forest": (RandomForestRegressor() if self.task_type == "regression" else RandomForestClassifier()),
                "Gradient Boosting": (GradientBoostingRegressor() if self.task_type == "regression" else GradientBoostingClassifier()),
                "Decision Tree": (DecisionTreeRegressor() if self.task_type == "regression" else DecisionTreeClassifier()),
                "Ridge/Logistic": (Ridge() if self.task_type == "regression" else LogisticRegression(max_iter=1000)),
                "Lasso/KNN": (Lasso() if self.task_type == "regression" else KNeighborsClassifier()),
            }
            self.model = models.get(manual_type, RandomForestRegressor())
            self.selected_model_name = manual_type
            self.selection_rationale.append(f"Manual override: User selected {manual_type}.")
        else:
            # PRO DECISION ENGINE
            if self.task_type == "regression":
                if n_samples < 300:
                    self.model, self.selected_model_name = Ridge(), "Ridge Regression"
                    self.selection_rationale.append("Small dataset detected. Ridge provides L2 regularization to prevent overfitting on noise.")
                elif n_features > n_samples * 0.1:
                    self.model, self.selected_model_name = Lasso(), "Lasso Regression"
                    self.selection_rationale.append("High dimensionality relative to samples. Lasso performs automatic feature selection via L1 penalty.")
                else:
                    self.model, self.selected_model_name = RandomForestRegressor(n_estimators=100), "Random Forest"
                    self.selection_rationale.append("Standard scale dataset. Random Forest was chosen for its non-linear modeling capabilities and robustness to outliers.")
            else:
                if n_samples < 500:
                    self.model, self.selected_model_name = LogisticRegression(max_iter=1000), "Logistic Regression"
                    self.selection_rationale.append("Small sample size. Logistic regression offers a stable, high-bias baseline.")
                else:
                    self.model, self.selected_model_name = GradientBoostingClassifier(), "Gradient Boosting"
                    self.selection_rationale.append("Large dataset with complex boundaries. Gradient Boosting iteratively corrects errors for maximum accuracy.")

    def preprocess(self):
        self.df.dropna(subset=[self.target_col], inplace=True)
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col != self.target_col:
                if self.df[col].nunique() < 50:
                    self.categorical_mappings[col] = self.df[col].dropna().unique().tolist()
                else: self.df.drop(columns=[col], inplace=True)

        if self.task_type == "classification" and self.df[self.target_col].dtype == 'object':
             le = LabelEncoder(); self.df[self.target_col] = le.fit_transform(self.df[self.target_col])

        valid_cat = [c for c in self.categorical_mappings.keys() if c in self.df.columns]
        if valid_cat: self.df = pd.get_dummies(self.df, columns=valid_cat, drop_first=True)

        X = self.df.drop(columns=[self.target_col])
        self.training_columns = X.columns.tolist()
        imputer, scaler = SimpleImputer(strategy='median'), StandardScaler()
        X_final = scaler.fit_transform(imputer.fit_transform(X))
        return X_final, self.df[self.target_col], imputer, scaler

    def train_and_visualize(self, X, y):
        # Validation: Check if we have enough classes for classification
        if self.task_type == "classification" and len(np.unique(y)) < 2:
            raise ValueError(f"Target variable '{self.target_col}' only contains one class after cleaning. Classification requires at least 2 unique classes.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
                                                            stratify=y if self.task_type == "classification" else None)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # 1. Performance Charts (Plotly)
        if self.task_type == "regression":
            self.metrics = {"R2": r2_score(y_test, y_pred), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))}
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, 
                             title="Prediction Accuracy", template="plotly_dark")
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
            self.plots['performance'] = fig
        else:
            self.metrics = {"Accuracy": accuracy_score(y_test, y_pred)}
            cm = confusion_matrix(y_test, y_pred)
            self.plots['performance'] = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                                                 title="Confidence Matrix", template="plotly_dark")

        # 2. Importance Chart
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)[0] if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
        else:
            # Fallback para modelos sin importancia directa (KNN, etc.)
            imp = np.ones(len(self.training_columns)) / len(self.training_columns)
        
        self.feature_importance = dict(sorted(zip(self.training_columns, imp), key=lambda x: x[1], reverse=True))
        top_df = pd.DataFrame(list(self.feature_importance.items())[:10], columns=['Feature', 'Importance'])
        self.plots['importance'] = px.bar(top_df, x='Importance', y='Feature', orientation='h', 
                                         title="Top 10 Decision Drivers", template="plotly_dark", color='Importance')

    def generate_narrative(self) -> str:
        return f"""
### 🧠 KikoLens Decision Engine: {self.selected_model_name}
**Rationale:** {self.selection_rationale[0]}

**Technical Workflow:**
1. **Analysis:** Target `{self.target_col}` identified as **{self.task_type.upper()}**.
2. **Preprocessing:** Applied Median Imputation for missing values and Standard Scaling for numeric stability.
3. **Encoding:** Converted categorical variables into mathematical flags using One-Hot Encoding.
4. **Optimization:** Selected the **{self.selected_model_name}** architecture because it balances complexity and generalization for your specific data volume.
"""

def run_automl_analysis(df: pd.DataFrame, target_col: str, manual_type: str = None):
    explainer = AutoMLExplainer(df, target_col)
    explainer.analyze_task(manual_type)
    X, y, imputer, scaler = explainer.preprocess()
    explainer.train_and_visualize(X, y)
    return {
        "metrics": explainer.metrics, "narrative": explainer.generate_narrative(),
        "model": explainer.model, "imputer": imputer, "scaler": scaler,
        "columns": explainer.training_columns, "categorical_mappings": explainer.categorical_mappings,
        "task_type": explainer.task_type, "feature_importance": explainer.feature_importance, 
        "target": target_col, "plots": explainer.plots
    }
