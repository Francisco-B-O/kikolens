# 💎 KikoLens: The Ultimate AI-Powered Data Intelligence Suite

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Ollama Local AI](https://img.shields.io/badge/AI-Ollama%20Local-orange.svg?style=for-the-badge&logo=openai&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)

<h3><i>Data intelligence, redefined for the terminal.</i></h3>

</div>

---

## 🌟 Introduction

**KikoLens** is a high-performance, privacy-first intelligence engine designed for modern data scientists, engineers, and analysts. By seamlessly integrating **local Large Language Models (LLMs)** via Ollama, KikoLens transforms raw data into professional, interactive, and actionable intelligence—all without your data ever leaving your machine.

Whether you need to generate a full executive report, clean a messy dataset, or ask questions in plain English, KikoLens empowers you to do it all from the command line with unparalleled speed and style.

### ✨ Key Features
*   **🛡️ 100% Private & Local**: Zero data egress. All AI analysis is performed on your hardware using Ollama.
*   **🚀 Universal Data Loader**: Native support for CSV, JSON, Excel, Parquet, and SQL Databases (PostgreSQL, MySQL, SQLite).
*   **📊 Executive-Grade Reporting**: One-click generation of unified HTML reports with glassmorphism UI, interactive charts, and AI narratives.
*   **🧠 Cognitive Automation**: Automated machine learning (AutoML) for predictive modeling and feature importance analysis.
*   **💬 Natural Language Interface**: Chat with your data using the `ask` command to generate SQL and get answers instantly.
*   **🧼 Intelligent Cleaning**: AI-driven diagnostics that not only find errors but write the Python code to fix them.

---

## 🛠️ Installation & Setup

### 1. System Requirements
*   **Python 3.12+**
*   **Ollama Runtime**: Required for AI features. [Download Ollama](https://ollama.com/)
*   **Git**: For cloning the repository.

### 2. Initialize AI Model
KikoLens requires a local LLM to function. We recommend `llama3.1` for the best balance of speed and intelligence.
```bash
ollama pull llama3.1
```

### 3. Install KikoLens
Clone the repository and install it in editable mode to get immediate access to the CLI.
```bash
git clone https://github.com/Francisco-B-O/KikoLens.git
cd kikolens
pip install -e .
```

---

## 🚀 The Command Suite

KikoLens provides a comprehensive suite of commands to handle every stage of the data lifecycle.

### 🏆 `report`: The 360° Intelligence Dossier
The flagship command. Generates a unified, high-fidelity HTML report containing statistics, correlations, AI insights, visualizations, and a cleaning plan.

```bash
kikolens report <FILEPATH> [OPTIONS]
```
| Option | Description | Default |
| :--- | :--- | :--- |
| `--target <COLUMN>` | Specify a target column to trigger **AutoML** predictive analysis. | `None` |
| `--model <NAME>` | Choose the local Ollama model to use for insights. | `llama3.1` |
| `--output-dir <PATH>` | Custom directory for report output. | `./kikolens_output` |

**Example:**
```bash
kikolens report data/sales_2024.csv --target "revenue" --model llama3.1
```

---

### 🧠 `ask`: Natural Language Querying
Ask questions about your data in plain English. KikoLens translates your question into optimized SQL, executes it, and presents the results.

```bash
kikolens ask <FILEPATH> "<QUESTION>" [OPTIONS]
```
**Example:**
```bash
kikolens ask data/users.db "Who are the top 5 customers by lifetime value?"
```

---

### 🤖 `predict`: AutoML & Predictive Narrative
Train a machine learning model on your data automatically. Returns a detailed narrative on model performance, feature importance, and metrics.

```bash
kikolens predict <FILEPATH> <TARGET_COLUMN>
```
**Example:**
```bash
kikolens predict data/churn.csv "churn_status"
```

---

### 🧼 `suggest`: AI Data Health Check
Get an expert cleaning strategy. KikoLens analyzes your dataset for missing values, outliers, and inconsistencies, then generates a cleaning plan.

```bash
kikolens suggest <FILEPATH> [OPTIONS]
```
**Example:**
```bash
kikolens suggest data/raw_leads.json
```

---

### 📊 `visualize`: The Visual Gallery
Instantly generate a suite of high-resolution plots, including Histograms, Boxplots, Correlation Heatmaps, and PCA projections.

```bash
kikolens visualize <FILEPATH> [OPTIONS]
```
| Option | Description |
| :--- | :--- |
| `--all` | Generate the comprehensive suite of all available plot types. |
| `--export <FORMAT>` | Export the gallery summary to `html` or `md`. |

**Example:**
```bash
kikolens visualize data/housing.csv --all --export html
```

---

### 💻 `serve`: The Pro Web GUI
Prefer a visual interface? Launch the interactive Streamlit-based dashboard for real-time exploration and filtering.

```bash
kikolens serve [FILEPATH]
```
**Example:**
```bash
kikolens serve data/marketing.xlsx
```

---

### 🔍 `analyze`: Deep Statistical Profiling
Execute a surgical strike of statistical analysis with optional AI insights and cleaning.

```bash
kikolens analyze <FILEPATH> [OPTIONS]
```
| Option | Description |
| :--- | :--- |
| `--clean` | Apply automated data cleaning before analysis. |
| `--plots` | Generate basic statistical plots. |
| `--insights` | Request AI-driven insights on the statistical profile. |
| `--export <FORMAT>` | Export results to `html`, `md`, or `json`. |

---

### 🩺 `diagnose`: Structural Diagnostics
Perform a technical health check on your dataset, identifying data types, missing values, duplicates, and memory usage.

```bash
kikolens diagnose <FILEPATH> [OPTIONS]
```

---

### 🔌 `sql`: Direct Database Interaction
Run raw SQL queries against your local files (treated as tables) or connected remote databases.

```bash
kikolens sql <FILEPATH> "<QUERY>" [OPTIONS]
```
**Example:**
```bash
kikolens sql data.csv "SELECT * FROM data WHERE age > 30 ORDER BY income DESC"
```

---

### 💡 `ai-insight`: Pure Cognitive Analysis
Focus exclusively on getting AI-driven qualitative insights about your dataset without the heavy statistical reporting.

```bash
kikolens ai-insight <FILEPATH> [OPTIONS]
```

---

### 📚 `explain`: The Knowledge Library
Understand complex statistical terms instantly. Uses the built-in glossary or queries the AI for novel concepts.

```bash
kikolens explain "<TERM>"
```
**Example:**
```bash
kikolens explain "heteroscedasticity"
```

---

## 🔌 Data Connectivity

KikoLens features a **Universal Loader** that automatically detects and handles various data sources:

| Source Type | Syntax / Examples |
| :--- | :--- |
| **Local Files** | `data.csv`, `records.json`, `sheet.xlsx`, `archive.parquet` |
| **PostgreSQL** | `postgresql://user:password@localhost:5432/dbname` |
| **MySQL** | `mysql://user:password@localhost:3306/dbname` |
| **SQLite** | `sqlite:///data.db` |
| **Web URL** | `https://example.com/data.csv` |

---

## 📂 Output Structure

By default, KikoLens creates a `kikolens_output` directory in your current workspace:

```text
kikolens_output/
├── data_source_KIKOLENS_REPORT.html  # The Master Report
├── visualize/                        # High-res PNG plots
│   ├── correlation_heatmap.png
│   ├── pca_projection.png
│   └── ...
├── data_source_kikolens_automl.md    # Predictive modeling results
└── ...
```

---

## 🤝 Contributing

We welcome contributions from the community!
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

<div align="center">
  <p><b>KikoLens</b> is an open-source project licensed under the MIT License.</p>
  <p><i>Empowering data professionals with local AI.</i></p>
</div>
