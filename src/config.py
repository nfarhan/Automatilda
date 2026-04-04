# --------------------------------------------------
# Step 0.1: Import & Library Setup
# --------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# Core
import os
import logging
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Data
import numpy as np
import pandas as pd
import duckdb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Modeling (we import here to confirm installation)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

print("All libraries imported successfully")
print(f"NumPy:   {np.__version__}")
print(f"Pandas:  {pd.__version__}")
print(f"DuckDB:  {duckdb.__version__}")
print(f"XGBoost: {xgb.__version__}")


# --------------------------------------------------
# Step 0.2: Project Constants & Path Configuration
# --------------------------------------------------

# Paths
DATA_DIR: PATH = Path("../data")
OUTPUT_DIR: PATH = Path("../outputs")
FIGURES_DIR: PATH = OUTPUT_DIR / "figures"
MODELS_DIR: PATH = OUTPUT_DIR / "models"

RAW_DATA_PATH: PATH = DATA_DIR / "2017_Yellow_Taxi_Trip_Data.csv"
CLEAN_DATA_PATH: PATH = DATA_DIR / "taxi_clean.parquet"

# Shared in-memory DuckDB connection
CON: duckdb.DuckDBPythonConnection = duckdb.connect(database=":memory:")

# Modelling
DATASET_SIZE: int = 1000000
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
TARGET_COL: str = "fare_amount"

# Color palette (consistent across all plots)
PALETTE: dict = {
    "primary": "#2563EB",    # blue
    "secondary": "#16A34A",    # green
    "accent": "#DC2626",    # red
    "neutral": "#6B7280",    # grey
    "background": "#F9FAFB",    # off-white
}

# Logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%H:%M:%S"
)
logger = logging.getLogger("taxi_fare")


# Create directory tree
def create_project_dirs():
    """
    Create all of the necessary directories listed above.
    """
    for _dir in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR]:
        _dir.mkdir(parents=True, exist_ok=True)

logger.info("Directory tree verified.")
logger.info(f"Target column: {TARGET_COL}")
logger.info(f"Test split size: {TEST_SIZE}")
logger.info(f"Random state: {RANDOM_STATE}")


# --------------------------------------------------
# Step 0.3: Data Loading
# --------------------------------------------------

def load_raw_data(url: str, path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw TLC CSV into a pandas DataFrame.

    Datetime columns are parsed at load time. Column names are 
    normalized to lowercase and stripped of whitespaces so all 
    downstream phases use a consistent namming contention.

    Resolution order:
        1. Local file at ``path``: used if it exists.
        2. Remove ``url``: downloaded if local file is missing.
        3. Raise an error: if both sources fail.

    Parameters
    -----------
    url: str
        Fallback URL to read the CSV from if the local file is absent.
    path: Path
        File-system path to the raw CSV file.
    

    Returns
    --------
    pd.DataFrame
        Raw dataset with parsed datetimes and normalized column names.

    Raises
    -------
    FileNotFoundError
        If the local file is missing and the URL is unreachable or invalid.
    """
    parse_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    
    if path.exists():
        logger.info(f"Loading data from local path: {path}")
        df = pd.read_csv(path, nrows=DATASET_SIZE, parse_dates=parse_cols)
        
    elif url:
        logger.warning(f"Local file not found at '{path}'. Attempting URL...")
        
        try:    
            df = pd.read_csv(url, parse_dates=parse_cols)
            logger.info(f"Data loaded from URL: {url}")
        except Exception as e:
            raise FileNotFoundError(
                f"Local file missing at '{path}' and URL failed.\n"
                f"URL attempted: {url}\n"
                f"Error: {e}"
            )
    else:
        raise FileNotFoundError(
            f"Local file not found at '{path}' and no fallback URL was provided.\n"
            "Please place '2017_Yellow_Taxi_Trip_Data.csv' inside the data/ directory or provide a valid URL."
        )

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    logger.info(f"Raw data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


# ------------------------------------------------------------
# Step 0.4: DuckDB Initialization & Table Registration
# ------------------------------------------------------------

def register_duckdb_table(df: pd.DataFrame, table_name: str = "trips") -> None:
    """
    Register a pandas DataFrame as a DuckDB virtual table.

    Once registered, any phase can query the table via ``quick_sql()``
    using the global ``CON`` connection without duplicating data in memory.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to expose to DuckDB.
    table_name: str, optional
        Name to use in SQL queries (default ``"trips"``).
    """    
    CON.register(table_name, df)
    row_count = CON.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    logger.info(f"DuckDB table '{table_name}' registered ({row_count:,} rows).")


def quick_sql(query: str) -> pd.DataFrame:
    """
    Execute a SQL query against the in-memory DuckDB database.

    Parameters
    ----------
    query: str
        Valid DuckDB  SQL query

    Returns
    -------
    pd.DataFrame
        Query result as a pandas DataFrame

    Examples
    --------
    >>> quick_sql("SELECT vendroid, COUNT(*) AS n FROM trips GROUP BY 1")
    """
    return CON.execute(query).df()


# # Register & smoke-test
# register_duckdb_table(df_raw, table_name="trips")

# quick_sql("""
#     SELECT
#         MIN(tpep_pickup_datetime) AS earliest_pickup,
#         MAX(tpep_pickup_datetime) AS latest_pickup,
#         ROUND(AVG(fare_amount), 2) AS avg_fare_usd,
#         COUNT(*) AS total_trips
#     FROM trips
# """)


# ------------------------------------------------------------
# Step 0.5: Plot Theme & Style Configuration
# ------------------------------------------------------------

def set_plot_theme() -> None:
    """
    Apply a consistent matplotlib/seaborn theme for all project phases.

    Sets figure dimensions, font sizes, spine visibility, grid style,
    and the project color palette so every plot looks uniform without
    any per-figure configuration.
    """
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["background"],
        "axes.edgecolor": "#D1D5DB",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.alpha": 0.6
    })

    sns.set_palette([
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["neutral"],
    ])

    logger.info("Plot theme applied.")


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the figures output directory.

    A timestamp is appended to the filename automatically to prevent
    overwiting previous versions.

    Parameters
    ----------
    fig: plt.Figure
        The figure object to save.
    name: str
        File step (no extension).
    dpi: int, optional
        Resolution in dots-per-inch (default 150).

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = FIGURES_DIR / f"{name}_{ts}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info(f"Figure saved -> {path}")

    return path


# # Verification plot
# fig, ax = plt.subplots()
# ax.bar(
#     ["Sample Trips", "Full Dataset (est.)"],
#     [1_000_000, 113_000_000],
#     color=[PALETTE["primary"], PALETTE["neutral"]],
#     width=0.4
# )
# ax.set_title("Dataset Coverage: Sample vs Full TLC Dataset")
# ax.set_ylabel("Number of Trips")
# ax.yaxis.set_major_formatter(
#     plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
# )
# plt.tight_layout()
# save_figure(fig, "phase0_theme_verification")
# plt.show()


# Run on import
create_project_dirs()
set_plot_theme()
