"""Shared configuration and helpers for the Document Summarization project."""
import os, sys
from pyspark.sql import SparkSession

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
RAW_CSV    = os.path.join(DATA_DIR, "reuters_raw.csv")
CLEAN_PQ   = os.path.join(DATA_DIR, "reuters_clean.parquet")
SUMM_PQ    = os.path.join(DATA_DIR, "reuters_summaries.parquet")

# Optional AWS S3 settings (leave S3_BUCKET empty to use local storage only)
S3_BUCKET  = os.getenv("S3_BUCKET", "")
S3_KEY     = os.getenv("S3_KEY", "datasets/reuters_raw.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_spark(app_name="DocSummarization"):
    """Create a local SparkSession."""

    # Use the same Python interpreter that is running this script so that
    # Spark workers have access to all installed packages.
    python_path = sys.executable

    # On Windows, PySpark needs HADOOP_HOME pointing to a folder that contains
    # bin/winutils.exe and bin/hadoop.dll.  The DLL must also be on PATH so
    # the JVM can load it via System.loadLibrary (UnsatisfiedLinkError fix).
    hadoop_home = os.path.join(BASE_DIR, "hadoop_home")
    hadoop_bin  = os.path.join(hadoop_home, "bin")
    os.makedirs(hadoop_bin, exist_ok=True)
    os.environ["HADOOP_HOME"]    = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home
    # Prepend to PATH so hadoop.dll is visible to the JVM
    current_path = os.environ.get("PATH", "")
    if hadoop_bin not in current_path:
        os.environ["PATH"] = hadoop_bin + os.pathsep + current_path

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[1]")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.pyspark.python", python_path)
        .config("spark.pyspark.driver.python", python_path)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark