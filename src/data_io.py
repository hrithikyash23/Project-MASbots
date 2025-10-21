from pathlib import Path
import pandas as pd
from typing import List, Dict


EXPECTED_COLUMNS = {"frame", "bot_id", "x", "y"}


def load_runs(data_dir: Path) -> List[Dict]:
    runs = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(p)
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} missing columns: {missing}")
        runs.append({"run_id": p.stem, "path": p, "df": df})
    return runs


