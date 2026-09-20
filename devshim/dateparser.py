import pandas as pd


def parse(s, *a, **k):
    try:
        return pd.Timestamp(s).to_pydatetime()
    except Exception:
        return None
