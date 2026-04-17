import re

with open("data/loader.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add _optimize_numeric_types
downcast_func = """
def _optimize_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    '''Lossless downcasting of numeric columns to save memory and speed up cache hits.'''
    for col in df.select_dtypes(include=["float"]):
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["integer"]):
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def _drop"""
content = re.sub(r"def _drop", downcast_func, content, count=1)

# Add it to the read functions
new_excel = """        df = _coerce_numeric_like_object_columns(df)
        df = _optimize_numeric_types(df)
        df = _drop_high_cardinality_geo"""
content = re.sub(r"        df = _coerce_numeric_like_object_columns\(df\)\n        df = _drop_high_cardinality_geo", new_excel, content)

new_csv1 = """                df = _coerce_numeric_like_object_columns(df)
                df = _optimize_numeric_types(df)
                df = _drop_high_cardinality_geo"""
content = re.sub(r"                df = _coerce_numeric_like_object_columns\(df\)\n                df = _drop_high_cardinality_geo", new_csv1, content)

new_csv2 = """        df = _coerce_numeric_like_object_columns(df)
        df = _optimize_numeric_types(df)
        df = _drop_high_cardinality_geo"""
content = re.sub(r"        df = _coerce_numeric_like_object_columns\(df\)\n        df = _drop_high_cardinality_geo", new_csv2, content)

with open("data/loader.py", "w", encoding="utf-8") as f:
    f.write(content)
