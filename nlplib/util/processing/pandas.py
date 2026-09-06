import pandas as pd


def dict_column_to_columns(df, col, is_multiindex=True):
    """
    Converts the given column, whose elements are dictionaries, into
    separate columns.

    Parameters
    ----------
    is_multiindex:
        the resulting format of the column names
        True: a two-level index [original column name, dict key]
        False: col_*dict key*
    """
    df = df.apply(lambda row: row[col], axis=1, result_type='expand')

    if is_multiindex:  
        columns = pd.MultiIndex.from_arrays([[col] * len(df.columns), df.columns]) 
    else:
        columns = [col + '_' + column for column in df.columns.to_list()]

    df.columns = columns
    return df


def dict_columns_to_columns(df, cols=None, is_multiindex=False):
    """
    Expands the given dict-valued columns (if not specified, only
    those that contain dictionaries) into new columns.
    """
    if not cols:
        cols = [col for col in df.columns.tolist() if isinstance(df[col][0], dict)]
    return pd.concat([ 
        dict_column_to_columns(df, col, is_multiindex=is_multiindex) if col in cols else df[col]
        for col in df.columns
    ], axis=1)
