
import pandas as pd

def value_counts_df(
        df, 
        columns
    ):
    """
    pandas.value_counts с преобразованием значений группировки в столбцы

    """
    return pd.DataFrame(df.value_counts(subset = ['entity', 'word_normal'])).reset_index()


def dict_column_to_columns(df, col, is_multiindex=True):
    '''
        преобразует заданный столбец с элементами - словарями в отдельные столбцы
        is_multiindex - итоговый формат имен столбцов
            True: двойной индекс [исходное название столбца, ключи словаря]
            False: col_*ключ словаря*
    '''
    df = df.apply(lambda row: row[col], axis=1, result_type='expand')

    if is_multiindex:  
        columns = pd.MultiIndex.from_arrays([[col] * len(df.columns), df.columns]) 
    else:
        columns = [col + '_' + column for column in df.columns.to_list()]

    df.columns = columns

    return df


def dict_columns_to_columns(df, cols=None, is_multiindex=False):
    '''
        раскладывает заданные столбцы-словари (если не задано, то только те что
        содержат словари) в новые столбцы
    '''
    if not cols:
        cols = [col for col in df.columns.tolist() if isinstance(df[col][0], dict)]
    return pd.concat([ 
        dict_column_to_columns(df, col, is_multiindex=is_multiindex) if col in cols else df[col]
        for col in df.columns
    ], axis=1)




def dicts_to_columns_deep(df):
    '''
    todo
    '''
    dfs = []
    for col in df.columns:
        df_col = df[[col]]
        # print(df_col.loc[0].item())
        if isinstance(df_col.loc[0].item(), dict):
            df_col = df.apply(lambda row: row[col], axis=1, result_type='expand')
            df_col = dicts_to_columns_deep(df_col)
        df_col.columns = pd.MultiIndex.from_arrays([[col]*len(df_col.columns), list(zip(df_col.columns))])
        dfs.append(df_col)

    concat_df = pd.concat(dfs, axis=1)

    return concat_df
