import pandas as pd
import hashlib
def tidy_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a DataFrame from a wide format to a tidy format.
    """
    id_vars = [col for col in df.columns if not col.startswith('response_')]

    # Metric variables are all columns that do start with 'metric_'.
    metric_vars = [col for col in df.columns if col.startswith('response_')]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=metric_vars,
        var_name='Metric_Raw',
        value_name='llm_response'
    )
    df_melted['model_name'] = df_melted['Metric_Raw'].str.replace('response_', '')
    tidy_df = df_melted[['model_name', 'sample_uuid','prompt','before_merge_without_docstrings','after_merge_without_docstrings', 'llm_response']]

    return tidy_df

def transform_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a DataFrame from a wide format to a tidy format.

    The function melts metric columns, extracts model and metric names,
    and creates a unique sample ID.

    Args:
        df: The input DataFrame in the wide format. It is expected to have
            metric columns prefixed with 'metric_{model_name}_{metric_name}'.
            It must also contain 'before_merge' and 'after_merge' columns
            for generating a sample ID.

    Returns:
        A new DataFrame in a tidy format with the columns:
        'model_name', 'sample_uuid', 'metric_name', 'metric_value'.
    """
    # 1. Identify ID variables and metric variables for melting.
    # ID variables are all columns that do not start with 'metric_'.
    id_vars = [col for col in df.columns if not col.startswith('metric_')]

    # Metric variables are all columns that do start with 'metric_'.
    metric_vars = [col for col in df.columns if col.startswith('metric_')]

    # Melt the dataframe to convert metric columns from wide to long format.
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=metric_vars,
        var_name='Metric_Raw',
        value_name='metric_value'
    )

    # 2. Split the 'Metric_Raw' column to extract Model and Metric_name.
    # The format is 'metric_{model_name}_{metric_name}'. We split on '_'
    # and take the second and third parts.
    metric_parts = df_melted['Metric_Raw'].str.split('_', n=2, expand=True)
    df_melted['model_name'] = metric_parts[1]
    df_melted['metric_name'] = metric_parts[2]


    # 4. Select and reorder columns for the final tidy DataFrame.
    tidy_df = df_melted[['model_name', 'sample_uuid','prompt','traceback_type', 'metric_name', 'metric_value']]

    return tidy_df
def create_uuid_and_category(df: pd.DataFrame ,prompt_str: str) -> pd.DataFrame:
    """
    Reads a file and returns a DataFrame.
    """
    # Guarantee required columns exist
    required_cols = ["before_merge_without_docstrings", "after_merge_without_docstrings"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['sample_uuid'] = df.apply(
        lambda row: hashlib.md5(
            (row['before_merge_without_docstrings'] + row['after_merge_without_docstrings']).encode('utf-8')
        ).hexdigest(),
        axis=1
    )
    df['prompt'] = prompt_str
    return df
