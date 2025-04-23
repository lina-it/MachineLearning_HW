import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, List


def split_train_val(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets, stratifying by the target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys 'train' and 'val' mapping to respective DataFrames.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(
    df_dict: Dict[str, pd.DataFrame], 
    input_cols: List[str], 
    target_col: str
) -> Dict[str, Any]:
    """
    Separates features and targets from training and validation sets.

    Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing 'train' and 'val' DataFrames.
        input_cols (List[str]): List of input feature column names.
        target_col (str): Name of the target column.

    Returns:
        Dict[str, Any]: Dictionary with separated features and targets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def impute_missing_values(
    data: Dict[str, Any], 
    numeric_cols: List[str]
) -> None:
    """
    Imputes missing numeric values with the mean of the training set.

    Parameters:
        data (Dict[str, Any]): Data dictionary containing input features.
        numeric_cols (List[str]): List of numeric column names.
    """
    imputer = SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])


def scale_numeric_features(
    data: Dict[str, Any], 
    numeric_cols: List[str]
) -> None:
    """
    Scales numeric features to the [0, 1] range using MinMaxScaler.

    Parameters:
        data (Dict[str, Any]): Data dictionary containing input features.
        numeric_cols (List[str]): List of numeric column names.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])


def encode_categorical_features(
    data: Dict[str, Any], 
    categorical_cols: List[str]
) -> None:
    """
    Applies one-hot encoding to categorical features.

    Parameters:
        data (Dict[str, Any]): Data dictionary containing input features.
        categorical_cols (List[str]): List of categorical column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'].drop(columns=categorical_cols),
             pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)],
            axis=1
        )
    data['encoded_cols'] = encoded_cols


def preprocess_data(
    df: pd.DataFrame, 
    target_col: str
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: splits data, imputes missing values, scales numerics, and encodes categoricals.

    Parameters:
        df (pd.DataFrame): The raw input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        Dict[str, Any]: A dictionary containing processed training and validation features and targets.
    """
    exclude_cols = {'id', 'RowNumber', 'CustomerId', 'Surname', target_col}
    input_cols = [col for col in df.columns if col not in exclude_cols]

    df_splits = split_train_val(df, target_col)
    data = create_inputs_targets(df_splits, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    impute_missing_values(data, numeric_cols)
    scale_numeric_features(data, numeric_cols)
    encode_categorical_features(data, categorical_cols)

    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
    }