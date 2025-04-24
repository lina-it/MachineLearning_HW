import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, List, Optional


class PreprocessingArtifacts:
    def __init__(self):
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.encoded_cols: List[str] = []


def split_train_val(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets with stratification.

    Parameters:
        df (pd.DataFrame): The complete dataset.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed for random number generation.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'train' and 'val' DataFrames.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(
    df_dict: Dict[str, pd.DataFrame],
    input_cols: List[str],
    target_col: str
) -> Dict[str, Any]:
    """
    Separates input features and target values for training and validation sets.

    Parameters:
        df_dict (Dict[str, pd.DataFrame]): Dictionary with 'train' and 'val' DataFrames.
        input_cols (List[str]): List of columns to use as input features.
        target_col (str): Name of the target column.

    Returns:
        Dict[str, Any]: Dictionary containing input features and target labels for each split.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def impute_missing_values(
    data: Dict[str, Any],
    numeric_cols: List[str],
    artifacts: PreprocessingArtifacts
) -> None:
    """
    Imputes missing values in numeric columns using the mean strategy.

    Parameters:
        data (Dict[str, Any]): Dictionary containing input features.
        numeric_cols (List[str]): List of numeric columns.
        artifacts (PreprocessingArtifacts): Object to store fitted imputer.
    """
    artifacts.imputer = SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = artifacts.imputer.transform(data[f'{split}_inputs'][numeric_cols])


def scale_numeric_features(
    data: Dict[str, Any],
    numeric_cols: List[str],
    artifacts: PreprocessingArtifacts
) -> None:
    """
    Scales numeric features to [0, 1] range using MinMaxScaler.

    Parameters:
        data (Dict[str, Any]): Dictionary containing input features.
        numeric_cols (List[str]): List of numeric columns.
        artifacts (PreprocessingArtifacts): Object to store fitted scaler.
    """
    artifacts.scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = artifacts.scaler.transform(data[f'{split}_inputs'][numeric_cols])


def encode_categorical_features(
    data: Dict[str, Any],
    categorical_cols: List[str],
    artifacts: PreprocessingArtifacts
) -> None:
    """
    One-hot encodes categorical features and integrates them into the dataset.

    Parameters:
        data (Dict[str, Any]): Dictionary containing input features.
        categorical_cols (List[str]): List of categorical columns.
        artifacts (PreprocessingArtifacts): Object to store fitted encoder and column names.
    """
    artifacts.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    artifacts.encoded_cols = list(artifacts.encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val']:
        encoded = artifacts.encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'].drop(columns=categorical_cols),
             pd.DataFrame(encoded, columns=artifacts.encoded_cols, index=data[f'{split}_inputs'].index)],
            axis=1
        )


def preprocess_data(
    df: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:
    """
    Executes full preprocessing pipeline on training dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame with features and target.
        target_col (str): Name of the target column.

    Returns:
        Dict[str, Any]: Dictionary with preprocessed training and validation sets, and preprocessing artifacts.
    """
    artifacts = PreprocessingArtifacts()

    exclude_cols = {'id', 'RowNumber', 'CustomerId', 'Surname', target_col}
    input_cols = [col for col in df.columns if col not in exclude_cols]

    df_splits = split_train_val(df, target_col)
    data = create_inputs_targets(df_splits, input_cols, target_col)

    artifacts.numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    artifacts.categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    impute_missing_values(data, artifacts.numeric_cols, artifacts)
    scale_numeric_features(data, artifacts.numeric_cols, artifacts)
    encode_categorical_features(data, artifacts.categorical_cols, artifacts)

    X_train = data['train_inputs'][artifacts.numeric_cols + artifacts.encoded_cols]
    X_val = data['val_inputs'][artifacts.numeric_cols + artifacts.encoded_cols]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'preprocessor': artifacts,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    artifacts: PreprocessingArtifacts
) -> pd.DataFrame:
    """
    Preprocesses new input data using fitted transformers from the training phase.

    Parameters:
        new_df (pd.DataFrame): New data to transform.
        artifacts (PreprocessingArtifacts): Fitted preprocessing components and column info.

    Returns:
        pd.DataFrame: Preprocessed feature matrix ready for inference.
    """
    imputed = artifacts.imputer.transform(new_df[artifacts.numeric_cols])
    scaled = artifacts.scaler.transform(imputed)

    encoded = artifacts.encoder.transform(new_df[artifacts.categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=artifacts.encoded_cols, index=new_df.index)

    numeric_df = pd.DataFrame(scaled, columns=artifacts.numeric_cols, index=new_df.index)
    return pd.concat([numeric_df, encoded_df], axis=1)
