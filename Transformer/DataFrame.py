from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from Transformer.timefeatures import time_features
import pandas as pd
import numpy as np

class DataFrameDataset(Dataset):
    def __init__(self, df, flag, size, target, auxil_features, timeenc=0, freq='d', feature_scaler=None, target_scaler=None):
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        #self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.auxil_features = auxil_features
        
        # Process data columns
        missing_features = [f for f in auxil_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing auxiliary features: {missing_features}")
        self.cols_data = auxil_features + [target]
        df_data = df[self.cols_data]
        # Handle scaling
        if flag == 'train':
            # Initialize new scalers for training data
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            
            if len(self.cols_data) > 1:
                # Safety check for NaN/inf
                if df[auxil_features].isnull().values.any():
                    raise ValueError("NaN values found in auxiliary features")
                self.data_x_features = self.feature_scaler.fit_transform(df[auxil_features].values)
                
                if df[[target]].isnull().values.any():
                    raise ValueError("NaN values found in target")
                self.data_x_target = self.target_scaler.fit_transform(df[[target]].values)
                
                self.data_x = np.concatenate([self.data_x_features, self.data_x_target], axis=1)
            else:
                if df[[target]].isnull().values.any():
                    raise ValueError("NaN values found in target")
                self.data_x = self.target_scaler.fit_transform(df[[target]].values)
        else:
            # For val/test, use provided scalers
            if feature_scaler is None or target_scaler is None:
                raise ValueError("Scalers must be provided for validation/test data")
                
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            
            if len(self.cols_data) > 1:
                # Check feature dimensions match
                if len(auxil_features) != 8:
                    raise ValueError("Auxiliary features dimension mismatch with scaler")
                
                self.data_x_features = feature_scaler.transform(df[auxil_features].values)
                self.data_x_target = target_scaler.transform(df[[target]].values)
                self.data_x = np.concatenate([self.data_x_features, self.data_x_target], axis=1)
            else:
                self.data_x = target_scaler.transform(df[[target]].values)
        
        self.data_y = self.data_x  # For MS mode
        
        # Process time features
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column")
            
        self.dates = pd.to_datetime(df['date'])
        self.data_stamp = self._process_time_features()

    def _process_time_features(self):
        """Process time features based on timeenc parameter"""
        if self.timeenc == 0:
            time_steps = pd.DataFrame(index=self.dates)
            time_steps['month'] = time_steps.index.month
            time_steps['day'] = time_steps.index.day
            time_steps['weekday'] = time_steps.index.weekday
            return time_steps.values.astype(np.float32)
        elif self.timeenc == 1:
            try:
                return time_features(self.dates.values, freq=self.freq).transpose(1, 0).astype(np.float32)
            except:
                time_steps = pd.DataFrame(index=self.dates)
                time_steps['dayofweek'] = time_steps.index.dayofweek / 6.0 - 0.5
                time_steps['dayofmonth'] = (time_steps.index.day - 1) / 30.0 - 0.5
                time_steps['dayofyear'] = (time_steps.index.dayofyear - 1) / 365.0 - 0.5
                return time_steps.values.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, is_target=True):
        """
        Inverse transform data back to original scale
        Args:
            data: Input data to inverse transform
            is_target: Whether the data is target values (use target scaler) or features
        """
        if is_target:
            return self.target_scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            if len(self.cols_data) > 1:
                return self.feature_scaler.inverse_transform(data)
            else:
                raise ValueError("No feature scaler available when only target feature is used")