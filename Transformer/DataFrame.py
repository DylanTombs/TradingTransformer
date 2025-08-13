from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataFrameDataset(Dataset):
    def __init__(self, df, flag, size, target, auxilFeatures, featureScaler=None, targetScaler=None, stockColumn = 'ticker'):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        self.df = df.copy()
        self.flag = flag
        self.seqLen, self.labelLen, self.predLen = size
        self.target = target
        self.auxilFeatures = auxilFeatures
        self.stock_column = stockColumn
        self.window_step = 1

        missingFeatures = [f for f in auxilFeatures if f not in df.columns]
        if missingFeatures:
            raise ValueError(f"Missing auxiliary features: {missingFeatures}")
        self.colsData = auxilFeatures + [target]

        if flag == 'train':
            self.featureScaler = StandardScaler()
            self.targetScaler = StandardScaler()
            print("Creating Training DataSet with Scalers")

            if len(self.colsData) > 1:
                if df[auxilFeatures].isnull().values.any():
                    raise ValueError("NaN values found in auxiliary features")
                self.dataXFeatures = self.featureScaler.fit_transform(df[auxilFeatures].values)

                if df[[target]].isnull().values.any():
                    raise ValueError("NaN values found in target")
                self.dataXTarget = self.targetScaler.fit_transform(df[[target]].values)

                self.dataX = np.concatenate([self.dataXFeatures, self.dataXTarget], axis=1)
            else:
                if df[[target]].isnull().values.any():
                    raise ValueError("NaN values found in target")
                self.dataX = self.targetScaler.fit_transform(df[[target]].values)
        else:
            if featureScaler is None or targetScaler is None:
                raise ValueError("Scalers must be provided for validation/test data")

            self.featureScaler = featureScaler
            self.targetScaler = targetScaler

            if len(self.colsData) > 1:

                self.dataXFeatures = featureScaler.transform(df[auxilFeatures].values)
                self.dataXTarget = targetScaler.transform(df[[target]].values)
                self.dataX = np.concatenate([self.dataXFeatures, self.dataXTarget], axis=1)
            else:
                self.dataX = targetScaler.transform(df[[target]].values)

        self.dataY = self.dataX

        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column")

        self.dates = pd.to_datetime(df['date'])
        self.dataStamp = self._processTimeFeatures()

        self.valid_indices = []
        grouped = df.reset_index().groupby(stockColumn)  # reset_index for positional slicing
        for _, group in grouped:
            start_positions = list(range(0, len(group) - (self.seqLen + self.predLen) + 1, self.window_step))
            for pos in start_positions:
                self.valid_indices.append(group['index'].iloc[pos])

    def _processTimeFeatures(self):
        timeSteps = pd.DataFrame(index=self.dates)
        timeSteps['month'] = timeSteps.index.month
        timeSteps['day'] = timeSteps.index.day
        timeSteps['weekday'] = timeSteps.index.weekday
        return timeSteps.values.astype(np.float32)
    def _processTimeFeaturesLarge(self):
        dates = self.dates
        timeSteps = pd.DataFrame(index=dates)
    
        # Cyclical month encoding
        timeSteps['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
        timeSteps['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
    
        # Cyclical day encoding
        timeSteps['day_sin'] = np.sin(2 * np.pi * dates.dt.day / 31)
        timeSteps['day_cos'] = np.cos(2 * np.pi * dates.dt.day / 31)
    
        # Cyclical weekday encoding
        timeSteps['weekday_sin'] = np.sin(2 * np.pi * dates.dt.weekday / 7)
        timeSteps['weekday_cos'] = np.cos(2 * np.pi * dates.dt.weekday / 7)
    
        # Day of year normalized (trend position in year)
        timeSteps['day_of_year'] = dates.dt.dayofyear / 365.0

        if self.stock_column in self.df.columns:
            time_index = np.zeros(len(self.df))
            for _, group in self.df.groupby(self.stock_column):
                idx = group.index
                normalized_idx = np.arange(len(group)) / len(group)
                time_index[idx] = normalized_idx
            timeSteps['time_index'] = time_index

        return timeSteps.values.astype(np.float32)

    def __getitem__(self, index):
        sBegin = index
        sEnd = sBegin + self.seqLen

        if self.flag == "pred":
            seqX = self.dataX[sBegin:sEnd]
        
            # Create properly sized output tensor (labelLen + predLen)
            seqY = np.zeros((self.labelLen + self.predLen, self.dataX.shape[1]))
        
            # Fill available future data if possible
            if len(self.dataX) > sEnd:
                available = min(self.labelLen + self.predLen, len(self.dataX) - sEnd)
                seqY[:available] = self.dataX[sEnd:sEnd+available]
            
            seqXMark = self.dataStamp[sBegin:sEnd]
            seqYMark = self.dataStamp[-len(seqY):] 
        else: 
            rBegin = sEnd - self.labelLen
            rEnd = rBegin + self.labelLen + self.predLen

            seqX = self.dataX[sBegin:sEnd]
            seqY = self.dataY[rBegin:rEnd]
            seqXMark = self.dataStamp[sBegin:sEnd]
            seqYMark = self.dataStamp[rBegin:rEnd]

        return seqX, seqY, seqXMark, seqYMark

    def __len__(self):
        if self.flag == "pred":
            return 1
        return len(self.dataX) - self.seqLen - self.predLen + 1

    def inverseTransform(self, data, isTarget=True):
        if isTarget:
            return self.targetScaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            if len(self.colsData) > 1:
                return self.featureScaler.inverse_transform(data)
            else:
                raise ValueError("No feature scaler available when only target feature is used")
