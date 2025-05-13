import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from get_source_data import *
import features
from typing import List
import glob
from tqdm import tqdm
import yfinance as yf

cleaned_data_dir = 'feature_dataframes/cleaned'
processed_data_dir = 'feature_dataframes/preprocessed'

class DataProcessor:
    """DataProcessor class to handle the preprocessing of stock data for model training and inference.
    
    This class fetches stock data, applies various indicators, adjusts for stock splits, creates sequences and labels,
    splits the data into training, validation, and test sets, scales the data, and saves the processed data.
    """

    def __init__(self, provider:str, tickers:List[str], train_split_amount:float=0.8, val_split_amount:float=0.1, lead:int=2,
                 lag:int=20, inference:bool=False, unknown_y:bool=False, window_size:int=15, step:int=0, start_date:str=None):
        """Initialize the DataProcessor"""
        # Check for valid input
        assert train_split_amount + val_split_amount <= 1, 'Train and validation split amounts must sum to 1 or less'
        assert window_size >= 1, 'Window size must be a positive integer'
        assert lead > 0, 'Lead must be a positive integer'
        assert lag > 0, 'Lag must be a positive integer'
        assert step >= 0, 'Skip must be a non-negative integer'
        provider = eval(provider)
        #assert provider in globals(), f"{provider} is not a valid class in get_source_data"
        self.step = step
        self.window_size = window_size
        self.provider = provider
        self.tickers = tickers
        self.unknown_y = unknown_y
        self.inference = inference or unknown_y
        self.lead = lead
        self.lag = lag
        self.start_date = pd.to_datetime(start_date).tz_localize("America/New_York")

        # Set the split amounts if not in inference mode
        if not inference:
            self.train_split_amount = train_split_amount
            self.val_split_amount = val_split_amount

        # Ensure cleaned and processed data directories exist
        os.makedirs(cleaned_data_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)

    def clean_ticker_dataframe(self, ticker) -> bool:
        """Clean and prepare the ticker dataframe.
            
            Return: bool indicating success of the cleaning process on the given ticker.
        """
        print(f"Cleaning {ticker}")
        # Fetch data for the given ticker
        ticker_df = self.provider.fetch_by_ticker(ticker)
        # If no data is returned, return False
        if ticker_df is None or ticker_df.empty:
            return False
        # Check if the minimum date in the 'date' column is less than or equal to self.start_date
        min_date = ticker_df.index.min()
        if pd.to_datetime(min_date) > self.start_date:
            return False
        # Crop the dataframe at the minimum date
        ticker_df = ticker_df[ticker_df.index >= self.start_date]
        # Exclude data points beyond three standard deviations from the mean (assuming normal distribution)
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = ticker_df[col].mean(skipna=True)
            std = ticker_df[col].std(skipna=True)
            ticker_df = ticker_df[(ticker_df[col] >= mean - 3 * std) & (ticker_df[col] <= mean + 3 * std)]
        ticker_df.reset_index(inplace=True)
        # Fill holes in dataset with previous value
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.ffill(inplace=True)
        ticker_df.dropna(inplace=True)
        # Adjust for stock splits
        self.adjust_for_stock_splits(ticker_df)
        # Track market cap and industry
        yf_ticker = yf.Ticker(ticker)
        ticker_info = yf_ticker.info
        try:
            market_cap = ticker_info.get('marketCap')
            current_price = ticker_info.get('currentPrice')
            ticker_df['market_cap'] = ticker_df['open'] * (int(market_cap / current_price))
            ticker_df.attrs['industry'] = ticker_info.get('industry')
            ticker_df.attrs['ticker'] = ticker  # Store only the ticker string, not the yfinance object
        except Exception as e:
            print(f"Error fetching market cap or industry for {ticker}: {e}")
            return False
        # Sample data on the set interval
        if self.step:
            prev_col = len(ticker_df) - 1
            for col in range(len(ticker_df) - self.step, 0, -self.step):
                segment = ticker_df.iloc[col:prev_col + 1]
                # OHLCV aggregation
                ticker_df.at[prev_col, 'high'] = segment['high'].max()
                ticker_df.at[prev_col, 'low'] = segment['low'].min()
                ticker_df.at[prev_col, 'volume'] = segment['volume'].sum()
                ticker_df.at[prev_col, 'open'] = ticker_df.at[col, 'open']
                prev_col = col - 1
            ticker_df = ticker_df.iloc[self.step - 1 + len(ticker_df) % self.step:: self.step]
        # Calculate VWAP (Volume Weighted Average Price)
        ticker_df['vwap'] = ((ticker_df['close'] + ticker_df['high'] + ticker_df['low']) / 3
                             * ticker_df['volume']).cumsum() / ticker_df['volume'].cumsum()
        # Set 'date' as the index
        ticker_df.set_index('date', inplace=True)
        # Save the cleaned data
        self.save_clean_data(ticker, ticker_df)
        return True

    
    def process_tickers(self, tickers):
        # Load all cleaned dataframes and ensure they have the same columns and date index
        ticker_dfs = []
        for ticker in tickers:
            df = self.load_clean_data(ticker)
            if df is not None:
                ticker_dfs.append((ticker, df))
                df.to_csv(f"{ticker}.csv")
        
        if not ticker_dfs:
            return None
        

        # Ensure all dataframes have the same columns and date index
        # Check that each dataframe has the same index and column titles
        columns_set_list = [set(df.columns) for _, df in ticker_dfs]
        index_set_list = [set(df.index) for _, df in ticker_dfs]
        if not all(cols == columns_set_list[0] for cols in columns_set_list):
            raise ValueError("All dataframes must have the same column titles to interleave.")
        if not all(idx == index_set_list[0] for idx in index_set_list):
            print("Index values that differ between dataframes:")
            for i, idx_set in enumerate(index_set_list):
                if idx_set != index_set_list[0]:
                    diff1 = idx_set - index_set_list[0]
                    diff2 = index_set_list[0] - idx_set
                    print(f"Ticker {tickers[i]}: Extra indices: {diff1}, Missing indices: {diff2}")
            raise ValueError("All dataframes must have the same index to interleave.")

        # Interleave rows by date index
        interleaved_rows = []
        date_index = ticker_dfs[0][1].index
        for date in date_index:
            for ticker, df in ticker_dfs:
                row = df.loc[date].copy()
                row['ticker'] = ticker  # Optionally add ticker column
                interleaved_rows.append(row)
        # Create a new dataframe from interleaved rows
        ticker_df = pd.DataFrame(interleaved_rows)
        ticker_df.set_index('date', inplace=True, drop=False)
        print(ticker_df.head())
        exit()
   
        # Create features
        self.create_features(ticker_df)
        # Drop the original columns
        ticker_df.drop(['close', 'open', 'high', 'low', 'volume', 'vwap'], axis=1, inplace=True)
        # Create sequences and labels
        period_splits = self.create_sequences_and_labels(ticker_df) # Dict of sequences and labels
        return period_splits
    
    def adjust_for_stock_splits(self, ticker_df):
        """Adjust the stock data for stock splits"""
        # Identify the rows where the percentage change is greater than 100%
        ticker_df['temp_pct_change'] = ticker_df['close'][::-1].pct_change()[::-1]
        indices_to_check = ticker_df.index[ticker_df['temp_pct_change'] > 1].tolist()[::-1]
        # Adjust the OHLC and volume that precede the split
        for idx in indices_to_check:
            split_value = int(ticker_df.iloc[idx]['temp_pct_change']) + 1
            # Ensure columns are float before division/multiplication to avoid dtype issues
            cols = [c for c in ticker_df.columns if c != 'date']
            ticker_df[cols] = ticker_df[cols].astype(float)
            ticker_df.loc[0:idx, ['close', 'open', 'high', 'low']] /= split_value
            ticker_df.loc[0:idx, ['volume']] *= split_value
        # Drop the 'temp_pct_change' column
        ticker_df.drop(columns=['temp_pct_change'], inplace=True)
    
    def create_sequences_and_labels(self, ticker_df):
        """Create sequences and labels for the model"""
        period_splits = []
        # Each period consists of a sliding window of 81 steps.
        # Training set: 486 samples, Validation set: 81 samples, Test set: 81 samples.
        seq_splits_length = 486+81+81
        
        # Convert pandas to numpy, keep track of column locations
        # Get the column locations for numpy
        feature_locs = [ticker_df.columns.get_loc(col) for col in ticker_df.columns.tolist()]
        return_rate_loc = ticker_df.columns.get_loc('RETURN_RATE')
        trend_direction_loc = ticker_df.columns.get_loc('TREND_DIRECTION')

        # Convert the dataframe to a numpy array (for performance)
        ticker_df = ticker_df.values
        period_count = 1
        # Step through the dataframe to create sequences, using a sliding window approach for standardization
        for j in tqdm(range(0, len(ticker_df), 81), desc=f"Processing sequences for period {period_count}"):
            period_count += 1
            X, Y = [], []  # Sequence and label arrays
            scaler = StandardScaler()
                
            # Define training sequence  
            sub_seq = ticker_df[j:j+seq_splits_length+1]
            # Fit the scaler to the training sequence
            scaler.fit(sub_seq[:, feature_locs])
            sub_seq[:, feature_locs] = scaler.transform(sub_seq[:, feature_locs])
            # Separate target column standardization
            mean_return_rate = np.mean(sub_seq[:, return_rate_loc])
            std_return_rate = np.std(sub_seq[:, return_rate_loc], ddof=1) # Match the behavior of pandas
            mean_trend_direction = np.mean(sub_seq[:, trend_direction_loc])
            std_trend_direction = np.std(sub_seq[:, trend_direction_loc], ddof=1) # Match the behavior of pandas
            # Scale the target columns
            sub_seq[:, return_rate_loc] = (sub_seq[:, return_rate_loc] - mean_return_rate) / std_return_rate
            sub_seq[:, trend_direction_loc] = (sub_seq[:, trend_direction_loc] - mean_trend_direction) / std_trend_direction
            # Ensure we have enough data for a full window + lead
            if len(sub_seq) < self.lag + self.lead:
                continue

            # Sliding window for X and y
            for i in range(0, len(sub_seq) - self.lag - self.lead + 1):
                # X: lag window of features
                lagged_sequences_array = sub_seq[i:i+self.lag, feature_locs]
                # y: the return_rate and trend_direction at the next time step after lag window + lead-1
                y_return_rate = sub_seq[i+self.lag+self.lead-1, return_rate_loc]
                y_trend_direction = sub_seq[i+self.lag+self.lead-1, trend_direction_loc]
                y = np.array([y_return_rate, y_trend_direction], dtype=np.float32)
                X.append(lagged_sequences_array)
                Y.append(y)
                
            training_sequences = X[486:]
            training_labels = Y[486:]
            validation_sequences = X[486:486+81]
            validation_labels = Y[486:486+81]
            test_sequences = X[486+81:]
            test_labels = Y[486+81:]

            period_splits.append({
                'train': {'X': np.array(training_sequences), 'y': np.array(training_labels)},
                'val': {'X': np.array(validation_sequences), 'y': np.array(validation_labels)},
                'test': {'X': np.array(test_sequences), 'y': np.array(test_labels)},
                'mean': {'return_rate': mean_return_rate, 'trend_direction': mean_trend_direction},
                'std': {'return_rate': std_return_rate, 'trend_direction': std_trend_direction}
            })
        return period_splits
    
    def create_features(self, ticker_df):
        """Apply features to the dataframe"""
        # Apply specified features to the dataframe
        ticker_df['RETURN_RATE'] = ticker_df['open'].pct_change().fillna(0)
        ticker_df['TREND_DIRECTION'] = np.where(ticker_df['RETURN_RATE'] > 0, 1, -1)
        features.apply_alpha360_features(ticker_df)
    

    def create_categoricals(self, ticker_df):
        """Apply categorical features to the dataframe"""
        features.apply_categorical_features(ticker_df)

    def save_dataframe(self, filename, dataframe):
        """Save the dataframe"""
        # Create directories if they don't exist
        os.makedirs('feature_dataframes', exist_ok=True)
        # Save the dataframe
        with open(filename, 'wb') as f:
            pickle.dump(dataframe, f)

    def load_dataframe(self, filename):
        """Load the dataframe"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"File {filename} does not exist.")
            return None

    def save_clean_data(self, ticker, data_splits):
        """Save the clean data"""
        self.save_dataframe(f'{cleaned_data_dir}/{ticker}_cleaned.pkl', data_splits)
    
    def load_clean_data(self, ticker):
        """Load the clean data"""
        # Check if the directory exists
        filename = f'{cleaned_data_dir}/{ticker}_cleaned.pkl'
        return self.load_dataframe(filename)

    def save_data_splits(self, ticker, data_splits):
        """Save the data splits"""
        self.save_dataframe(f'{processed_data_dir}/{ticker}_processed.pkl', data_splits)

    def process_all_tickers(self):
        """Process all tickers"""
        # Clean each ticker dataframe
        for ticker in self.tickers[:]:
            if not self.clean_ticker_dataframe(ticker):
                print(f"Insufficient data quality for {ticker}. Skipping this ticker.")
                self.tickers.remove(ticker)
                continue
        # Feature processing and data splitting
        data_splits = self.process_tickers(self.tickers)
        for ticker in self.tickers:
            if data_splits:
                self.save_data_splits(ticker, data_splits)
            else:
                print(f"Failed to process {ticker}. Skipping this ticker.")
                self.tickers.remove(ticker)
                continue
            del data_splits  # Free up memory

if __name__ == "__main__":
    print("Example Usage")
    provider = 'YahooFinance'
    processor = DataProcessor(provider, ['AAPL'], lag=20, lead=2, train_split_amount=0.90, 
                              val_split_amount=0.05, start_date='2025-04-01')
    columns = processor.process_all_tickers()
    print("Data processing complete.")