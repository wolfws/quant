# FinancialData.read_stock_names(STOCKS_FILE)
import pandas as pd
import quandl as Quandl
from yahoo_finance import Share

class FinancialData(object):
    """
    Read financial data
    """
    @staticmethod
    def read_stock_names(STOCKS_FILE):
        """Read stock names from input file (S&P500 stocks)"""
        names_df = pd.read_csv(STOCKS_FILE)
        return list(names_df.ix[:,1].values)

    @staticmethod
    def get_stock_df(ticker,start_date,end_date):
        """Get stock dataframe"""
        share = Share(ticker)
        share_hist = share.get_historical(start_date,end_date)
        len_share_hist = len(share_hist)
        dates = ['']*len_share_hist
        open = [0.]*len_share_hist
        close = [0.]*len_share_hist
        high = [0.]*len_share_hist
        low = [0.]*len_share_hist
        volume = [0.]*len_share_hist
        adj_close = [0.]*len_share_hist
        for i in range(len_share_hist):
            dates[i] = share_hist[i][DATE_STR]
            open[i] = float(share_hist[i][OPEN_STR])
            close[i] = float(share_hist[i][CLOSE_STR])
            adj_close[i] = float(share_hist[i][ADJ_CLOSE_STR])
            high[i] = float(share_hist[i][HIGH_STR])
            low[i] = float(share_hist[i][LOW_STR])
            volume[i] = float(share_hist[i][VOLUME_STR])
        df = pd.DataFrame(open, index = pd.to_datetime(dates), columns=[OPEN_STR])
        df[CLOSE_STR] = close
        df[ADJ_CLOSE_STR] = adj_close
        df[HIGH_STR] = high
        df[LOW_STR] = low
        df[VOLUME_STR] = volume
        df.index.name = DATE_STR
        return df.sort_index()
    
    @staticmethod
    def getSingleTS(TOKEN,ticker,start,end):
        product = "GOOG/NYSE_" + ticker
        d = Quandl.get(product, authtoken=TOKEN, trim_start=start, trim_end=end)
        d['ticker'] = ticker
        return d

