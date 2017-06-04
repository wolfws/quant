# TimeSeriesFunctions.dates_insample_monthly(insample_start_year,insample_end_year)
# TimeSeriesFunctions.dates_outsample_monthly(outsample_start_year,outsample_end_year)
# TimeSeriesFunctions.returnsMonthly(self.dfs,self.column_name)

import datetime as dt
import pandas as pd
import numpy as np
from yahoo_finance import Share

class TimeSeriesFunctions(object): 
    """
    Financial analysis functions
    """
    @staticmethod
    def sharpe_ratio_snp500(returns,date,column_name):
        """Sharpe ratio for a month ended with date"""
        day = pd.to_datetime(date)
        day_minus_month = day - dt.timedelta(days=30)
        day = str(day)[:10] # conversion to string
        day_minus_month = str(day_minus_month)[:10] # conversion to string
        snp500_df = get_stock_df('^GSPC',day_minus_month,day)
        price_snp500 = snp500_df[column_name].values
        returns_snp500 = price_snp500[1:] / price_snp500[:-1]
        len_returns = len(returns)
        len_snp500 = len(returns_snp500)
        len_min = min(len_returns,len_snp500)
        returns = returns[:len_min]
        returns_snp500 = returns_snp500[:len_min]
        return (returns - returns_snp500).mean() / (returns - returns_snp500).std()

    @staticmethod
    def sharpe_ratio(returns1,returns2):
        """Sharpe ratio for two datasets"""
        len1 = len(returns1)
        len2 = len(returns2)
        len_min = min(len1,len2)
        returns1 = returns1[:len_min]
        returns2 = returns2[:len_min]
        return (returns1 - returns2).mean() / (returns1 - returns2).std()

    @staticmethod
    def portfolio_return_monthly(ticker_list,weight_dict,date,column_name):
        """Monthly portfolio return"""
        # weight_dict = {'AAPL': 0.05, ...}
        day = pd.to_datetime(date)
        day_minus_month = day - dt.timedelta(days=30)
        day = str(day)[:10] # conversion to string
        day_minus_month = str(day_minus_month)[:10] # conversion to string
        portfolio_returns_list = []
        for ticker in ticker_list:
            #print (ticker)
            df = get_stock_df(ticker,day_minus_month,day)
            prices = df[column_name].values
            returns = (prices[1:] / prices[:-1] - 1)
            portfolio_returns_list.append(returns * weight_dict[ticker])
        len_pr_array = np.zeros([len(ticker_list)])
        for i in range(len(portfolio_returns_list)): # cleaning and correcting length
            len_pr_array[i] = len(portfolio_returns_list[i])
        min_len_pr = int(len_pr_array[len_pr_array>0].min())
        for i in range(len(portfolio_returns_list)):
            if portfolio_returns_list[i].size > 0:
                portfolio_returns_list[i] = portfolio_returns_list[i][:min_len_pr]
            else:
                portfolio_returns_list[i] = np.zeros([min_len_pr])
        return sum(portfolio_returns_list)

    @staticmethod
    def returnsMonthly(dfs, column_name, dates_insample_monthly, dates_outsample_monthly):   #  'Close'
        """Monthly returns for all stocks for given dates"""
        d = dates_insample_monthly + dates_outsample_monthly
        zero_date = pd.to_datetime(d[0]) - dt.timedelta(days=30)
        zero_date = str(zero_date)[:10]
        d.insert(0,zero_date)
        slice = {}
        ret_df_dict = {}
        failed_tickers = []
        for key in list(dfs.keys()): # tickers
            df = dfs[key]
            dates_asof = []
            for date in d:
                dates_asof.append(df.index.asof(date))
            dates_asof = pd.DatetimeIndex(dates_asof) # pandas index of monthly dates

            try:
                slice[key] = df.loc[dates_asof]
                ret = slice[key][column_name].values[1:] / slice[key][column_name].values[:-1] - 1
                ret_df = pd.DataFrame(ret,index=slice[key].index.values[1:],columns=['Return'])
                ret_df_dict[key] = ret_df
            except:
                failed_tickers.append(key)
                print("loading error: {}".format(key))
                            
        return ret_df_dict,  failed_tickers

    @staticmethod
    def dates_monthly(startyear,endyear):
        dates = []
        for y in range(startyear,endyear):
            for m in range(1,13):
                date = str(dt.date(y,m,1))
                dates.append(date)
        return dates
    
    @staticmethod
    def bd_range(start_date, end_date):
        """Business day range"""
        return pd.bdate_range(start_date, end_date)

