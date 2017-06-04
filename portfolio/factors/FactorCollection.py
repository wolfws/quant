import pandas as pd
import datetime as dt
import numpy as np
from pandas.tseries.offsets import BDay

CLOSE_STR = 'Close'
VOLUME_STR = 'Volume'
OPEN_STR = 'Open'
DATE_STR = 'Date'
LOW_STR = 'Low'
HIGH_STR = 'High'
ADJ_CLOSE_STR = 'Adj_Close'
SYMBOL_STR = 'Symbol'


class FactorCollection(object):
         
    # the idea is to shift period 1 day forward for forward testing...
    # not properly implemented as only start date is passed without lookback shifting window
    def cut_dataframe(self,df,start_date=None,end_date=None):
        all_dates = df.index.values
        if not end_date:
            end_date = all_dates.max()
        if not start_date:
            start_date = all_dates.min()
        cut_df = df.ix[df.index.searchsorted(start_date):(1+df.index.searchsorted(end_date))]
        return cut_df
  
    def slopeWeekly(self,df,date,column_name):
        """Price slope of 10 Week Exponential Moving Average (EMA) over 5 weeks"""
        day = df.index.asof(pd.to_datetime(date))
        cut_df = self.cut_dataframe(df,end_date=day)
        ema = pd.ewma(cut_df[column_name], span=50).ix[-25:]
        ema_dates = ema.index.values
        ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
        slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
        return slope

    def volumentumWeekly(self,df,date,column_name):
        """[Price (End of this week) - Price (End of last week)]*[Avg Week Volume / Avg 6 Mo Volume]"""
        day = df.index.asof(pd.to_datetime(date))
        friday1 = df.index.asof(day - dt.timedelta(days=(day.weekday() - 4) % 7, weeks=0))
        friday2 = df.index.asof(day - dt.timedelta(days=(day.weekday() - 4) % 7, weeks=1))
        one_week = [day - dt.timedelta(i) for i in range(7)]
        six_months = [day - dt.timedelta(i) for i in range(180)]
        avg_week_volume = df.loc[one_week].dropna()[column_name].mean()
        avg_six_months_volume = df.loc[six_months].dropna()[column_name].mean()
        return (df[column_name].loc[friday1] - df[column_name].loc[friday2]) * avg_week_volume / avg_six_months_volume

    def volumentumMonthly(self,df,date,column_name):
        """[Price (End of this month) - Price (End of last month)]*[Avg Monthly Volume / Avg 12 Mo Volume]"""
        day = df.index.asof(pd.to_datetime(date))
        end_of_month1 = pd.to_datetime(dt.date(day.year,day.month,1)) - dt.timedelta(days=1)
        end_of_month1 = df.index.asof(pd.to_datetime(end_of_month1))
        end_of_month2 = pd.to_datetime(dt.date(end_of_month1.year,end_of_month1.month,1)) - dt.timedelta(days=1)
        end_of_month2 = df.index.asof(pd.to_datetime(end_of_month2))
        one_month = [df.index.asof(day - dt.timedelta(i)) for i in range(30)]
        twelve_months = [df.index.asof(day - dt.timedelta(i)) for i in range(360)]
        avg_monthly_volume = df.loc[one_month].dropna()[column_name].mean()
        avg_twelve_months_volume = df.loc[twelve_months].dropna()[column_name].mean()
        return (df[column_name].loc[end_of_month1] - df[column_name].loc[end_of_month2]) \
               * avg_monthly_volume / avg_twelve_months_volume

    def momentumNMo(self,df,date,column_name,number_of_months):
        """Avg of daily returns of last N months"""
        end_date = df.index.asof(pd.to_datetime(date))
        start_date = end_date - dt.timedelta(days=30*number_of_months)
        start_date = df.index.asof(start_date)
        cut_df = self.cut_dataframe(df,start_date=start_date,end_date=date)[column_name]
        cut_df_minus1 = self.cut_dataframe(df,start_date=start_date-BDay(1),end_date=end_date-BDay(1))[column_name]
        len_df = len(cut_df)
        len_df_minus1 = len(cut_df_minus1)
        len_min = min(len_df,len_df_minus1)
        daily_returns = cut_df.values[:len_min] / cut_df_minus1.values[:len_min]
        return daily_returns.mean()

    def meanReversion(self,df,date,column_name,n_days,N_days):
        """(Price Avg for n Days - Price Avg for N Days)/Price Avg for N Days"""
        day = df.index.asof(pd.to_datetime(date))
        n_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(n_days)]
        N_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(N_days)]
        avg_n_days = df.loc[n_day_range].dropna()[column_name].mean()
        avg_N_days = df.loc[N_day_range].dropna()[column_name].mean()
        return avg_n_days / avg_N_days - 1.0

    def highLowRange(self,df,date,column_name):
        """(Current price ‐ 52 week price low)/(52 Week High – 52 Week Low)"""
        day = df.index.asof(pd.to_datetime(date))
        fifty_two_day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(52*5)]
        fifty_two_week_low = df.loc[fifty_two_day_range].dropna()[LOW_STR].min()
        fifty_two_week_high = df.loc[fifty_two_day_range].dropna()[HIGH_STR].max()
        current_price = df.loc[day][column_name]
        return (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)

    def moneyFlow(self,df,date):
        """Money Flow = (((Close‐Low) ‐ (High‐Close)) / (High‐Low)) * Volume"""
        day = df.index.asof(pd.to_datetime(date))
        close = df.loc[day][CLOSE_STR]
        low = df.loc[day][LOW_STR]
        high = df.loc[day][HIGH_STR]
        volume = df.loc[day][VOLUME_STR]
        return (((close - low) - (high - close)) / (high - low)) * volume

    def moneyFlowPersistency(self,df,date,number_of_months):
        """No of days when Money Flow was positive in N months / Number of Days in N months"""
        day = df.index.asof(pd.to_datetime(date))
        day_range = [df.index.asof(day - dt.timedelta(i)) for i in range(number_of_months*30)]
        money_flows = np.array([self.moneyFlow(df,day1 - BDay(1)) for day1 in day_range])
        signs_of_money_flows = np.sign(money_flows)
        return (signs_of_money_flows[signs_of_money_flows>0]).sum() / (number_of_months*30)

    def slopeDaily(self,df,date,column_name):
        """Price slope of 10 Day Exponential Moving Average (EMA) over 5 days"""
        day = df.index.asof(pd.to_datetime(date))
        cut_df = self.cut_dataframe(df,end_date=day)
        ema = pd.ewma(cut_df[column_name], span=10).ix[-5:]
        ema_dates = ema.index.values
        ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
        slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
        return slope

    def slopeMonthly(self,df,date,column_name):
        """Price slope of 10 Month Exponential Moving Average (EMA) over 5 months"""
        day = df.index.asof(pd.to_datetime(date))
        cut_df = self.cut_dataframe(df,end_date=day)
        ema = pd.ewma(cut_df[column_name], span=300).ix[-150:]
        ema_dates = ema.index.values
        ema_time_delta_days = pd.Timedelta(ema_dates.max()-ema_dates.min()).total_seconds()/3600/24
        slope = (ema.values[-1] - ema.values[0]) / ema_time_delta_days
        return slope

    def pxRet(self,df,date,column_name, number_of_days):
        """Price return in percentage in N days"""
        day = df.index.asof(pd.to_datetime(date))
        day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
        cut_df = self.cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
        return cut_df.values[-1] / cut_df.values[0]

    def currPxRet(self,df,date,column_name):
        """(Current Price – Moving Avg of Last 3 yrs Price) / Current Price"""
        day = df.index.asof(pd.to_datetime(date))
        price = df.loc[day][column_name]
        day_start = df.index.asof(day - dt.timedelta(days=3*360))
        price_mean = self.cut_dataframe(df,start_date=day_start, end_date=day)[column_name].mean()
        return 1.0 - price_mean / price

    def nDayADR(self,df,date,column_name,number_of_days):
        """Avg of daily returns of last N days"""
        day = df.index.asof(pd.to_datetime(date))
        day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
        cut_df = self.cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
        cut_df_minus1 = self.cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
        return (cut_df.values / cut_df_minus1.values).mean()

    def nDayADP(self,df,date,column_name,number_of_days):
        """Avg of daily price change of last N days"""
        day = df.index.asof(pd.to_datetime(date))
        day_minus_n_days = df.index.asof(day - dt.timedelta(days=number_of_days))
        cut_df = self.cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
        cut_df_minus1 = self.cut_dataframe(df,end_date=day,start_date=day_minus_n_days)[column_name]
        return (cut_df.values - cut_df_minus1.values).mean()

    def pxRet2(self,df,date,column_name,N_days,n_days):
        """-50% of N days price return + 50% of n days price return"""
        return -0.5 * self.pxRet(df,date,column_name,N_days) + 0.5 * self.pxRet(df,date,column_name,n_days)

    def currPxRetSlope(self,df,date,column_name):
        """-50% of 3YrCurrPxRet + 50% of SlopeWeekly"""
        return -0.5 * self.currPxRet(df,date,column_name) + 0.5 * self.slopeWeekly(df,date,column_name)

    def allFactors(self,df_list,date,column_name):
        """All Factors as an array"""
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        f5 = []
        f6 = []
        f7 = []
        f8 = []
        f9 = []
        f10 = []
        f11 = []
        f12 = []
        f13 = []
        f14 = []
        f15 = []
        f16 = []
        f17 = []
        f18 = []
        f19 = []
        f20 = []
        f21 = []
        f22 = []
        f23 = []
        f24 = []
        f25 = []
        f26 =[]
        f27 = []
        f28 = []
        for df in df_list:
            if not df.empty:
                f1.append(self.slopeWeekly(df,date,column_name))
                f2.append(self.volumentumWeekly(df,date,column_name))
                f3.append(self.volumentumMonthly(df,date,column_name))
                f4.append(self.momentumNMo(df,date,column_name,3))
                f5.append(self.momentumNMo(df,date,column_name,6))
                f6.append(self.momentumNMo(df,date,column_name,9))
                f7.append(self.meanReversion(df,date,column_name,5,250))
                f8.append(self.meanReversion(df,date,column_name,5,500))
                f9.append(self.meanReversion(df,date,column_name,5,1000))
                f10.append(self.highLowRange(df,date,column_name)) 
                f11.append(self.moneyFlow(df,date))
                f12.append(self.moneyFlowPersistency(df,date,1))
                f13.append(self.moneyFlowPersistency(df,date,3))
                f14.append(self.moneyFlowPersistency(df,date,6))
                f15.append(self.slopeDaily(df,date,column_name))
                f16.append(self.slopeMonthly(df,date,column_name))
                f17.append(self.pxRet(df,date,column_name,360*3))
                f18.append(self.pxRet(df,date,column_name,30))
                f19.append(self.pxRet(df,date,column_name,60) )
                f20.append(self.pxRet(df,date,column_name,90))
                f21.append(self.currPxRet(df,date,column_name))
                f22.append(self.nDayADR(df,date,column_name,90))
                f23.append(self.nDayADP(df,date,column_name,60))
                f24.append(self.nDayADP(df,date,column_name,90))
                f25.append(self.pxRet2(df,date,column_name,360*3,30))
                f26.append(self.pxRet2(df,date,column_name,360*3,60))
                f27.append(self.pxRet2(df,date,column_name,360*3,90))
                f28.append(self.currPxRetSlope(df,date,column_name))
                
        factor_names = ['slopeWeekly',
                'volumentumWeekly',
                'volumentumMonthly',
                'momentumNMo(3)',
                'momentumNMo(6)',
                'momentumNMo(9)',
                'meanReversion(5,250)',
                'meanReversion(5,500)',
                'meanReversion(5,1000)',
                'highLowRange', 
                'moneyFlow',
                'moneyFlowPersistency(1)',
                'moneyFlowPersistency(3)',
                'moneyFlowPersistency(6)',
                'slopeDaily',
                'slopeMonthly',
                'pxRet(360*3))',
                'pxRet(30)',
                'pxRet(60)',
                'pxRet(90)',
                'currPxRet',
                'nDayADR(90)',
                'nDayADP(60)',
                'nDayADP(90)',
                'pxRet2(360*3,30)',
                'pxRet2(360*3,60)',
                'pxRet2(360*3,90)',
                'currPxRetSlope']        
                         
        return [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28], factor_names    

    

    
    def allFactors_small(self,df_list,date,column_name):
        """All Factors as an array"""
        f1 = []
        f18 = []
        for df in df_list:
                f1.append(self.slopeWeekly(df,date,column_name))
                f18.append(self.pxRet(df,date,column_name,30))
        return [f1,f18]

