import FinancialData
import TimeSeriesFunctions
import pandas as pd

QUANDL_TOKEN = ""
BENCHMARK_INDEX = "SPY"

# returns_snp500 = BenchmarkReturn().getBenchmarkReturn(years,column_name,dates_insample,dates_outsample)
class BenchmarkReturn(object):
    
    def getBenchmarkReturn(self,years,column_name,dates_insample,dates_outsample):
        snp500_df = FinancialData.FinancialData.getSingleTS(QUANDL_TOKEN,BENCHMARK_INDEX,years[0],years[-1])
        
        returns_snp500, failedtickers = TimeSeriesFunctions.TimeSeriesFunctions.returnsMonthly(
            {BENCHMARK_INDEX:snp500_df},
            column_name,
            dates_insample,
            dates_outsample)
        returns_snp500 = returns_snp500[BENCHMARK_INDEX]
        monthly_returns = returns_snp500.groupby(returns_snp500.index).first().dropna()
        annual_returns = monthly_returns.groupby(pd.TimeGrouper("A")).sum().dropna()
               
        return monthly_returns,annual_returns
