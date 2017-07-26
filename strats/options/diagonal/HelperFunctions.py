import pandas as pd
import numpy as np
# libraries to load underlying daily time series
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import fix_yahoo_finance as yf #https://github.com/ranaroussi/fix-yahoo-finance
    yf.pdr_override() # fix


# TIPS: get cell value: #data.loc[data.index == '2017-06-29'].iloc[0]['Volatility']

class HelperFunctions(object):
    
    def cleanParams(self, a):
        ND = a[0]
        FD = a[1]
        NO = [str(np.round(i,2)) for i in a[2]]
        FO = [str(np.round(i,2)) for i in a[3]]
        return [ND,FD,NO,FO]

    def printStrat(self, strategies, mode="print"):
        simid = []; sim_pnl = [];wins = [];losses = [];trades = [];params = []; daysForward = []
        for i in strategies.keys():
            simid.append(i)
            sim_pnl.append(strategies[i].get("Total PNL"))
            wins.append(strategies[i].get("Win Trades"))
            losses.append(strategies[i].get("Lose Trades"))
            trades.append(strategies[i].get("Number of Trades"))
            params.append(strategies[i].get("Params: ND,FD,NO,FO"))
            daysForward.append(strategies[i].get("daysForward"))

        df_pnl = pd.DataFrame({'id':simid,'pnl':sim_pnl, 'wins':wins,'losses':losses,'tr':trades,
                               'params [ND,FD,NO,FO]':params,"daysForward": daysForward})

        df_pnl["params [ND,FD,NO,FO]"] = df_pnl.apply(lambda x: self.cleanParams(x["params [ND,FD,NO,FO]"]), axis=1)


        df_pnl["ND"] = df_pnl.apply(lambda x: x["params [ND,FD,NO,FO]"][0], axis=1)
        df_pnl["FD"] = df_pnl.apply(lambda x: x["params [ND,FD,NO,FO]"][1], axis=1)
        df_pnl["NO"] = df_pnl.apply(lambda x: x["params [ND,FD,NO,FO]"][2], axis=1)
        df_pnl["FO"] = df_pnl.apply(lambda x: x["params [ND,FD,NO,FO]"][3], axis=1)
        del df_pnl["params [ND,FD,NO,FO]"] 

        df_pnl["w_tr"] = df_pnl.apply(lambda x: x["wins"][0], axis=1)
        df_pnl["w_pc"] = df_pnl.apply(lambda x: x["wins"][1], axis=1)
        df_pnl["w_amt"] = df_pnl.apply(lambda x: x["wins"][2], axis=1)
        df_pnl["w_avg"] = df_pnl.apply(lambda x: x["wins"][3], axis=1)

        df_pnl["l_tr"] = df_pnl.apply(lambda x: x["losses"][0], axis=1)
        df_pnl["l_pc"] = df_pnl.apply(lambda x: x["losses"][1], axis=1)
        df_pnl["l_amt"] = df_pnl.apply(lambda x: x["losses"][2], axis=1)
        df_pnl["l_avg"] = df_pnl.apply(lambda x: x["losses"][3], axis=1)

        df_pnl = df_pnl[['id','pnl','tr','ND','FD','NO','FO','daysForward',
                         'w_tr','w_pc','w_amt','w_avg','l_tr','l_pc','l_amt','l_avg']]

        df_pnl.sort_values(by='pnl', axis=0, ascending=False, inplace=True)
        pd.options.display.max_columns = 30
        if mode == "print":
            print(df_pnl.to_string( index_names=False, col_space=0,justify='left',
                               formatters={'w_pc':'{:,.0f}'.format,'w_amt':'${:,.0f}'.format, 'w_avg':'${:,.0f}'.format,
                                          'l_pc':'{:,.0f}'.format,'l_amt':'${:,.0f}'.format, 'l_avg':'${:,.0f}'.format,
                                          'pnl':'${:,.0f}'.format}))
        elif mode == "object":
            return df_pnl
        else:
            raise Exception("unknown mode")

    # USAGE
    # printStrat(win_strategies, "print") 
    # printStrat(win_strategies, "object").to_csv("c:/data/winners717_3.csv") 
    # printStrat(lose_strategies, "object").to_csv("c:/data/losers717_3.csv") 

    # Analyze select basket pnl
    def getVol(self, data, dt, duration):
        if duration == 30:
            v = data.loc[data.index == dt].iloc[0]['Vol30']
        elif duration == 10:
            v = data.loc[data.index == dt].iloc[0]['Vol10']
        elif duration == 5:
            v = data.loc[data.index == dt].iloc[0]['Vol5']
        else:
            raise Exception("wrong type")
        return v

    def addVol2PnLTable(self, pnltable):
        b = pnltable
        b["px_chg"] = b.apply(lambda x: (x['f_close'] - x['close'])/x['close'], axis = 1)
        # Get underlying time series
        data = pdr.get_data_yahoo(ticker, start="2015-11-01", end="2017-07-14")
        data['Log_Ret'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        # Compute Volatility using the pandas rolling standard deviation function
        data['Vol30'] = pd.rolling_std(data['Log_Ret'], window=30) * np.sqrt(30) # 30 day volatility of returns
        data['Vol10'] = pd.rolling_std(data['Log_Ret'], window=10) * np.sqrt(10) # 10 day volatility of returns
        data['Vol5'] = pd.rolling_std(data['Log_Ret'], window=5) * np.sqrt(5) # 5 day volatility of returns
        b['vol30'] = b.apply(lambda x: getVol(data, x['date'],30), axis = 1)
        b['vol10'] = b.apply(lambda x: getVol(data, x['date'],10), axis = 1)
        b['vol5'] = b.apply(lambda x: getVol(data, x['date'],5), axis = 1)
        b.sort_values(by=["total_pnl"],ascending=False, inplace=True)
        return b

    # USAGE:
        #pnltbl = win_strategies[2]['pnl table'] 
        #addVol2PnLTable(pnltbl)

        # https://www.interactivebrokers.com/en/?f=marginnew&p=opt1
        # https://www.optionshouse.com/margins-buying-power/margin-requirements/short-uncovered-call-equity-options/

     
    def getMarginReqs_shortCalendarSpread(self, optType, stockPX, contracts, optPX, strike, stockPerc):
        # only select back month strikes and prices here..
        # stockPerc = 20% for OptH, 30% for IB
        # this is a naked sell as longer-term option is uncovered
        if optType == "call":
            case1Amount = (stockPX*stockPerc - abs(stockPX - strike) + optPX)*contracts*100
            case2Amount = (stockPX*0.1 + optPX)*contracts*100
        elif optType == "put":
            case1Amount = (stockPX*stockPerc - abs(stockPX - strike) + optPX)*contracts*100
            case2Amount = (strike*0.1 + optPX)*contracts*100

        return max(case1Amount, case2Amount)
    
    def getMarginReqs_longCalendarSpread(self, optType, stockPX, contracts, frontMonthPX, frontMonthStrike, backMonthPX, backMonthStrike):
        # the margin would be (assuming both strikes are out of the money):
        # call margin = backMonthStrike - frontMonthStrike + frontMonthPX - backMonthStrike
        # put margin =  frontMonthStrike - backMonthStrike + frontMonthPX - backMonthStrike
        # simplified spread margin = max(call margin, put margin)
        if optType == "call":
            margin = backMonthStrike - frontMonthStrike + frontMonthPX - backMonthPX
        elif optType == "put":
            margin = frontMonthStrike - backMonthStrike + frontMonthPX - backMonthPX
        
        return margin*contracts*100

    
    
    class ConvertExceptions(object):
        # Defining a custom __call__() method in the meta-class allows 
        # the class's instance to be called as a function, not always modifying the instance itself.
        # https://stackoverflow.com/questions/9663562/what-is-difference-between-init-and-call-in-python
        # class A(): with def __init__(self): print("init") and def __call__(self): print("call")
        # a = A()  # ==> init
        # a() # ==> call

        func = None

        def __init__(self, exceptions, replacement=None):
            self.exceptions = exceptions
            self.replacement = replacement

        def __call__(self, *args, **kwargs):
            if self.func is None:
                self.func = args[0]
                return self
            try:
                return self.func(*args, **kwargs)
            except self.exceptions:
                return self.replacement       

            
    @ConvertExceptions(Exception, "getSpreadMargin error")
    def getSpreadMargin(self, p_contracts, stockPerc, optionStrategy, strategiesDs):
        # p_contracts = 30, stockPerc = 0.3, optionStrategy = "ln" # rs/ln
        l = len(strategiesDs)
        legs = 4
        trades_inside = int(l) / legs
        marginByDate = {}
        for i in range(0, int(trades_inside),1):
            begin = i*4 
            end = i*4 + 4
            subds = strategiesDs[begin:end]
            subds.sort_values("exp", inplace=True)
            if optionStrategy == "rs":
                # only get back month options data  => get value ["date"].iloc[0]
                subds = subds[-2:]
                call_ds = subds[subds['type']=="C"]
                put_ds = subds[subds['type']=="P"]
                call_margin = self.getMarginReqs_shortCalendarSpread("call", # optType, 
                                                                     call_ds["close"].iloc[0], # stockPX, 
                                                                     p_contracts, #contracts, 
                                                                     call_ds["meanpx"].iloc[0], # optPX, 
                                                                     call_ds["s"].iloc[0], # strike, 
                                                                     stockPerc
                                                                    )

                put_margin = self.getMarginReqs_shortCalendarSpread("put", #optType,
                                                                    put_ds["close"].iloc[0], # stockPX, 
                                                                    p_contracts, # contracts, 
                                                                    put_ds["meanpx"].iloc[0], # optPX, 
                                                                    put_ds["s"].iloc[0], # strike, 
                                                                    stockPerc
                                                                   )

                total_margin = call_margin + put_margin
                #print("total margin", subds[:1]["date"].iloc[0], total_margin)
                marginByDate[subds[:1]["date"].iloc[0]] = total_margin

            elif optionStrategy == "ln":
                call_ds = subds[subds['type']=="C"]
                put_ds = subds[subds['type']=="P"]

                call_margin = self.getMarginReqs_longCalendarSpread("call", # optType,
                                                                    call_ds["close"].iloc[0], #stockPX,  
                                                                    p_contracts, # contracts, 
                                                                    call_ds["meanpx"].iloc[0], #frontMonthPX, 
                                                                    call_ds["s"].iloc[0], # frontMonthStrike, 
                                                                    call_ds["meanpx"].iloc[1], #backMonthPX, 
                                                                    call_ds["s"].iloc[1], # backMonthStrike
                                                                   )

                put_margin = self.getMarginReqs_longCalendarSpread("put", #optType,
                                                                   put_ds["close"].iloc[0], #stockPX, 
                                                                   p_contracts, # contracts, 
                                                                   put_ds["meanpx"].iloc[0], #frontMonthPX, 
                                                                   put_ds["s"].iloc[0], # frontMonthStrike, 
                                                                   put_ds["meanpx"].iloc[1], #backMonthPX, 
                                                                   put_ds["s"].iloc[1], # backMonthStrike
                                                                  )

                total_margin = max(call_margin,put_margin)
                #print("total margin 2", subds[:1]["date"].iloc[0], total_margin)
                marginByDate[subds[:1]["date"].iloc[0]] = total_margin 
                        
        return marginByDate
     
        
        
 
        
        
    
    
    
