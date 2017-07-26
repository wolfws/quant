import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import numba
import random
import pickle
import numpy as np
import pandas as pd
import itertools # https://docs.python.org/2/library/itertools.html # iterators for efficient looping
# https://docs.python.org/3/howto/functional.html#functional-howto-iterators
import warnings
warnings.filterwarnings("ignore")
from collections import namedtuple
from datetime import datetime, timedelta # http://strftime.org/
# import forward tester modules
from DataLoader import DataLoader as dl 
from HelperFunctions import HelperFunctions as hf 
from Config import MAXIMUM_NUMBER_OF_PERMUTATIONS, VERBOSE

# greeks https://quant.stackexchange.com/questions/4118/portfolio-greek-exposure-equations
# https://www.quantinsti.com/blog/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/

class ForwardTester(object):
    
    def __init__(self, ticker, optionStrategy, options_ts_id, p_contracts = 30, 
                 executionType = "mid", stockPerc = 0.3, minimumSamples = 30000, 
                 days_forward  = None, p_NearDiff = None, p_FarDiff = None, 
                 p_NearOTM = None, p_FarOTM = None, NearFarMaxSpread = None):
        
        self.ticker = ticker
        self.optionStrategy = optionStrategy # current options: rs = reverse diag calendar spread, ln = long diag calendar spread
        self.minimumSamples = minimumSamples 
        self.z = dl().getOptionsDataFromDb(options_ts_id)
        # TRADE SIMULATION PARAMETERS
        self.days_forward  = days_forward or [5,6,8,10,11]  # How many days to hold position until closing the trade
        self.OptStrat = {
            "rs": { "NearStrat":"B", # Buy   # ReverseDoubleDiagonalSpread  # shortCalendarSpread
                     "FarStrat":"S", # Sell 
                     "legs": 4
                    },
            "ln": { "NearStrat":"S",  # longCalendarSpread
                    "FarStrat":"B", 
                    "legs": 4
                },
        }
        
        self.legs =  self.OptStrat[self.optionStrategy]["legs"]

        self.p_NearDiff = p_NearDiff or [9, 20]
        self.NearDiff = [i for i in range(self.p_NearDiff[0],self.p_NearDiff[1])]  # 14,20  #10/20
        
        self.p_FarDiff = p_FarDiff or [16, 35]
        self.FarDiff = [i for i in range(self.p_FarDiff[0],self.p_FarDiff[1])] # 17,33    #17/33
        
        self.p_NearOTM = p_NearOTM or [0.01, 0.09, 0.03] # start, stop, step
        self.NearOTM = list(np.arange(self.p_NearOTM[0], self.p_NearOTM[1], self.p_NearOTM[2]))  # 0.03,0.07,0.03  #(0.0, 0.18, 0.03)
        
        self.p_FarOTM = p_FarOTM or [0.01, 0.15, 0.03] # start, stop, step
        self.FarOTM = list(np.arange(self.p_FarOTM[0], self.p_FarOTM[1], self.p_FarOTM[2]))  # 0.06, 0.10, 0.03   # (0.0, 0.18, 0.03)  
        
        ddiff = list(itertools.product(self.NearDiff,self.FarDiff))
        datediff = [x for x in ddiff if x[1] >= x[0]]
        otms = list(itertools.product(self.NearOTM,self.FarOTM))
        otms = [sorted(i) for i in otms]
        otms = sorted([list(i) for i in set(tuple(i) for i in sorted(otms))])
        otms = [i for i in otms if i[0] != i[1]]
        
        self.NearFarMaxSpread = NearFarMaxSpread or [0.029, 0.05] # Max spread between near and far strikes
        OTM = [x for x in otms if abs(x[1] - x[0]) >= self.NearFarMaxSpread[0] and abs(x[1] - x[0]) <= self.NearFarMaxSpread[1]]        
        self.variants = list(itertools.product(datediff,OTM,OTM,self.days_forward)) #Diagsprd selldate buydate, OTMSprds fr sells/buys
        self.samples = random.sample(range(0, len(self.variants)), min(self.minimumSamples,len(self.variants)))
        
        # HYPERPARAMETERS
        self.p_contracts = p_contracts 
        self.stockPerc = stockPerc  # Margin parameter: 0.3 for Interactive Brokers, 0.2 for OptionsHouse
        self.executionType = executionType   #asking #mid
        
        # new objects
        self.win_strategies = {}
        self.lose_strategies = {}
        self.err = {}
        self.errors = []
        
        #roundNum = lambda x: "%.2f" % x
        
        if VERBOSE:
            print("total strategies: {}".format(len(self.variants)))
            print("total strategies sampled in this forward test: {}".format(len(self.samples)))
            if self.optionStrategy == "rs":
                print("Need a full naked sell margin")
            elif self.optionStrategy == "ln":
                print("Margin equals to max loss")
            else:
                raise Exception("strategy type not implemented yet")

    # for each contract find days_forward px
    @numba.jit() # prod default: nopython=True # warn=False
    def getForwardValBase(self, sym, curdate, key, days_forward):
        if key == "fdate":
            return curdate + timedelta(days_forward)
        else:
            return self.z[(self.z["sym"]== sym) & (self.z["date"]== curdate + timedelta(days_forward))][key].ix[0]

    def getForwardVal(self, sym, curdate, key, days_forward):
        try:
            ret = self.getForwardValBase(sym,curdate,key,days_forward)
        except:
            ret = np.nan
        return ret

    @numba.jit
    def getMidPnl(self, a, b, c, d):
        return (a + b - c - d)/2

    @numba.jit
    def getBidAskPnl(self, a, b):
        return a - b

                      
    def runForwardTest(self):
        # 1. GROUP "C" trades by date
        # 2. GROUP "P" trades by date
        # FOR EACH [C,P]:
        # 	3. if more than 1 leg, chose 1 leg
        # 	4. if missing legs, then drop this trade, into error dict
        #		CONTINUE
        # 	5. get EOT (end of trade) pricing for each leg
        # 	6. calculate each leg P&L
        # 7. GROUP C & P by date, check a total of 4 legs (2P + 2C)
        # 8. CALCULATE TOTAL 4 LEG PNL
        #z = pickle.load( open("ts.p", "rb"))
        #optTypes = ['C','P']
        #for oType in optTypes:
    
        ctr = 0
        for i, variant in enumerate(self.variants):
            if i not in self.samples:
                continue
            if (ctr + 1) % 100 == 0:
                print("loop {}".format(ctr + 1))
            if True:
            #try:
                ND = variant[0][0] 
                FD = variant[0][1] 
                NO = sorted(variant[1]) 
                FO = sorted(variant[2])
                daysForward = variant[3]
                ctr +=1
                if ctr > MAXIMUM_NUMBER_OF_PERMUTATIONS: 
                    break

                x = self.z[self.z["dif"].isin([ND,FD])]
                if len(x) < 10: # skip as too few records
                    continue

                # Filter long and short strikes
                c = x[(x["s"].between(x["close"]*(1 + FO[0]),x["close"]*(1 + FO[1]))) & (x["type"] == 'C') 
                      & (x["dif"].isin([FD])) | 
                      (x["s"].between(x["close"]*(1 + NO[0]),x["close"]*(1 + NO[1]))) & (x["type"] == 'C') 
                      & (x["dif"].isin([ND]))]

                p = x[(x["s"].between(x["close"]*(1 - FO[1]),x["close"]*(1 - FO[0]))) 
                      & (x["type"] == 'P') & (x["dif"].isin([FD])) | 
                      (x["s"].between(x["close"]*(1 - NO[1]),x["close"]*(1- NO[0]))) 
                      & (x["type"] == 'P') & (x["dif"].isin([ND]))]

                # Map Strategies
                p["strat"] = np.where(p["dif"]==ND, self.OptStrat[self.optionStrategy]["NearStrat"], 
                                      np.where(p["dif"]==FD, self.OptStrat[self.optionStrategy]["FarStrat"], "N/A"))
                p["spread_perc"] = abs(p["ask"] - p["bid"])/np.mean([p["ask"],p["bid"]])

                c["strat"] = np.where(c["dif"]==ND, self.OptStrat[self.optionStrategy]["NearStrat"], 
                                      np.where(c["dif"]==FD, self.OptStrat[self.optionStrategy]["FarStrat"], "N/A"))
                c["spread_perc"] = abs(c["ask"] - c["bid"])/np.mean([c["ask"],c["bid"]])

                c1 = c.groupby(["date","strat"])["strat"].count()
                c1 = c1.unstack(1)
                if not ['B', 'S'] == sorted(list(c1.columns)):
                    continue
                c1 = c1[(c1["B"] >=1) & (c1["S"]>=1)]
                c_list = list(c1.index)
                c1 = c[c.index.isin(c_list)]
                cfilt = c1.groupby(["date","strat"])["otm"].max().reset_index()
                call_spread = pd.merge(c1,cfilt,on=["date","strat","otm"])
                if len(call_spread) < 10: # "not enough call spread data"
                    continue

                p1 = p.groupby(["date","strat"])["strat"].count()
                p1 = p1.unstack(1)
                if not ['B', 'S'] == sorted(list(p1.columns)):
                    continue
                p1 = p1[(p1["B"] >=1) & (p1["S"]>=1)]
                p_list = list(p1.index)
                p1 = p[p.index.isin(p_list)]
                pfilt = p1.groupby(["date","strat"])["otm"].max().reset_index()
                put_spread = pd.merge(p1,pfilt,on=["date","strat","otm"])
                if len(put_spread) < 10: # "not enough put spread data"
                    continue

                # Strategy forward test  
                call_spread.dropna(axis=0, how='any')
                call_spread['f_close']=call_spread.apply(lambda x:self.getForwardVal(x['sym'],x['date'],'close',daysForward),axis = 1)
                call_spread['f_ask'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'ask',daysForward), axis = 1)
                call_spread['f_bid'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'bid',daysForward), axis = 1)
                call_spread['f_iv'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'iv',daysForward), axis = 1)
                call_spread['f_vlm'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'vlm',daysForward), axis = 1)
                call_spread['f_date'] =call_spread.apply(lambda x:self.getForwardVal(x['sym'],x['date'],'fdate',daysForward),axis = 1)
                call_spread['f_d'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'d',daysForward), axis = 1)
                call_spread['f_v'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'v',daysForward), axis = 1)
                call_spread['f_g'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'g',daysForward), axis = 1)
                call_spread['f_th'] = call_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'th',daysForward), axis = 1)
                call_spread['f_meanpx']=call_spread.apply(lambda x:self.getForwardVal(x['sym'],x['date'],'meanpx',daysForward),axis=1)
                call_spread.dropna(axis=0, how='any')

                if self.executionType == "asking":
                    call_spread['pnl'] = np.where(call_spread['strat'] == "B", \
                                    call_spread.apply(lambda x: self.getBidAskPnl(x['f_bid'],x['ask']), axis = 1), \
                                    call_spread.apply(lambda x: self.getBidAskPnl(x['bid'],x['f_ask']), axis = 1) \
                                                  )    
                if self.executionType == "mid":
                    call_spread['pnl'] = np.where(call_spread['strat'] == "B", \
                                    call_spread.apply(lambda x: self.getMidPnl(x['f_bid'],x['f_ask'],x['ask'],x['bid']), axis = 1), \
                                    call_spread.apply(lambda x: self.getMidPnl(x['bid'],x['ask'],x['f_ask'],x['f_bid']), axis = 1) \
                                                  )

                put_spread.dropna(axis=0, how='any')
                put_spread['f_close']=put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'close',daysForward), axis = 1)
                put_spread['f_ask'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'ask',daysForward), axis = 1)
                put_spread['f_bid'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'bid',daysForward), axis = 1)
                put_spread['f_iv'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'iv',daysForward), axis = 1)
                put_spread['f_vlm'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'vlm',daysForward), axis = 1)
                put_spread['f_date'] =put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'fdate',daysForward), axis = 1)
                put_spread['f_d'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'d',daysForward), axis = 1)
                put_spread['f_v'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'v',daysForward), axis = 1)
                put_spread['f_g'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'g',daysForward), axis = 1)
                put_spread['f_th'] = put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'th',daysForward), axis = 1)
                put_spread['f_meanpx']=put_spread.apply(lambda x: self.getForwardVal(x['sym'],x['date'],'meanpx',daysForward),axis= 1)
                put_spread.dropna(axis=0, how='any')

                if self.executionType == "asking":
                    put_spread['pnl'] = np.where(put_spread['strat'] == "B", \
                                    put_spread.apply(lambda x: self.getBidAskPnl(x['f_bid'],x['ask']), axis = 1), \
                                    put_spread.apply(lambda x: self.getBidAskPnl(x['bid'],x['f_ask']), axis = 1) \
                                    )

                if self.executionType == "mid":
                    put_spread['pnl'] = np.where(put_spread['strat'] == "B", \
                                    put_spread.apply(lambda x: self.getMidPnl(x['f_bid'],x['f_ask'],x['ask'],x['bid']), axis = 1), \
                                    put_spread.apply(lambda x: self.getMidPnl(x['bid'],x['ask'],x['f_ask'],x['f_bid']), axis = 1) \
                                                  )
                # PNL & Risk Statistics
                spreads = pd.concat([call_spread,put_spread])
                spreads.sort_values(by=["date"],inplace=True)
                pnl = spreads.groupby(["date","close","f_close","f_date"])["pnl"].sum().reset_index()
                pnl = pd.DataFrame(pnl)
                pnl["contracts"] = self.p_contracts
                pnl["total_pnl"] = self.p_contracts * 100 * pnl["pnl"]
                pnl["cum_sum"] = pnl["total_pnl"].cumsum()

                if len(pnl) < 1:
                    continue

                Trades = len(pnl)  
                WinTrades = len(pnl[pnl["total_pnl"]>0])
                PctWinTrades = np.around(100*WinTrades/Trades,0)
                USDWinTrades = np.around(np.sum(pnl[pnl["total_pnl"]>0]['total_pnl']),0)
                AvgWinTrade = np.around(USDWinTrades / WinTrades)

                LoseTrades = len(pnl[pnl["total_pnl"]<0])
                PctLoseTrades = np.around(100*LoseTrades/Trades,0)
                USDLoseTrades = np.around(np.sum(pnl[pnl["total_pnl"]<0]['total_pnl']),0)
                AvgLoseTrade = np.around(USDLoseTrades / LoseTrades)
                StartAbsMV = self.p_contracts * 100 * spreads.iloc[0]["close"]
                pnl = pnl.dropna(axis=0, how='any')
 
                if USDWinTrades + USDLoseTrades < 0:
                    if VERBOSE:
                        print("LOSE loop {}".format(ctr + 1))
                        print("LOSE PNL: {}".format(USDWinTrades + USDLoseTrades))
                        print("ND,FD,NO,FO,daysForward",ND,FD,NO,FO,daysForward)
                        print(pnl.to_string(formatters={'*** PROFIT *** total_pnl':'${:,.2f}'.format,'cum_sum':'${:,.2f}'.format}))
                    self.lose_strategies[ctr] = {}
                    self.lose_strategies[ctr].update({"pnl table": pnl,
                              "Number of Trades": Trades,
                              "Win Trades": (WinTrades,PctWinTrades,USDWinTrades,AvgWinTrade),
                              "Lose Trades": (LoseTrades,PctLoseTrades,USDLoseTrades,AvgLoseTrade),
                              "StartAbsMV":StartAbsMV,
                              "Total PNL": USDWinTrades + USDLoseTrades,
                              "Total days":str(pnl.index[-1] - pnl.index[0]),
                              "Begin": pnl.index[0],
                              "End": pnl.index[-1],
                              "Params: ND,FD,NO,FO": (ND,FD,NO,FO),
                              "Trades": spreads,
                              "daysForward": daysForward,
                              "total_strategies": len(self.variants),
                              "total_sampled_strategies": len(self.samples),
                              "margin_by_date": hf().getSpreadMargin(self.p_contracts, self.stockPerc, self.optionStrategy, spreads)
                             })

                if USDWinTrades + USDLoseTrades > 0:
                    if VERBOSE:
                        print("WIN loop {}".format(ctr + 1))
                        print("ND,FD,NO,FO,daysForward",ND,FD,NO,FO,daysForward)
                        print("found records for {} {}: ".format(ND,FD), len(x) )
                        print(pnl.to_string(formatters={'*** PROFIT *** total_pnl':'${:,.2f}'.format,'cum_sum':'${:,.2f}'.format}))
                      
                    self.win_strategies[ctr] = {}   
                    self.win_strategies[ctr].update({"pnl table": pnl,
                                          "Number of Trades": Trades,
                                          "Win Trades": (WinTrades,PctWinTrades,USDWinTrades,AvgWinTrade),
                                          "Lose Trades": (LoseTrades,PctLoseTrades,USDLoseTrades,AvgLoseTrade),
                                          "StartAbsMV":StartAbsMV,
                                          "Total PNL": USDWinTrades + USDLoseTrades,
                                          "Total Days":str(pnl.index[-1] - pnl.index[0]),
                                          "Begin": pnl.index[0],
                                          "End": pnl.index[-1],
                                          "Params: ND,FD,NO,FO": (ND,FD,NO,FO),
                                          "Trades": spreads,
                                          "daysForward": daysForward,
                                          "total_strategies": len(self.variants),
                                          "total_sampled_strategies": len(self.samples),
                                          "margin_by_date": hf().getSpreadMargin(self.p_contracts, 
                                                                                 self.stockPerc, 
                                                                                 self.optionStrategy, 
                                                                                 spreads)
                                         })
            #except Exception as er:
            #    print(er)
            #    self.errors.append([ctr,er])
            #    pass

        # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # pickle.dump(win_strategies, open("experiments/{}_win_strategies_{}.p".format(ticker,timestamp), "wb"))
        # pickle.dump(lose_strategies, open("experiments/{}_lose_strategies_{}.p".format(ticker,timestamp), "wb"))
                      
        # record errors to log, if any
        if len(self.errors) > 0:
            dl().saveLogToDb(ticker, self.errors, None, "insert", True)

        # insert forward rest results into the database
        if len(self.win_strategies) > 0:
            winpnl = hf().printStrat(self.win_strategies, "object")
            # top win pnl stats
            top_winpnl = winpnl[:1].iloc[0]
            p_pnl = top_winpnl['pnl']
            p_tr = top_winpnl['tr']
            p_nd = top_winpnl['ND']
            p_fd = top_winpnl['FD']
            p_no = top_winpnl['NO']
            p_fo = top_winpnl['FO']
            p_daysforward = top_winpnl['daysForward']
            p_wpc = top_winpnl['w_pc']
            winpnl = pickle.dumps(winpnl)
        else:
            p_pnl = 0
            p_tr = 0
            p_nd = None
            p_fd = None
            p_no = None
            p_fo = None
            p_daysforward = None
            p_wpc = None
            winpnl = pickle.dumps(None)
        
        if len(self.lose_strategies) > 0:
            losepnl = hf().printStrat(self.lose_strategies, "object")
            # top win pnl stats
            top_losepnl = losepnl[-1:].iloc[0]
            lose_pnl = top_losepnl['pnl']
            lose_tr = top_losepnl['tr']
            losepnl = pickle.dumps(losepnl)  
        else:
            lose_pnl = 0
            lose_tr = 0
            losepnl = pickle.dumps(None)
        
        #winobj = pickle.dumps(self.win_strategies) # Optional data load in the dictionary / series format
        #loseobj = pickle.dumps(self.lose_strategies) # Optional data load in the dictionary / series format
                
        #if VERBOSE:
            #print(self.ticker, self.optionStrategy, self.legs, winpnl, losepnl, self.days_forward)
            #print([self.p_NearDiff[0], self.p_NearDiff[1]],[self.p_FarDiff[0], self.p_FarDiff[1]])
            #print([self.p_NearOTM[0],self.p_NearOTM[1],self.p_NearOTM[2]])
            #print([self.p_FarOTM[0], self.p_FarOTM[1],self.p_FarOTM[2]])
            #print(p_pnl, p_tr, p_nd, p_fd, p_no, p_fo, p_daysforward, p_wpc)

        _id = dl().saveExperimentToDb(self.ticker, self.optionStrategy, self.legs, winpnl, losepnl,
                                      self.days_forward, 
                                      [self.p_NearDiff[0], self.p_NearDiff[1]],
                                      [self.p_FarDiff[0], self.p_FarDiff[1]],
                                      [self.p_NearOTM[0],self.p_NearOTM[1],self.p_NearOTM[2]], 
                                      [self.p_FarOTM[0], self.p_FarOTM[1],self.p_FarOTM[2]],
                                      p_pnl, p_tr, p_nd, p_fd, p_no, p_fo, p_daysforward, p_wpc,
                                      lose_pnl, lose_tr, self.minimumSamples,
                                      len(self.z),
                                       self.z.iloc[0]['date'],
                                      self.z.iloc[-1]['date'],
                                      (self.z.iloc[-1]['date'] - self.z.iloc[0]['date']).days,
                                      self.p_contracts
                               )
                  
        print("done with forward test")
        return _id