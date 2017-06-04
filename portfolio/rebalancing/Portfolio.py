from cvxpy import *
import cvxpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahoo_finance import Share
np.set_printoptions(suppress=True)

class Portfolio(object):

    def calculatePnLStats(self, hist):
        hist = pd.DataFrame(hist)
        hist = hist.transpose()
        hist.columns = ["model","alloc","mv","pnl","pnl_perc","tickers","px"] 
        print (hist)
        hist.sort_index(inplace=True)
        portfolio_pnl = hist.tail(1)['mv'][0] - hist.head(1)['mv'][0]
        if hist.head(1)['mv'][0] != 0:
            portfolio_pnl_chg = round(float((hist.tail(1)['mv'][0] - hist.head(1)['mv'][0])/hist.head(1)['mv'][0]),4)
            print ("Shared.portfolio_pnl_chg: {}".format(portfolio_pnl_chg))
        else:
            print("mv error as base period mv is zero")
        
        print ("Shared.portfolio_pnl: {}".format(portfolio_pnl))
        
        return True

 
    def calculate_portfolio_weights(self,cvxtype, returns_function, long_only, exp_return, 
                            selected_solver, max_pos_size):
        assert cvxtype in ['minimize_risk','maximize_return']
        # mu is the vector of expected returns.
        # sigma is the covariance matrix.
        # gamma is a Parameter that trades off risk and return.
        # x is a vector of stock holdings as fractions of total assets.
        gamma = Parameter(sign="positive")
        gamma.value = 1
        returns, stocks, betas = returns_function
        print ("returns {}".format(str(returns.shape)))
        #print "betas {}".format(betas)

        cov_mat = returns.cov() # covariance matrix of returns
        Sigma = cov_mat.values # np.asarray(cov_mat.values) 
        w = Variable(len(cov_mat))  # #number of stocks for portfolio weights
        print ("w: " + str(w))
        risk = quad_form(w, Sigma)  #expected_variance => w.T*C*w =  quad_form(w, C)
        num_stocks = len(cov_mat)

        if cvxtype == 'minimize_risk': # Minimize portfolio risk / portfolio variance
            if long_only == True:
                prob = Problem(Minimize(risk), [sum_entries(w) == 1, w > 0, abs(w) <= max_pos_size ])  # Long only
            else:
                prob = Problem(Minimize(risk), [sum_entries(w) == 1, abs(w) <= max_pos_size]) # Long / short 

        elif cvxtype == 'maximize_return': # Maximize portfolio return given required level of risk
            mu = np.array([exp_return]*len(cov_mat)) # mu is the vector of expected returns.
            expected_return = np.reshape(mu,(-1,1)).T * w  # w is a vector of stock holdings as fractions of total assets.   
            objective = Maximize(expected_return - gamma*risk) # Maximize(expected_return - expected_variance)
            if long_only == True:
                constraints = [sum_entries(w) == 1, w > 0]
            else: 
                constraints=[sum_entries(w) == 1]
                for i in range(len(cov_mat)):
                    constraints.extend([ w[i] < max_pos_size, w[i] > -max_pos_size])
            prob = Problem(objective, constraints)

        prob.solve(solver=selected_solver)

        weights = []
        for weight in w.value:
            weights.append(float(weight[0]))

        if cvxtype == 'maximize_return':
            print("# The optimal expected return.")
            print(expected_return.value)

        print("# The optimal risk.")
        print(risk.value*100, " %")

        return weights


    def getReturns(self, stocks = 'MSFT,AAPL,NFLX,JPM,UVXY,RSX,TBT', period_days = 100, end = '2016-12-09' ):
        if not isinstance(stocks,list):
            stocks = stocks.split(",")
        index = 'SPY'
        stocks.append(index)
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(period_days)).strftime("%Y-%m-%d")
        else:
            start = (datetime.strptime(end, '%Y-%m-%d') - timedelta(period_days)).strftime("%Y-%m-%d")
        i = 0
        w= pd.DataFrame()
        t = []
        for s in stocks:
            z = Share(s)
            #print ("start {}".format(start))
            #print ("end {}".format(end)) 
            px = pd.DataFrame(z.get_historical(start,end))[['Close','Date']]
            px['Close']=px['Close'].astype(float)
            px.index = px['Date']
            del px['Date']
            px.columns = [s]
            t.append(px)
        w = pd.concat(t,axis=1, join='inner')
        w = w.sort_index().pct_change()  
        betas = [] #calculate betas
        for s in stocks:
            if s != index:
                col = np.column_stack((w[s],w[index]))
                b = np.cov(col)/np.var(w[index])
                betas.append(b)  
        stocks.remove(index)
        del w[index]
        returns = w
        return returns,stocks,np.round(betas,4)

    def getHistoricalPrice(self, ticker, date):
        yahoo = Share(ticker)
        #print ("getHistoricalPrice: {} {}".format(ticker, date))
        px = yahoo.get_historical(date, date)
        return px[0].get('Adj_Close')
