import datetime
import numpy as np
import pandas as pd
import pickle
from datetime import datetime,  timedelta
from bson.objectid import ObjectId
from pymongo import MongoClient
import gridfs
from Config import MONGOHOST, MONGODATABASE, UNDERLYING_TS_DAYS_BACK
from functools import wraps
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import fix_yahoo_finance as yf #https://github.com/ranaroussi/fix-yahoo-finance
    yf.pdr_override() # fix

#from bson.binary import Binary
# LOAD DATA
#serializeObject = True
#filePath = "c://data//svxy//data_download.csv"
# pickle.dump(z, open(outputFile, "wb"))
# z = pickle.load(open(outputFile, "rb"))
# z = pd.read_pickle("uvxy_prepped.pkl") # newer pandas is then able to read old pickles...
# insert into log: _id = dl().saveLogToDb("AAPL-TEST", "some error msg", None, "insert", True)
# update log: dd = dl().saveLogToDb(None,None, fsid = _id, command = "update", iserror =  True)

class DataLoader(object):
    
    def __init__(self):
        self.client = MongoClient(MONGOHOST, 27017)
        self.db = self.client[MONGODATABASE]
            
    def logDecorator(message):
        def real_decorator(function):
            @wraps(function)
            def wrapper(self, *args, **kwargs):
                ticker = args[0]
                msg = args[1]
                _id = self.saveLogToDb(ticker, msg = message, fsid = None, command = "insert", iserror =  False) 
                function(self, *args, **kwargs)
                self.saveLogToDb(ticker, msg = message, fsid = _id, command = "update", iserror =  False) 
            return wrapper
        return real_decorator
        
    def saveLogToDb(self, ticker, msg = None, fsid = None, command = "insert", iserror =  False):
        collection = self.db['log']  
        if command == "insert":
            record = { 
                "date" : datetime.utcnow(), 
                "ticker" : ticker, 
                "msg": msg,
                "end": "",
                "iserror": iserror
            }    
            _id = collection.insert_one(record).inserted_id
            return _id
            
        elif command == "update": 
            collection.update_one(
                {"_id":ObjectId(fsid)},
                { "$set": {
                "end": datetime.utcnow()
                } }
                )
            return fsid
                

    @logDecorator("added underlying ts")
    def importUnderlyingTsToDb(self, ticker, src = "yahoo finance"):
        enddate = datetime.now().strftime("%Y-%m-%d")
        startdate = (datetime.now() - timedelta(days=UNDERLYING_TS_DAYS_BACK)).strftime("%Y-%m-%d")
        data = pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
        data['Log_Ret'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        temp_obj = pickle.dumps(data)
        fs = gridfs.GridFS(self.db)
        fsid = fs.put(temp_obj, ticker=ticker, date=datetime.utcnow())
        collection = self.db['ts_underlying']  
        record = { 
                "date" : datetime.utcnow(), 
                "ticker" : ticker, 
                "fsid" : fsid,
                "rows": len(data),
                "startdate": data.index[0].strftime("%Y-%m-%d"),
                "enddate": data.index[-1].strftime("%Y-%m-%d"),
                "days": (data.index[-1] - data.index[0]).days,
                "src": src
                }    
        _id = collection.insert_one(record).inserted_id
        
        return _id
            
    @logDecorator("added option ts")
    def saveOptionsDataToDb(self, ticker, filePath):
        # load and prepare the data
        x = pd.read_csv(filePath)
        x.columns = ['symbol', 'ex', 'date', 'adjclose','sym', 'exp', 's', 'type', 
                     'style', 'ask','bid', 'meanpx', 'iv', 'vlm', 'opint',
                     'close', 'inf', 'd', 'v', 'g', 'th', 'r']
        z = x[ ["date", "sym","exp", "close","s","type","ask","bid","meanpx","iv","vlm", "d","v","g","th"] ] .copy()
        z['date'] = pd.to_datetime(z['date'])
        z['exp'] = pd.to_datetime(z['exp'])
        z['dif'] = (z['date'] - z['exp']) / np.timedelta64(-1, 'D')
        z.index = z["date"]
        z["otm"] = round(((z["s"] - z["close"])/z["close"])*100,2)
        z.sort_index(axis=0,inplace=True)
        # serialize the pandas object
        #temp_obj = Binary(pickle.dumps(z, protocol=2), subtype=128 )
        temp_obj = pickle.dumps(z)
        # save options time series to GridFS, get id and add to the collection of available tickers
        fs = gridfs.GridFS(self.db)
        fsid = fs.put(temp_obj, ticker=ticker, filePath=filePath, date=datetime.utcnow())
        collection = self.db['ts_options']  
        record = { 
                "date" : datetime.utcnow(), 
                "ticker" : ticker, 
                "fsid" : fsid,
                "rows": len(z),
                "startdate": z.iloc[0]['date'],
                "enddate": z.iloc[-1]['date'],
                "days": (z.iloc[-1]['date'] - z.iloc[0]['date']).days
                }    
        _id = collection.insert_one(record).inserted_id
        
        return _id

        
    def getOptionsDataFromDb(self, _id): 
        # unpickle options ts as a pandas tbl
        fs = gridfs.GridFS(self.db)
        ds_pickle = fs.get(ObjectId(_id)).read()
        ds = pickle.loads(ds_pickle)
        
        return ds
    
    
    def saveExperimentToDb(self, ticker, optionStrategy, legs, winpnl, losepnl, 
                            days_forward, NearDiff, FarDiff, NearOTM, FarOTM, 
                           p_pnl, p_tr, p_nd, p_fd, p_no, p_fo, p_daysforward, p_wpc, lose_pnl, lose_tr,
                          minimumSamples, ts_rows, ts_startdate, ts_enddate, ts_days, p_contracts):

        # To store a binary object in a collections, one needs to create a binary field
        # https://stackoverflow.com/questions/18089598/
        #is-there-a-way-to-store-python-objects-directly-in-mongodb-without-serializing-t
        # from bson.binary import Binary; import pickle
        # thebytes = pickle.dumps(myObj)
        # coll.insert({'bin-data': Binary(thebytes)})
        
        collection = self.db['experiments']  
        # save experiment pickles to GridFS first and get corresponding ids
        fs = gridfs.GridFS(self.db)
        winpnlid = str(fs.put(winpnl))
        losepnlid = str(fs.put(losepnl))
        
        #print(ticker, optionStrategy, legs, days_forward)
        #print(NearDiff, FarDiff, NearOTM, FarOTM)
        #print(p_pnl, p_tr, p_nd, p_fd, p_no, p_fo, p_daysforward, p_wpc, lose_pnl, lose_tr)
        
        record = { 
                "date" : datetime.utcnow(), 
                "ticker" : ticker, 
                "strategy" : optionStrategy, 
                "legs" : legs, 
                "winpnlid" : winpnlid, 
                "losepnlid" : losepnlid,
                "days_forward": days_forward,
                "near_diff": NearDiff,
                "far_diff": FarDiff,
                "NearOTM": NearOTM,
                "FarOTM": FarOTM,
                "p_pnl": float(p_pnl) or 0, 
                "p_tr": float(p_tr) or 0, 
                "p_nd": float(p_nd) or 0, 
                "p_fd": float(p_fd) or 0, 
                "p_no": p_no, 
                "p_fo": p_fo, 
                "p_daysforward": int(p_daysforward) or 0, 
                "p_wpc": float(p_wpc) or 0,
                "lose_pnl": float(lose_pnl) or 0,
                "lose_tr": int(lose_tr) or 0,
                "minimumSamples": minimumSamples,
                "ts_rows": ts_rows,
                "ts_startdate": ts_startdate,
                "ts_enddate": ts_enddate,
                "ts_days": ts_days,
                "contracts": p_contracts
                }    
        _id = collection.insert_one(record).inserted_id
        
        return _id
        
