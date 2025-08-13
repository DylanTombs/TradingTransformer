import numpy as np
import pandas as pd
import torch
import random
from argparse import Namespace as dotdict
from Transformer.Interface import Model_Interface

# Configuration
trainPath = "all_stocks_processed.csv"
testPath = "small_test_stock_processed.csv"

def prepareData(path):
    """Load and prepare the data"""
    data = pd.read_csv(path, parse_dates=['date'])  
    return data

def setupArgs():
    """Configure all the model arguments"""
    args = dotdict()
    
    # Data loader configuration
    args.target = 'close'
    args.auxilFeatures = ['high','low','volume','adj close','P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3','obv',
        'volume_zscore', 'rsi', 'macd','macds','macdh','sma','lma','sema','lema','overnight_gap',
        'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility', 'SR_K', 'SR_D', 
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']
    args.checkpoints = './checkpoints/'
    
    # Forecasting task parameters
    args.seqLen = 30
    args.labelLen = 10
    args.predLen = 5
    
    # Model architecture
    args.encIn = len([args.target]) + len(args.auxilFeatures)
    args.decIn = args.encIn
    args.cOut = 1
    args.dModel = 256
    args.nHeads = 4
    args.eLayers = 2
    args.dLayers = 1
    args.dFf = 256
    args.factor = 1
    args.dropout = 0.05
    
    # Optimization parameters
    args.numWorkers = 0
    args.itr = 1
    args.trainEpochs = 100
    args.batchSize = 128
    args.patience = 10
    args.learningRate = 0.0005
    
    args.devices = '0,1,2,3'
    args.seed = 1234
    
    # Projector params
    args.pHiddenDims = [128, 128]
    args.pHiddenLayers = 2
    
    return args

def main(): 
    args = setupArgs()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print('Args in experiment:')
    print(vars(args))
    
    for i in range(args.itr):
        model = Model_Interface(args)
        trainData = prepareData(trainPath)
        print(f'>>>>>>>start training>>>>>>>>>>>>')
        model.train(trainData)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
