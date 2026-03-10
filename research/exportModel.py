import torch
import os
import torch.nn as nn
from argparse import Namespace as dotdict
from transformer.Interface import Model_Interface


class TransformerInferenceWrapper(nn.Module):

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.predLen: int = args.predLen
        self.labelLen: int = args.labelLen

    def forward(self, xEnc: torch.Tensor, xMarkEnc: torch.Tensor) -> torch.Tensor:
        batch = xEnc.shape[0]
        features = xEnc.shape[2]
        markFeatures = xMarkEnc.shape[2]

        xDec = torch.zeros(batch, self.labelLen + self.predLen, features)
        xMarkDec = torch.zeros(batch, self.labelLen + self.predLen, markFeatures)

        output, _ = self.model(xEnc, xMarkEnc, xDec, xMarkDec)
        return output


def export_model():
    args = load_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "Model3.pth")

    model_interface = Model_Interface(args)
    model_interface.model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model = model_interface.model
    model.eval()

    wrapper = TransformerInferenceWrapper(model, args)
    wrapper.eval()

    # Create dummy inputs matching your expected input shapes
    batch_size = 1
    num_features = args.encIn
    num_mark_features = 3  # adjust to your time mark feature count (e.g. hour, day, month, weekday)

    dummy_xEnc = torch.zeros(batch_size, args.seqLen, num_features)
    dummy_xMarkEnc = torch.zeros(batch_size, args.seqLen, num_mark_features)

    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, (dummy_xEnc, dummy_xMarkEnc))

    os.makedirs("models", exist_ok=True)
    traced_model.save("models/transformer.pt")
    print("Model exported to models/transformer.pt")


def load_args():
        args = dotdict()
        
        args.target = 'close'
        args.auxilFeatures = ['high','low','volume','adj close','P', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3','obv',
        'volume_zscore', 'rsi', 'macd','macds','macdh','sma','lma','sema','lema','overnight_gap',
        'return_lag_1', 'return_lag_3', 'return_lag_5', 'volatility', 'SR_K', 'SR_D', 
                    'SR_RSI_K', 'SR_RSI_D', 'ATR', 'HL_PCT', 'PCT_CHG']
        args.checkpoints = './models/'

        args.seqLen = 30
        args.labelLen = 10
        args.predLen = 5

        args.encIn = len([args.target]) + len(args.auxilFeatures)
        args.decIn = args.encIn
        args.cOut = 1
        args.dModel = 256       
        args.nHeads = 8      
        args.eLayers = 3     
        args.dLayers = 2        
        args.dFf = 512           
        args.factor = 1         
        args.dropout = 0.1

        args.num_workers = 0
        args.itr = 1
        args.trainEpochs = 100
        args.batchSize = 128
        args.patience = 10
        args.learningRate = 0.0005

        args.devices = '0,1,2,3'
        args.seed = 1234

        args.pHiddenDims = [128, 128]
        args.pHiddenLayers = 2

        return args

if __name__ == "__main__":
    export_model()