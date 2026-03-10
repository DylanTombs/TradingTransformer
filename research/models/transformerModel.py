from Transformer.Interface import Model_Interface
import torch

class TransformerModel:

    def __init__(self, model_path, args):

        self.model = Model_Interface(args)

        self.model.model.load_state_dict(
            torch.load(model_path)
        )

        self.model.model.eval()

    def predict(self, window):

        with torch.no_grad():
            prediction = self.model.predict(window)

        return prediction