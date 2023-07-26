import torch
import torch.nn.functional as F
import sys
sys.path.append("../../../../")
from adversarial_malware_samples_generation.ml_models.torch_models.nonnegative_malconv_detector.malconv_arch import MalConv
import numpy as np

NONNEG_MODEL_PATH = 'models/nonneg.checkpoint'

class MalConvModel(object):
    def __init__(self, model_path, thresh=0.5, name='malconv'):
        self.model = MalConv(channels=256, window_size=512, embd_size=8).train()
        weights = torch.load(model_path,map_location='cpu')
        self.model.load_state_dict( weights['model_state_dict'])
        self.thresh = thresh
        self.__name__ = name

    def predict(self, bytez):
        np_bytez = np.frombuffer(bytez, dtype=np.uint8)[np.newaxis,:]
        #print(np_bytez)
        #print(np_bytez.shape)
        _inp = torch.from_numpy(np_bytez.copy())
        with torch.no_grad():
            outputs = F.softmax( self.model(_inp), dim=-1)
        #print(float(outputs[0][1]))
        return float(outputs[0][1]), outputs.detach().numpy()[0,1] > self.thresh


if __name__ == '__main__':
    import sys
    with open(sys.argv[1],'rb') as infile:
        bytez = infile.read()

    # thresholds are set here
    nonneg_malconv = MalConvModel( NONNEG_MODEL_PATH, thresh=0.35, name='nonneg_malconv' )

    models = [nonneg_malconv]
    for m in models:
        print( f'{m.__name__}: {m.predict(bytez)}')