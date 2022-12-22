import os.path
import torch

from bsrgan.utils import utils_image as util
from bsrgan.models.network_rrdbnet import RRDBNet 
from bsrgan.main_download_pretrained_models import attempt_download_from_hub

class BSRGAN:
    def __init__(self, model_path, device):
        self.device = device
        self.save = True
        self.hf_model = False
        
        if self.hf_model:
            self.model_path = attempt_download_from_hub(model_path)
        else:
            self.model_path = model_path
            
        self.load_model()
    
    def load_model(self):
        
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        if [model_name] in ['BSRGANx2']:
            sf = 2
        else:
            sf = 4
            
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
        model.load_state_dict(torch.load(self.model_path), strict=True)
        model.eval()
        
        for k, v in model.named_parameters():
            v.requires_grad = False
            
        model = model.to(self.device)
        
        self.model_name = model_name
        self.model = model
        
    
    def predict(self, img_path):
        img = util.imread_uint(img_path, n_channels=3)
        img = util.uint2tensor4(img)
        img = img.to(self.device)
        img = self.model(img)
        img = util.tensor2uint(img)
        
        if self.save:
            save_path = os.path.join('results')
            util.mkdir(save_path)
            result = util.imsave(img, os.path.join(save_path, self.model_name+'.png'))
            return result

if __name__ == '__main__':
    model_path = 'kadirnar/bsrgan'
    img_path = 'data/images/butterfly.png'
    bsrgan = BSRGAN(model_path, device='cuda:0')
    result = bsrgan.predict(img_path)