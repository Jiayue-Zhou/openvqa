import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
        pass

    def gqa_init(self, __C):
        pass

    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def vqa_forward(self, feat_dict):
        pass

    def gqa_forward(self, feat_dict):
        pass

    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)

        return img_feat, img_feat_mask


