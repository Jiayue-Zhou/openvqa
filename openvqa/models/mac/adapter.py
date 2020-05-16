# --------------------------------------------------------
# OpenVQA
# Written by Jiayue Zhou https://github.com/Jiayue-Zhou
# based on the implementation in https://github.com/rosinality/mac-network-pytorch
# Use ELU as the activation function in the CNN
# --------------------------------------------------------

import torch.nn as nn
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask
from torch.nn.init import kaiming_uniform_

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
        pass

    def gqa_init(self, __C):
        pass

    def clevr_init(self, __C):
        #self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], 1024)

        '''Two CNN layers before inputting to MAC units'''

        self.conv = nn.Sequential(nn.Conv2d(1024, __C.HIDDEN_SIZE, 3, padding = 1),
                                  nn.ELU(),
                                  nn.Conv2d(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, 3, padding = 1),
                                  nn.ELU())

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

    def vqa_forward(self, feat_dict):
        pass

    def gqa_forward(self, feat_dict):
        pass

    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)

        img_feat = grid_feat.permute(0, 2, 1)
        img_feat = img_feat.view(-1, 1024, 14, 14)
        img_feat = self.conv(img_feat)

        return img_feat, img_feat_mask


