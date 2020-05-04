import torch.nn as nn
from openvqa.models.mac.adapter import Adapter
from openvqa.models.mac.mac import MACUnit
from torch.nn.init import xavier_uniform_


def linear(in_dim, out_dim, bias = True):
    lin = nn.Linear(in_dim, out_dim, bias = bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(num_embeddings = token_size,
                                      embedding_dim = __C.WORD_EMBED_SIZE)

        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size = __C.WORD_EMBED_SIZE,
            hidden_size = __C.HIDDEN_SIZE,
            batch_first = True,
            bidirectional = True
        )

        self.adapter = Adapter(__C)

        self.backbone = MACUnit(__C)

        self.classifier = nn.Sequential(linear(__C.HIDDEN_SIZE * 3, __C.HIDDEN_SIZE),
                                        nn.ELU(),
                                        linear(__C.HIDDEN_SIZE, answer_size))
        self.b_size = __C.BATCH_SIZE

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        #lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, (h, _) = self.lstm(lang_feat)

        img_feat, _, = self.adapter(frcn_feat, grid_feat, bbox_feat)
        h = h.permute(1, 0, 2).contiguous().view(self.b_size, -1)
        memory = self.backbone(lang_feat, h, img_feat)

        return memory







