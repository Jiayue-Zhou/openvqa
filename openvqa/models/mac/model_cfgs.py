from openvqa.core.base_cfgs import BaseCfgs
class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.HIDDEN_SIZE = 512
        self.DROPOUT = 0.15
        self.MEMORY_GATE = False
        self.SELF_ATTENTION = False
        self.MAX_STEP = 12

