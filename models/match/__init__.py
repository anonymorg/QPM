# -*- coding: utf-8 -*-



from models.match.SiameseNetwork import SiameseNetwork

def setup(opt):
    print("matching network type: Siamese Network with " + opt.network_type)

    model = SiameseNetwork(opt)
    return model
