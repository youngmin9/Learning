import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from global_vars import *

class TumorBottleCNN(nn.Module):
    def __init__(self):
        super(TumorBottleCNN, self).__init__()

        self.featureExtraction = nn.Sequential()

        for x in range(len(convChannel) - 1):
            self.featureExtraction.add_module(
                name=f"Conv1d_{x + 1}", module=nn.Conv1d(
                    in_channels=convChannel[x], out_channels=convChannel[x + 1],
                    kernel_size=convKernel[x], stride=convStride[x],
                    padding=convPadding[x]
                )
            )
            self.featureExtraction.add_module(
                name=f"MaxPool1d_{x + 1}", module=nn.MaxPool1d(
                    kernel_size=2
                )
            )
            self.featureExtraction.add_module(
                name=f"Dropout_{x + 1}", module=nn.Dropout(
                    p=dropoutRate
                )
            )


        # self.featureExtraction2 = nn.Sequential()
        
        # for x in range(len(convChannel) - 1):
        #     self.featureExtraction2.add_module(
        #         name=f"Conv1d_{x + 1}", module=nn.Conv1d(
        #             in_channels=convChannel[x], out_channels=convChannel[x + 1],
        #             kernel_size=convKernel2[x], stride=convStride[x],
        #             padding=convPadding[x]
        #         )
        #     )
        #     self.featureExtraction2.add_module(
        #         name=f"MaxPool1d_{x + 1}", module=nn.MaxPool1d(
        #             kernel_size=3, stride=2
        #         )
        #     )
        #     self.featureExtraction2.add_module(
        #         name=f"Dropout_{x + 1}", module=nn.Dropout(
        #             p=dropoutRate
        #         )
        #     )

        self.flatten = nn.Flatten()

        flattenSize = windowSize
        for x in range(len(convChannel) - 1):
            # flattenSize -= (convKernel[x] - 1)
            flattenSize /= 2
            flattenSize = int(flattenSize)
            # flattenSize -= 1
        flattenSize = int(flattenSize * convChannel[-1])

        # flattenSize2 = windowSize
        # for x in range(len(convChannel) - 1):
        #     flattenSize2 -= (convKernel2[x] - 1)
        #     flattenSize2 /= 2
        #     flattenSize2 -= 1
        # flattenSize2 = int(flattenSize2 * convChannel[-1])
        # print(flattenSize, flattenSize2)
        #
        # flattenSize += flattenSize2

        self.classification = nn.Sequential(
            # nn.Linear(flattenSize, 64),
            # nn.Linear(64, 2)
            # nn.Linear(1024, 256),

#            nn.Linear(flattenSize, 256),
 #           nn.Linear(256, 64),
  #          nn.Linear(64, len(typeSet))
        )

    def forward(self, x):
        feature = self.flatten(self.featureExtraction(x))
        # return feature
        # feature2 = self.flatten(self.featureExtraction2(x))
        logits = self.classification(feature)
        # newFeature = torch.cat((feature, feature2), dim=1)
        # logits = self.classification(newFeature)
        return logits
