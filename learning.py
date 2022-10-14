import os
import sys
import numpy as np
from tqdm import tqdm
import torchsummary

import torch
from torch.utils.data import DataLoader

from global_vars import *
from readDataset import readDataset
from makeDataset import makeDataset
from TumorBottleCNN import TumorBottleCNN
from trainModel import trainModel
from testModel import testModel

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import interp
print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


if __name__ == "__main__":
    try:
        if stepAngle % minStepAngle != 0:
            raise Exception(f"stepAngle= {stepAngle} must be separated "
                f"by minStepAngle= {minStepAngle}!")

        totDataSet, totLabelSet = readDataset()

        if not os.path.exists(logPath):
            os.mkdir(logPath)
        os.mkdir(logFullPath)

        for fold in range(nFold):
            try:
                if useKfoldValidation is False:
                    fold = nFold - 1
                    print("\t** DOES NOT USE K-FOLD CROSS VALIDATION **")
                    print("")

                if nValFold > 0:
                    trainSet, valSet, testSet =\
                        makeDataset(totDataSet, totLabelSet, fold)
                else:
                    trainSet, testSet =\
                        makeDataset(totDataSet, totLabelSet, fold)

                for rep in range(nFoldRep):
                    trainLoader = DataLoader(
                        trainSet, batch_size=batchSize, shuffle=True
                    )
                    testLoader = DataLoader(
                        testSet, batch_size=testBatchSize, shuffle=False
                    )
                    if nValFold > 0:
                        valLoader = DataLoader(
                            valSet, batch_size=testBatchSize, shuffle=False
                        )

                    model = TumorBottleCNN().to(device)
                    torchsummary.summary(
                        model, input_size=(inChannel, windowSize), device=device
                    )
                    print("")

                    # optimizer =\
                    #     torch.optim.SGD(model.parameters(), lr=learningRate)
                    optimizer =\
                        torch.optim.Adam(model.parameters(), lr=learningRate)


                    if nValFold > 0:
                        trainModel(trainLoader, model, optimizer,
                            fold, rep, valLoader
                        )
                    else:
                        trainModel(trainLoader, model, optimizer,
                            fold, rep, valLoader=""
                        )

                    testModel(testLoader, model, optimizer, fold, rep)

                if useKfoldValidation is False:
                    break

                fileS = open(f"{logFullPath}summary", "a")
                fileS.write("\n")
                fileS.close()

            except Exception as ex:
                _, _, tb = sys.exc_info()
                print(f"\n\n[main:fold {fold} - rep {rep}:"
                    f"{tb.tb_lineno}] {ex}\n\n"
                )

                if useKfoldValidation is False:
                    break

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[main:{tb.tb_lineno}] {ex}\n\n")




#
#
# #matplotlib inline
# import glob
# from collections import defaultdict
# from itertools import chain
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# #import seaborn as sns
# import scipy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import transformers
# import tqdm
# from sklearn.metrics import classification_report
#
# import math
#

#
#
#
# model = Novelda0322CNN(config)
# from torchsummary import summary
# model.cuda()
# summary(model,(36,64))
#
#
#
#
# model = Novelda0322CNN(config).to("cuda")
#
# epochs = 20
# optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.9)
# loss_fn = nn.CrossEntropyLoss()
#
# losses = [np.nan]
# accs = [0.0]
# val_losses = [np.nan]
# val_accs = [0.0]
#
# bestModel = None
# bestValLoss = math.inf
#
# pbar = tqdm.trange(epochs)
# for epoch in pbar:
#     model.train()
#     for x, y in train_loader:
#         print(x.shape)
#         print(y.shape)
#         x, y = x.to(config.device), y.to(config.device)
#         optim.zero_grad()
#
#         out = model(x)
#         loss = loss_fn(out, y.long())
#         loss.backward()
#         optim.step()
#
#         losses.append(loss.item())
#
#         pbar.set_description(f"loss: {losses[-1]:.3f} val_loss: {np.mean(val_losses[-len(val_loader):]):.3f}, val_acc: {np.mean(val_accs[-len(val_loader):]):.3f}, best: {bestValLoss:.3f}", refresh=True)
#
#     model.eval()
#
#     for x, y in val_loader:
#         print(x.shape)
#         print(y.shape)
#         x, y = x.to(config.device), y.to(config.device)
#         out = model(x)
#         loss = loss_fn(out, y.long())
#
#
#         val_losses.append(loss.item())
#         print(val_losses)
#         correct = out.argmax(axis=1) == y
#         val_accs.append((correct.sum()/y.shape[0]).item())
#
#         pbar.set_description(f"loss: {np.mean(losses[-len(train_loader)]):.3f}, val_loss: {val_losses[-1]:.3f}, val_acc: {val_accs[-1]:.3f}, best: {bestValLoss:.3f}", refresh=True)
#
