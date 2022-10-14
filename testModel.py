import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from global_vars import *



def testModel(testLoader, model, optimizer, fold, rep, mode=""):
    try:
        # Assuming that one batch has one label in the test set,
        # one evaluation is made within the batch.
        totLoss = 0
        totCorrect = 0

        typeElement = len(testLoader) // len(typeSet)
        typeElementCount = 0
        typeCorrect = []
        typeCorrectCount = 0
        typeRawAcc = []
        typeRawCorrect = 0

        with torch.no_grad():
            for X, y in testLoader:
                pred = model(X.to(device))
                loss = lossFunction(pred, y.to(device).long())
                totLoss += loss

                correct =\
                    (pred.argmax(1) == y.to(device))\
                    .type(torch.float).sum().item()

                if correct > int(0.5 * testBatchSize):
                    typeCorrectCount += 1

                typeRawCorrect += correct

                typeElementCount += 1
                if typeElementCount >= typeElement:
                    typeCorrect.append(typeCorrectCount)
                    totCorrect += typeCorrectCount

                    typeElementCount = 0
                    typeCorrectCount = 0

                    typeRawAcc.append(
                        typeRawCorrect / typeElement / testBatchSize
                    )
                    typeRawCorrect = 0

        totLoss /= len(testLoader)

        if mode == "val":
            return totLoss, totCorrect
        else:
            file = open(f"{logFullPath}fold_{fold + 1}_{rep + 1}_test", "w")
            fileS = open(f"{logFullPath}summary", "a")

            print(f"Test result= {totCorrect:4d} / {len(testLoader):4d}, "
                f"acc= {100 * totCorrect / len(testLoader):6.2f}%")
            print("")
            file.write(f"{totLoss}\t{totCorrect}\t{len(testLoader)}\t"
                f"{totCorrect / len(testLoader)}\n"
            )
            fileS.write(f"{totCorrect / len(testLoader)}\t")

            for typeIdx in range(len(typeSet)):
                print(f"{typeSet[typeIdx]}\t"
                    f"{typeCorrect[typeIdx]:4d} / {typeElement:4d}, "
                    f"acc= {100 * typeCorrect[typeIdx] / typeElement:6.2f}%, "
                    f"raw acc= {100 * typeRawAcc[typeIdx]:6.2f}%")
                file.write(f"{typeSet[typeIdx]}\t{typeCorrect[typeIdx]}\t"
                    f"{typeElement}\t{typeCorrect[typeIdx] / typeElement}\t"
                    f"{typeRawAcc[typeIdx]}\n"
                )
                fileS.write(f"{typeCorrect[typeIdx] / typeElement}\t")
            for typeIdx in range(len(typeSet)):
                fileS.write(f"{typeRawAcc[typeIdx]}\t")
            print("")
            fileS.write("\n")

            file.close()
            fileS.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[testModel:{tb.tb_lineno}] {ex}\n\n")


# from sklearn.metrics import confusion_matrix

# pipe_svc.fit(X_train, y_train)
# y_pred = pipe_svc.predict(X_test)
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(confmat)
