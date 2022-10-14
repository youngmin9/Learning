import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from global_vars import *
from testModel import testModel

def trainModel(trainLoader, model, optimizer, fold, rep, valLoader):
    try:
        file = open(f"{logFullPath}fold_{fold + 1}_{rep + 1}_train", "w")
        fileV = open(f"{logFullPath}fold_{fold + 1}_{rep + 1}_validation", "w")

        print(f"\t** FOLD {fold + 1} - REP {rep + 1} **"
            f" TRAINING MAXIMUM {learningEpoch} EPOCH **\n"
        )

        bestLoss = 100  # Arbitrary big number
        bestEpoch = -1
        curPatience = 0
        bestTrainLoss = 0   # For summary logging
        bestTrainAcc = 0    # For summary logging
        bestValAcc = 0      # For summary logging 
        
        for epoch in range(learningEpoch):
            # Train
            for batch, (X, y) in enumerate(trainLoader):
                pred = model(X.to(device))
                loss = lossFunction(pred, y.to(device).long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Predict
            correct = 0

            with torch.no_grad():
                for X, y in trainLoader:
                    pred = model(X.to(device))
                    correct +=\
                        (pred.argmax(1) == y.to(device))\
                        .type(torch.float).sum().item()

            # Print result
            print(f"\rEpoch {epoch + 1:3d}\tTrain loss= {loss:.4f}, acc= "
                f"{100 * correct / len(trainLoader.dataset):6.2f}%", end="")

            file.write(f"{loss}\t{correct}\t{len(trainLoader.dataset)}\t"
                f"{correct / len(trainLoader.dataset)}\n"
            )

            if nValFold > 0:
                valLoss, valCorrect = testModel(
                    valLoader, model, optimizer, fold, rep, mode="val"
                )
                print(f"\tValidation loss= {valLoss:.4f}, "
                    f"acc= {100 * valCorrect / len(valLoader):6.2f}%",
                    end=""
                )
                fileV.write(f"{valLoss}\t{valCorrect}\t{len(valLoader)}\t"
                    f"{valCorrect / len(valLoader)}\n"
                )

            # Early Stopping
            if nValFold > 0:
                if correct / len(trainLoader.dataset) >= minTrainAcc:
                    if valLoss < bestLoss:
                        bestLoss = valLoss
                        bestEpoch = epoch
                        curPatience = 0
                        torch.save(model,
                            f"{logFullPath}fold_{fold + 1}_{rep + 1}_model"
                        )

                        bestTrainLoss = loss
                        bestTrainAcc = correct / len(trainLoader.dataset)
                        bestValAcc = valCorrect / len(valLoader)

                    else:
                        curPatience += 1

                        if curPatience >= earlyStopPatience:
                            print(f"\n\t\tEarly stop the training!")

                            model = torch.load(
                                f"{logFullPath}fold_{fold + 1}_{rep + 1}_model"
                            )
                            model.eval()

                            print(f"\t\tRestore the model weight at epoch"
                                f" {bestEpoch + 1}.\n"
                            )

                            break

                print(f"\t({curPatience:2d} / {earlyStopPatience:2d})"
                    f" Best= {bestLoss:.4f}     ", end=""
                )

        # When early stopping does not work
        if (epoch + 1) == learningEpoch:
            bestTrainLoss = loss
            bestTrainAcc = correct / len(trainLoader.dataset)
            bestValAcc = valCorrect / len(valLoader)
            bestLoss = valLoss

        # Summary logging
        fileS = open(f"{logFullPath}summary", "a")

        fileS.write(f"{bestTrainLoss}\t{bestTrainAcc}\t")
        if nValFold > 0:
            fileS.write(f"{bestLoss}\t{bestValAcc}\t")
        fileS.close()

        file.close()
        fileV.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[trainModel:{tb.tb_lineno}] {ex}\n\n")
