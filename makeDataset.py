import os
import sys
import numpy as np
from tqdm import tqdm

from global_vars import *
from BasicDataset import BasicDataset
from augmentDataset import augmentDataset

def makeDataset(totDataSet, totLabelSet, fold):
    try:
        if useTumorKfold is True:
            print(f"\t** FOLD {fold + 1} ** CONFIRM FOLD INFO **")
            print(f"testTumor= {tumorSet[fold]}")
            print("\t*****")

            trainDataSet = []
            trainLabelSet = []
            valDataSet = []
            valLabelSet = []
            testDataSet = []
            testLabelSet = []

            for x in range(nFold):
                if x != fold:
                    trainDataSet.extend(totDataSet[x][0])
                    trainLabelSet.extend(totLabelSet[x][0])
                    valDataSet.extend(totDataSet[x][1])
                    valLabelSet.extend(totLabelSet[x][1])
                else:
                    testDataSet.extend(totDataSet[x][0])
                    testDataSet.extend(totDataSet[x][1])
                    testLabelSet.extend(totLabelSet[x][0])
                    testLabelSet.extend(totLabelSet[x][1])
        else:
            if nValFold > 0:
                testFoldIdx = fold
                valFoldIdx = (fold - 1) % nFold
                trainFoldIdx = []
                for x in range(nFold - 2):
                    trainFoldIdx.append((fold + x + 1) % nFold)
            else:
                testFoldIdx = fold
                trainFoldIdx = []
                for x in range(nFold - 1):
                    trainFoldIdx.append((fold + x + 1) % nFold)

            print(f"\t** FOLD {fold + 1} ** CONFIRM FOLD INDEX **")
            print(f"trainFoldIdx= {trainFoldIdx}")
            if nValFold > 0:
                print(f"valFoldIdx= {valFoldIdx}, testFoldIdx= {testFoldIdx}")
            else:
                print(f"testFoldIdx= {testFoldIdx}")
            print("\t*****")

            trainDataSet = []
            trainLabelSet = []

            for x in trainFoldIdx:
                trainDataSet.extend(totDataSet[x])
                trainLabelSet.extend(totLabelSet[x])

            print(f"trainDataSet= {np.array(trainDataSet).shape}, "
                f"trainLabelSet= {np.array(trainLabelSet).shape}")

            if nValFold > 0:
                valDataSet = totDataSet[valFoldIdx]
                valLabelSet = totLabelSet[valFoldIdx]

                print(f"valDataSet= {np.array(valDataSet).shape}, "
                    f"valLabelSet= {np.array(valLabelSet).shape}")

            testDataSet = totDataSet[testFoldIdx]
            testLabelSet = totLabelSet[testFoldIdx]

            print(f"testDataSet= {np.array(testDataSet).shape}, "
                f"testLabelSet= {np.array(testLabelSet).shape}")

        print("\t** PROCEED AUGMENTATION **")

        trainDataSet, trainLabelSet =\
            augmentDataset(trainDataSet, trainLabelSet)

        print(f"trainDataSet= {np.array(trainDataSet).shape}, "
            f"trainLabelSet= {np.array(trainLabelSet).shape}")

        if nValFold > 0:
            valDataSet, valLabelSet =\
                augmentDataset(valDataSet, valLabelSet)

            print(f"valDataSet= {np.array(valDataSet).shape}, "
                f"valLabelSet= {np.array(valLabelSet).shape}")

        testDataSet, testLabelSet =\
            augmentDataset(testDataSet, testLabelSet)

        print(f"testDataSet= {np.array(testDataSet).shape}, "
            f"testLabelSet= {np.array(testLabelSet).shape}")

        print("\t*****")
        print("")

        trainSet = BasicDataset(trainDataSet, trainLabelSet)
        testSet = BasicDataset(testDataSet, testLabelSet)

        if nValFold > 0:
            valSet = BasicDataset(valDataSet, valLabelSet)
            return trainSet, valSet, testSet
        else:
            return trainSet, testSet

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[makeDataset:{tb.tb_lineno}] {ex}\n\n")
