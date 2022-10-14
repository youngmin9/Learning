import os
import sys
import numpy as np
from tqdm import tqdm

from global_vars import *

def augmentDataset(dataSet, labelSet):
    try:
        if useAngleAugmentation is True:
            angleDataset = []
            angleLabelset = []

            for setIdx in range(len(labelSet)):
                for foldAngle in range(nFoldAngle):
                    for rollAngle in range(
                        foldAngle, len(angleSet), nFoldAngle
                    ):
                        curData = np.roll(dataSet[setIdx], -rollAngle, axis=0)
                        if useWindowAugmentation is True:
                            newData = np.zeros(
                                (len(angleSet) // nFoldAngle,
                                windowSize + pickIdxRange
                                )
                            )
                        else:
                            newData = np.zeros(
                                (len(angleSet) // nFoldAngle, windowSize)
                            )

                        for x in range(len(angleSet) // nFoldAngle):
                            newData[x] = curData[x * nFoldAngle]

                        angleDataset.append(newData)
                        angleLabelset.append(labelSet[setIdx])

            augmentedDataset = angleDataset
            augmentedLabelset = angleLabelset
        else:
            augmentedDataset = dataSet
            augmentedLabelset = labelSet

        if useWindowAugmentation is True:
            windowDataset = []
            windowLabelset = []

            for setIdx in range(len(augmentedLabelset)):
                for x in range(pickIdxNum):
                    randomMargin = np.random.randint(pickIdxRange)
                    curData = augmentedDataset[setIdx]
                    newData = curData[:,
                        randomMargin : windowSize + randomMargin
                    ]
                    windowDataset.append(newData)
                    windowLabelset.append(augmentedLabelset[setIdx])

            augmentedDataset = windowDataset
            augmentedLabelset = windowLabelset

        return augmentedDataset, augmentedLabelset

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[augmentDataset:{tb.tb_lineno}] {ex}\n\n")
