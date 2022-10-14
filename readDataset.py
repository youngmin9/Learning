import os
import sys
import numpy as np
from tqdm import tqdm

from global_vars import *

def readDataset():
    try:
        totDataSet = []
        totLabelSet = []

        assert os.path.exists(dataPath), f"{dataPath} does not exist."

        if useTumorKfold is True:
            for x in range(nFold):
                totDataSet.append([[], []])
                totLabelSet.append([[], []])
            nVal = repitition * valRatio
        else:
            for x in range(nFold):
                totDataSet.append([])
                totLabelSet.append([])

            if repitition % nFold != 0:
                raise Exception(f"Repitition= {repitition} must be separated by "
                    f"nFold= {nFold}!")

            nFoldElement = int(repitition / nFold)

            if nFold != nTrainFold + nValFold + nTestFold:
                raise Exception(f"The sum of nTrainFold= {nTrainFold}, "
                    f"nValFold= {nValFold}, and nTestFold= {nTestFold} "
                    f"does not match nFold= {nFold}")

        print("\t** READ DATASET **")
        print(f"Distance= {distanceSet}")
        print(f"Type= {typeSet}")
        print(f"Tumor= {tumorSet}")
        print(f"Repitition = {repitition}")
        print(f"\t*****")
        print(f"Angle= {angleSet}")
        print(f"\t*****")
        print("")

        nData =\
            int(len(distanceSet) * len(typeSet) * len(tumorSet) * repitition)
        pbar = tqdm(total=nData, desc="READING", ncols=100, unit=" data")

        for distance in distanceSet:
            # calData = np.load(f"{dataPath}{distance}_C.npy")
            # # calData.shape == (120, 188)
            #
            # calData = np.mean(calData, axis=0)
            # # calData.shape == (188,)

            for typeIdx in range(len(typeSet)):
                for rep in range(repitition):
                #for tumorIdx in range(len(tumorSet)):
                    
                    #for rep in range(repitition):
                    for angle in angleSet:
                        tmpData = []

                        #for angle in angleSet:
                        for tumorIdx in range(len(tumorSet)):
                    
                            fileName =\
                                f"{distance}_{typeSet[typeIdx]}"\
                                f"_{tumorSet[tumorIdx]}_{rep + 1}_{angle}"

                            data = np.load(f"{dataPath}{fileName}.npy")
                            # data.shape == (120, 188)

                            data = np.mean(data, axis=0)
                            # data.shape == (188,)

                            if useWindowAugmentation is True:
                                margin = pickIdxRange // 2
                                tmpData.append(
                                    (data)[
                                    # (data - calData)[
                                        startIdx - margin :\
                                        startIdx + windowSize + margin
                                    ]
                                )
                            else:
                                tmpData.append(
                                    (data)[
                                    # (data - calData)[
                                        startIdx : startIdx + windowSize
                                    ]
                                )

                        # tmpData.shape == (36, 188)

                        if useTumorKfold is True:
                            if rep < repitition - nVal:
                                idx = 0
                            else:
                                idx = 1
                            totDataSet[tumorIdx][idx].append(tmpData)
                            totLabelSet[tumorIdx][idx].append(typeIdx)
                        else:
                            curFold = rep // nFoldElement
                            totDataSet[curFold].append(tmpData)
                            totLabelSet[curFold].append(typeIdx)

                        pbar.update(1)

        pbar.close()
        print("")



        print("\t*****")
        print("totDataSet= [ ", end="")
        for x in range(nFold):
            print(f"{np.array(totDataSet[x][0]).shape} ", end="")
            print(f"{np.array(totDataSet[x][1]).shape} ", end="")
        print("]")
        print("totLabelSet= [ ", end="")
        for x in range(nFold):
            print(f"{np.array(totLabelSet[x][0]).shape} ", end="")
            print(f"{np.array(totLabelSet[x][1]).shape} ", end="")
        print("]")


        # print("\t** VALIDATE DATASET **")
        # print(f"Repitition= {repitition}, nFold= {nFold}, "
        #     f"nFoldElement= {nFoldElement}")
        # print(f"nLabel= {len(typeSet)}")
        # print("\t*****")
        # print("totDataSet= [ ", end="")
        # for x in range(nFold):
        #     print(f"{np.array(totDataSet[x]).shape} ", end="")
        #     if len(totDataSet[x]) !=\
        #         int(nFoldElement * len(typeSet) * len(tumorSet)):
        #         raise Exception("Incomplete dataset!")
        # print("]")
        # print("totLabelSet= [ ", end="")
        # for x in range(nFold):
        #     print(f"{np.array(totLabelSet[x]).shape} ", end="")
        #     if len(totLabelSet[x]) !=\
        #         int(nFoldElement * len(typeSet) * len(tumorSet)):
        #         raise Exception("Incomplete dataset!")
        # print("]")
        print("\t*****")
        print("")

        return totDataSet, totLabelSet

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[readDataset:{tb.tb_lineno}] {ex}\n\n")
