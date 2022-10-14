import datetime
import torch

dataPath = "0706/"

executeTime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logPath = "log/"
logFullPath = f"{logPath}{executeTime}/"

distanceSet = [10]
# typeSet = ["B", "M"]
typeSet = ["B", "M", "W"]
tumorSet = ["05", "15", "20", "40"]
minStepAngle = 10
stepAngle = 10
nFoldAngle = stepAngle // minStepAngle
angleSet = [i for i in range(0, 360, minStepAngle)]

repitition = 10
nFold = 5
nValFold = 1    # must be zero or one
nTestFold = 1   # must be one
nTrainFold = nFold - nTestFold - nValFold
nFoldRep = 10

useTumorKfold = True
valRatio = 0.2
nFold = len(tumorSet)

# If false, K-fold cross validation is not performed.
useKfoldValidation = True
# useKfoldValidation = False

useAngleAugmentation = True
# useAngleAugmentation = False
# useWindowAugmentation = True
useWindowAugmentation = False
useAugmentation = useAngleAugmentation or useWindowAugmentation

# When useWindowAugmentation is True
# startIdx = 45
# windowSize = 20
# pickIdxRange = 10   # must be even number
# pickIdxNum = 4

# When useWindowAugmentation is False
startIdx = 0
windowSize = 188
# startIdx = 5
# windowSize = 180

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\t\tUsing {device} device\n")

inChannel = len(angleSet) // nFoldAngle
# must remain the first element as 'inChannel'
convChannel = [inChannel, 64, 128, 256]
convKernel = [3 for x in range(len(convChannel) - 1)]
# convKernel2 = [85 for x in range(len(convChannel) - 1)]
convStride = [1 for x in range(len(convChannel) - 1)]
# convPadding = ["valid" for x in range(len(convChannel) - 1)]
convPadding = ["same" for x in range(len(convChannel) - 1)]

dropoutRate = 0

batchSize = 64
testBatchSize = len(angleSet)

minTrainAcc = 0.99
earlyStopPatience = 10

lossFunction = torch.nn.CrossEntropyLoss()
learningEpoch = 1000
learningRate = 1e-4
