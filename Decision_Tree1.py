from math import log
def CalcShannonEnt(dataSet):
    numEntropy = len(dataSet)
    lableCounts = {}
    for featVec in dataSet:
        currentLable=featVec[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable] +=1
    shannonEnt=0.0
    for key in lableCounts:
        prob = float(lableCounts[key])/numEntropy
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
def splitDataSet(dataSet, axis, value):
    retDataSet=  []
    for featvect in dataSet:
        if featvect[axis] == value:
            ReducedFeatVect = featvect[:axis]
            ReducedFeatVect.extend(featvect[axis+1:])#提出没有featvext[axis]=value 的featvect
            retDataSet.append(ReducedFeatVect)
    return retDataSet

def chooseBestFeatureToSplitData(dataSet):
    numFeatures = len(dataSet[0])-1#扣除结果‘yes’or 'no'
    baseEntropy = CalcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*CalcShannonEnt(subDataSet)
        infoGain= baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
            subDataSetOut=subDataSet
    return bestFeature, subDataSetOut

