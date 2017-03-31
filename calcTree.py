import sys
sys.path.append(r'D:\Users\*****\PycharmProjects\ML in Py\'')
import Decision_Tree1
myDat,labels=Decision_Tree1.createDataSet()
myDat
print(myDat)
print(labels)
entropy=Decision_Tree1.CalcShannonEnt(myDat)
print(entropy)


[out]
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
['no surfacing', 'flippers']
0.9709505944546686
[in]
splitDataSet = Decision_Tree1.splitDataSet(myDat,1,1)
print(splitDataSet)
[out]
[[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]

bestFeature,subData = Decision_Tree1.chooseBestFeatureToSplitData(myDat)
print(bestFeature,subData)
print(myDat)
