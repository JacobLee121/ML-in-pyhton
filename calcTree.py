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
