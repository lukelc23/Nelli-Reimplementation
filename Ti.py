import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import itertools
import torch.nn.functional as F

class Ti:

    def createOneHot(n, num_items):
        pos = n-1
        oneHot = [0]* num_items
        oneHot[pos] = 1
        return oneHot

    #create individual object vectors all in one list of lists
    def createData(n):
        data = []
        for i in range(n):
            oneHot = [0]*n
            oneHot[i] = 1
            data.append(oneHot)
        return data
    
    #create concatenated vector based on two oneHot vectors from scalars
    def createTD(data):
        inputData = []
        for i in range(len(data) - 1):
            inputData.append(data[i] + data[i+1])
        return inputData
    
    #create reverse concatenated vector
    def createTDReverse(data):
        inputDataReverse = []
        for i in range(len(data) - 1):
            inputDataReverse.append(data[i+1] + data[i])
        return inputDataReverse
    
    #fully concatenated? 
    def createTDTotal(data):
        input = Ti.createTD(data)
        inputReverse = Ti.createTDReverse(data)
        return input + inputReverse
    
    #creates all labels based on an array of values  
    def createTDLabels(data):
        forwardArr = Ti.createTD(data)
        reverseArr = Ti.createTDReverse(data)
        forwardLabel = [1] * len(forwardArr)
        reverseLabel = [-1] * len(reverseArr)
        return forwardLabel + reverseLabel 
    
    def createTestSet_w_Labels(n):
        num_items = n
        item_indices = torch.arange(num_items)  # Indices for items 'A', 'B', 'C'
        one_hot_vectors = F.one_hot(item_indices, num_classes = num_items) #creates one hot vectors more easily than above in a tensor
        testSet = []
        testLabels = []
        for idx_i, i in enumerate(one_hot_vectors):
            for idx_j, j in enumerate(one_hot_vectors):
                    concatenated = torch.cat((i,j))
                    testSet.append(concatenated.tolist())
                    if idx_i < idx_j:
                        testLabels.append(1)
                    else:
                        testLabels.append(-1)
                    print(idx_i, idx_j)
        return [testSet, testLabels]

    def createTDExp(p,q,num_items):
        exception = Ti.createOneHot(p,num_items) + Ti.createOneHot(q,num_items)
        exceptionReverse = Ti.createOneHot(q,num_items) + Ti.createOneHot(p,num_items)  
        return [exception,exceptionReverse]

    def createTDLabelsExp():
        return [1,-1]
    
    def createTDTotalExp(data, p, q, num_items):
        inputTotal = Ti.createTDTotal(data)
        exception = Ti.createTDExp(p,q,num_items)
        print('inputTotal:', inputTotal)
        print('exception:', exception)
        return inputTotal + exception
    
    def createTDLabelsTotalExp(data):
        labels = Ti.createTDLabels(data)
        exceptionLabels = Ti.createTDLabelsExp()
        print('labels:', labels)
        print('labelsExp:', exceptionLabels)
        return labels + exceptionLabels

    def createTestSet_w_LabelsExp(n,p,q,data):
        num_items = n
        item_indices = torch.arange(num_items)  # Indices for items 'A', 'B', 'C'
        one_hot_vectors = F.one_hot(item_indices, num_classes = num_items) #creates one hot vectors more easily than above in a tensor
        testSet = []
        testLabels = []
        for idx_i, i in enumerate(one_hot_vectors):
            for idx_j, j in enumerate(one_hot_vectors):
                    concatenated = torch.cat((i,j))
                    testSet.append(concatenated.tolist())
                    if idx_i < idx_j:
                        testLabels.append(1)
                    else:
                        testLabels.append(-1)
        
        print('testSetException:', testSet)
        print('testLabelsException:', testLabels)
        return testSet, testLabels

    def itemsToTensors(num_items):
        testSet, testLabels = Ti.createTestSet_w_Labels(num_items)
        testSet = torch.tensor(testSet)
        testLabels = torch.tensor(testLabels)
        
        data = Ti.createData(num_items) 
        TDTotal = torch.tensor(Ti.createTDTotal(data))
        labels = torch.tensor(Ti.createTDLabelsTotalExp(data))

        print('data: \n', data)
        print('Concatenated Input vectors: \n' ,TDTotal)
        print('Labels: \n', labels)
        print('Testing Set: \n', testSet)
        print('Testing Labels: \n', testLabels)

        return testSet, testLabels, TDTotal, labels

    def itemsToTensorsException(num_items,p,q):
        data = Ti.createData(num_items) 
        
        testSetExp, testLabelsExp = Ti.createTestSet_w_LabelsExp(num_items, p, q, data)
        testSetExp = torch.tensor(testSetExp)
        testLabelsExp = torch.tensor(testLabelsExp)
        
        
        TDTotalExp = torch.tensor(Ti.createTDTotalExp(data, p, q, num_items))
        labelsExp = torch.tensor(Ti.createTDLabelsTotalExp(data))
            
        print('data: \n', data)
        print('Concatenated Input vectors Exp: \n' ,TDTotalExp)
        print('Labels Exp: \n', labelsExp)
        print('Testing SetExp: \n', testSetExp)
        print('Testing Labels Exp: \n', testLabelsExp)

        return testSetExp, testLabelsExp, TDTotalExp, labelsExp
