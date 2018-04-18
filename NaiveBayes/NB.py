from numpy import *

def loadDataSet():
    """
    :return: postingList(文本列表),classVec(分类)
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  # 这个是评论分词的结果
    classVec = [0,1,0,1,0,1]  # 这个是标签,0是正常评论,1是负面评论,这个是人工标注的
    return postingList,classVec

def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: (list)[list]分词结果
    :return: (list)[str]
    """
    vocabSet = set([])  #创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 去重,并合并所有的词语
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    创建该文本的词汇向量,词集模型(set of words model)
    :param vocabList: (list)[str]词汇表
    :param inputSet: (list)[str]文档
    :return:
    """
    returnVec = [0]*len(vocabList)  # 创建一个和词汇表相同长度的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 如果在词汇表中找到单词,则把该特征的值设为1
        else: print ("此词: %s 不在字典中!" % word)
    return returnVec


def setOfWords2Vec(vocabList, inputSet):
    """
    创建该文本的词汇向量,词袋模型(bag of words model)
    :param vocabList: (list)[str]词汇表
    :param inputSet: (list)[str]文档
    :return:
    """
    returnVec = [0]*len(vocabList)  # 创建一个和词汇表相同长度的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 如果在词汇表中找到单词,则把该特征的值设为1
        else: print ("此词: %s 不在字典中!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    """
    贝叶斯分类器训练函数,计算$P(w/c_i)$
    :param trainMatrix: (list)[list]文档词汇矩阵(其实是个二维列表,第一维度是文档列表,第二维度是每个文档里的词汇向量)
    :param trainCategory: (list)[int]标签向量(每个文档对应的标签)
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 训练文档的个数
    numWords = len(trainMatrix[0])  # 训练文档词语的个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算消极文档的概率$P(c_i)$(因为消极文档的label是1,所以可以直接sum)
    p0Num = ones(numWords); p1Num = ones(numWords) #将词语出现的次数初始化为1,防止某个词未出现而导致整个概率为0的情况
    p0Denom = 2.0; p1Denom = 2.0  # 因为分子为1,所以分母必须比1大
    for i in range(numTrainDocs):  # 遍历每一个文档
        if trainCategory[i] == 1:  # 如果该文档的标签为1,消极的
            p1Num += trainMatrix[i]  # 对应的词条加1
            p1Denom += sum(trainMatrix[i])  # 负面文章词语总数增加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])  # 计算正面文章里词语出现的次数
    p1Vect = log(p1Num/p1Denom)  # $p(w/c_1)$已知是负面文章的情况下,每个词语出现的次数除以总词语出现的次数,为了避免小数乘积过小溢出,用对数处理
    p0Vect = log(p0Num/p0Denom)  # $p(w/c_0)$已知是正面文章的情况下,每个词语出现的次数除以总词语出现的次数
    return p0Vect,p1Vect,pAbusive  #$p(w/c_1)$,$p(w/c_0)$,p(c_1)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    贝叶斯分类器
    :param vec2Classify: 要分类文章的词向量
    :param p0Vec: 正常文章词语出现概率向量
    :param p1Vec: 消极文章词语出现概率向量
    :param pClass1: 消极文章占比
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 这里没有计算分母的p(w)是因为无论哪个类别的p(w)都相同
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()