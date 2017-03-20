from numpy import *

"""从文件中读取数据"""
def loadDataSet(filename):
    fr = open(filename);
    dataMat = [];
    lableMat = [];
    fileLines = fr.readlines();
    for line in fileLines:
        lineArr = line.strip().split('\t');
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[-1]))
    return dataMat,lableMat

'''从0-m中随机选择一个不等于i的数'''
def selectJrand(i,m):
    j=i;
    while (j==i):
        j = int(random.uniform(0,m));
    return j

'''ALPHA的剪辑函数'''
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H;
    elif aj < L:
        aj = L;
    return aj;

'''简化版的SMO算法,外层循环不进行寻优处理'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):

    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0

    while (iter < maxIter): #连续出现maxIter次未修改则退出循环
        alphaPairsChanged = 0;
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b; #利用矩阵乘法实现 F(Xi)= j从1到N求和(aj)*(yj)[(xj)*(xi)]+b
            Ei = fXi - float(labelMat[i]);
            if( ( (labelMat[i]*Ei < -toler) and (alphas[i] != C) ) or ( ( labelMat[i]*Ei > toler ) and ( alphas[i]!=0 ) ) ):
                j = selectJrand(i,m);
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b;
                Ej = fXj - float(labelMat[j]);

                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();

                if( labelMat[i]!=labelMat[j] ):
                    L = max(0,alphas[j]-alphas[i]);
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C);
                    H = min(C,alphas[j]+alphas[i]);

                if L==H :
                    #print("L==H");
                    continue;
                eta = dataMatrix[i,:]*dataMatrix[i,:].T + dataMatrix[j,:]*dataMatrix[j,:].T - 2.0*dataMatrix[i,:]*dataMatrix[j,:].T;

                if( eta<=0 ):
                    #print("eta>=0");
                    continue;
                alphas[j] = alphas[j] + labelMat[j] * (Ei-Ej)/eta; #得到未经剪辑的alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L);#剪辑后的alphas

                if( abs(alphas[j] - alphaJold)<0.00001 ) :
                    #print("J is not moving enough")
                    continue;

                alphas[i] = alphas[i] + labelMat[j]*labelMat[i]*(alphaJold-alphas[j]);

                b1 = b - Ei-labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*labelMat[j,:]*labelMat[i,:].T;
                b2 = b - Ej-labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*labelMat[j,:]*labelMat[j,:].T;

                if ( (0 < alphas[i]) and ( alphas[i] < C )): b = b1;
                elif ( (0 < alphas[j] ) and ( alphas[j]<C)): b = b2;
                else: b = (b1+b2)/2;
                alphaPairsChanged += 1;

        if(alphaPairsChanged == 0): iter +=1;
        else: iter = 0;

    return b,alphas;

def kernelTrans(X,A):
    m,n = shape(X)
    k = mat(zeros((m,1)))
    for i in range(m):
        deltaRow = X[i,:] - A;
        sq = deltaRow*deltaRow.T
        k[i] = exp(sq/(-2*1.3))
    return k

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:])


def calcEk(os,k):
    #fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b;  # 利用矩阵乘法实现 F(Xi)= j从1到N求和(aj)*(yj)[(xj)*(xi)]+b
    #Ei = fXi - float(labelMat[i]);
    # fXk = float(multiply(os.alphas, os.labelMat).T * (os.X * os.X[k, :].T)) + os.b;
    fXk = float(multiply(os.alphas, os.labelMat).T * (os.K[:,k]) ) + os.b;
    Ek =  fXk - float(os.labelMat[k]);
    return Ek

'''内循环选择方式，当i确定后，选择j的方式'''
'''以下代码并没有完整表达出SMO的思想'''
def selectJ(i,os,Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    os.eCache[i] = [1,Ei]
    validEcacheList = nonzero(os.eCache[:,0].A)[0]
    if (len(validEcacheList) > 1 ):
        for k in validEcacheList:
            if k==i: continue;
            Ek = calcEk(os,k)
            deltaE = abs(Ei-Ek)
            if maxDeltaE < deltaE :
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,os.m)
        Ej = calcEk(os,j)

    return j,Ej

def updateEk(os,k):
    Ek = calcEk(os,k)
    os.eCache[k] = [1,Ek]

'''对内层点进行优化'''
def innerL(i,os):
    Ei = calcEk(os,i)

    if ( ( (os.labelMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C) ) or ( (os.labelMat[i]*Ei > os.tol) and (os.alphas[i] > 0)) ):
        j,Ej = selectJ(i,os,Ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()

        if(os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0,os.alphas[j] + os.alphas[i]-os.C)
            H = min(os.C,os.alphas[j] + os.alphas[i])

        if L==H: return 0;
        # eta = os.X[i,:]*os.X[i,:].T + os.X[j,:]*os.X[j,:].T-2.0*os.X[i,:]*os.X[j,:].T

        eta = os.K[i,i] + os.K[j,j] - 2.0*os.K[i,j]

        if(eta <= 0) : return 0
        os.alphas[j] = os.alphas[j] + os.labelMat[j]*(Ei-Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j],H,L)

        updateEk(os,j)
        if (abs(os.alphas[j] - alphaJold) < 0.000001 ):
            return 0;
        os.alphas[i] = os.alphas[i] + os.labelMat[i] * os.labelMat[j]*(alphaJold - os.alphas[j])
        updateEk(os,i)

        b1 = os.b - Ei- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i] - os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
        b2 = os.b - Ej- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j] - os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]

        # b1 = os.b - Ei- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[i,:].T - os.labelMat[j]*(os.alphas[j]-alphaJold)*os.X[i,:]*os.X[j,:].T
        # b2 = os.b - Ej- os.labelMat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[j,:].T - os.labelMat[j]*(os.alphas[j]-alphaJold)*os.X[j,:]*os.X[j,:].T

        if ( 0 < os.alphas[i]) and (os.alphas[i] < os.C ):os.b = b1
        elif(0< os.alphas[j] ) and (os.alphas[j] < os.C ): os.b = b2
        else: os.b =(b1+b2)/2.0
        return 1
    else: return 0

def smoP(dataMataIn,classLabels,C,toter,maxIter,kTup=('lin',0)):
    os = optStruct(mat(dataMataIn),mat(classLabels).transpose(),C,toter)
    iter = 0
    entireSet = True; alphaPairsChanged = 0

    while((iter < maxIter) and ( (alphaPairsChanged > 0 ) or (entireSet) ) ):#超过最大迭代次数 或者 （没有修改并且全局）
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i,os)
            iter+=1
        else:
            nonBounds = nonzero( (os.alphas.A>0)*(os.alphas.A < os.C))[0]
            for i in nonBounds:
                alphaPairsChanged += innerL(i,os)
            iter+=1
        if entireSet: entireSet = False;
        elif (alphaPairsChanged == 0):entireSet = True

    return os.b,os.alphas

def calcWs(os):
    w = mat(multiply(os.alphas,os.labelMat)).T * os.X
    return w

def testRBF():
    dataArr,classArr = loadDataSet("testSetRBF.txt")
    b,alphas = smoP(dataArr,classArr,200,0.00001,1000000)
    dataMat = mat(dataArr);lableMat = mat(classArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    lableSV = lableMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCounts = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:])
        predict = kernelEval.T*multiply(lableSV,alphas[svInd])+ b
        if sign(predict) != sign(classArr[i]): errorCounts += 1
    print("the training error rate is: %f" % (float(errorCounts) / m) )

    dataArr, classArr = loadDataSet("testSetRBF2.txt")
    errorCounts = 0
    dataMat = mat(dataArr)
    lableMat = mat(classArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:])
        predict = kernelEval.T * multiply(lableSV, alphas[svInd]) + b
        if sign(predict) != sign(classArr[i]): errorCounts += 1
    print("the test error rate is: %f" % (float(errorCounts) / m))

'''图片转向量'''
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

