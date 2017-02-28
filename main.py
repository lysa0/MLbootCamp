import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy as sp
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

#PREPARE DATA BEGIN
def importTrainData():
    X_train = pd.read_csv('x_train.csv', delimiter=';')
    y_train=pd.read_csv('y_train.csv', header=None)[0]
    X_test=pd.read_csv('x_test.csv', delimiter=';')
    return X_train, y_train, X_test
def normalizeData(X_train, X_test):
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean())/X_test.std()
    return X_train, X_test
def makeLocalData(X_train, y_train):
    X_tr, X_lt, y_tr, y_lt = train_test_split(X_train, y_train, test_size = 0.3, random_state = 11)
    return X_tr, X_lt, y_tr, y_lt
#PREPARE DATA END

#RANDOM FOREST BEGIN
def randomForestSearchParam(X_train, y_train):
    estim_array = [i for i in range(100, 200, 10)]
    rndst_array = [i for i in range(7, 30, 4)]
    rfc = ensemble.RandomForestClassifier()
    grid = GridSearchCV(rfc, param_grid={'n_estimators': estim_array, 'random_state': rndst_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print 'CV error    = ', 1 - grid.best_score_
    print 'best estim  = ', grid.best_estimator_.n_estimators
    print 'best rndst = ', grid.best_estimator_.random_state
    bestne=grid.best_estimator_.n_estimators
    bestrs=grid.best_estimator_.random_state
    return bestne, bestrs
def randomForestTest(ne, rs, X_tr, X_lt, y_tr, y_lt):
    rf = ensemble.RandomForestClassifier(n_estimators=ne, random_state=rs)
    rf.fit(X_tr, y_tr)
    zr=rf.predict_proba(X_lt)
    print 'Rnd: '+str(log_loss(y_lt, zr[:,1]))
    return zr[:,1]
def randomForestRes(ne, rs, X_train, X_test, y_train):
    rf = ensemble.RandomForestClassifier(n_estimators=ne, random_state=rs)
    rf.fit(X_train, y_train)
    zR=rf.predict_proba(X_test)
    y_res=open('y_testRndFor.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zR))
    return zR[:,1]
#RANDOM FOREST END

#GRADIENT BOOSTING BEGIN
def gboostSearchParam(X_train, y_train):
    estim_array = [i for i in range(100, 200, 15)]
    rndst_array = [i for i in range(7, 30, 4)]
    rfc = ensemble.GradientBoostingClassifier()
    grid = GridSearchCV(rfc, param_grid={'n_estimators': estim_array, 'random_state': rndst_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print 'CV error    = ', 1 - grid.best_score_
    print 'best estim  = ', grid.best_estimator_.n_estimators
    print 'best rndst = ', grid.best_estimator_.random_state
    bestne=grid.best_estimator_.n_estimators
    bestrs=grid.best_estimator_.random_state
    return bestne, bestrs
def gboostTest(ne, rs, X_tr, X_lt, y_tr, y_lt):
    gbt = ensemble.GradientBoostingClassifier(n_estimators=ne, random_state=rs)
    gbt.fit(X_tr, y_tr)
    zb=gbt.predict_proba(X_lt)
    print "Boost: "+str(log_loss(y_lt, zb[:,1]))
    return zb[:,1]
def gboostRes(ne, rs, X_train, X_test, y_train):
    gbt = ensemble.GradientBoostingClassifier(n_estimators=ne, random_state=rs)
    gbt.fit(X_train, y_train)
    zB=gbt.predict_proba(X_test)
    y_res=open('y_testGB.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zB))   
    return zB[:,1]
#GRADIENT BOOSTING END

#SVC RADIAN KERNEL BEGIN
def svcRadianKerSearchParam(X_train, y_train):
    C_array = np.logspace(-3, 3, num=10)
    gamma_array = np.logspace(-5, 2, num=10)
    svc = SVC(kernel='rbf', probability=True)
    grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print 'CV error    = ', 1 - grid.best_score_
    print 'best C      = ', grid.best_estimator_.C
    print 'best gamma  = ', grid.best_estimator_.gamma
    bestC=grid.best_estimator_.C
    bestG=grid.best_estimator_.gamma
    return bestC, bestG
def svcRadianKerTest(lC, lGamma, X_tr, X_lt, y_tr, y_lt):
    svc = SVC(kernel='rbf', C=lC, gamma=lGamma, probability=True)
    svc.fit(X_tr, y_tr)
    zs=svc.predict_proba(X_lt)
    print "Rad: "+str(log_loss(y_lt,zs[:,1]))
    return zs[:,1]
def svcRadianKerRes(lC, lGamma, X_train, X_test, y_train):
    svc = SVC(kernel='rbf', C=lC, gamma=lGamma, probability=True)
    svc.fit(X_train, y_train)
    zS=svc.predict_proba(X_test)
    y_res=open('y_testSVCRK.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zS))   
    return zS[:,1]
#SVC RADIAN KERNEL END

#XGB BEGIN
def xgboostSearchParam():
    bestParam={}
    return bestParam
def xgboostTest(X_tr, X_lt, y_tr, y_lt, param):
    if (not param):
        param = {'max_depth':4,'silent':1, 'objective':'binary:logistic'}
    bst = XGBClassifier(**param)
    bst.fit(X_tr, y_tr)
    zg = bst.predict_proba(X_lt)
    #print zg
    print "XGB: "+str(log_loss(y_lt, zg[:,1]))
    return zg[:,1]
def xgboostRes(X_train, X_test, y_train, param):
    if (not param):
        param = {'max_depth':4, 'silent':1, 'objective':'binary:logistic'}
    bst = XGBClassifier(**param)
    bst.fit(X_train, y_train)
    zG = bst.predict_proba(X_test)
    y_res=open('y_testXGB.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zG))      
    return zG[:,1]
#XGB END

def GBXGBTest(lab1, lab2, lab_lt):
    for i in range(len(lab1)):
        lab1[i]=(lab1[i]+lab2[i]*2)/3.
    print "GBXGB: "+str(log_loss(lab_lt, lab1))
def GBXGBRes(lab1, lab2):
    for i in range(len(lab1)):
        lab1[i]=(lab1[i]+lab2[i]*2)/3.
    y_res=open('y_testGBXGB.csv', 'w')
    y_res.write('\n'.join(str(v) for v in lab1))   

def GBXGBRKTest(lab1, lab2, lab3, lab_lt):
    for i in range(len(lab1)):
        lab1[i]=(lab1[i]+lab2[i]*2+lab3[i])/4.
    print "GBXGBRK: "+str(log_loss(lab_lt, lab1))

def RNDGBXGBTest(lab1, lab2, lab3, lab_lt):
    for i in range(len(lab1)):
        lab1[i]=(lab1[i]*3+lab2[i]*5+lab3[i])/9.
    print "RNDGBXGB: "+str(log_loss(lab_lt, lab1))

def main():
    X_train, y_train, X_test = importTrainData()
    X_train, X_test = normalizeData(X_train, X_test)
    X_tr, X_lt, y_tr, y_lt = makeLocalData(X_train, y_train)
    bestneRnd, bestrsRnd = 175, 22
    #bestneRnd, bestrsRnd = randomForestSearchParam(X_train, y_train)
    bestneGB, bestrsGB = 100, 16
    #bestneGB, bestrsGB = gboostSearchParam(X_train, y_train)
    bestC, bestG = 100.0, 0.001
    #bestC, bestG = svcRadianKerSearchParam(X_train, y_train)
    zRnd = randomForestTest(bestneRnd, bestrsRnd, X_tr, X_lt, y_tr, y_lt)
    zGb = gboostTest(bestneGB, bestrsGB, X_tr, X_lt, y_tr, y_lt)
    zSvc = svcRadianKerTest(bestC, bestG, X_tr, X_lt, y_tr, y_lt)
    zXgb = xgboostTest(X_tr, X_lt, y_tr, y_lt, {})
    zRND = randomForestRes(bestneRnd, bestrsRnd, X_train, X_test, y_train)
    zGB = gboostRes(bestneGB, bestrsGB,  X_train, X_test, y_train)
    #zSVC = svcRadianKerRes(bestC, bestG, X_train, X_test, y_train)
    zXGB = xgboostRes(X_train, X_test, y_train, {}) 
    GBXGBTest(zGb, zXgb, y_lt)
    GBXGBRes(zGB, zXGB)
    GBXGBRKTest(zGb, zXgb, zSvc, y_lt) 
    RNDGBXGBTest(zGb, zXgb, zRnd, y_lt)

main()

'''
for i in range(len(zr)):
    zr[i,1]=(zb[i,1]+zs[i,1])/2.
print log_loss(y_test, zr[:,1])
y_res=open('y_test.csv','w')
for i in range(len(zR)):
    zR[i,1]=(zB[i,1]+zS[i,1])/2.
y_res.write('\n'.join(str(v[1]) for v in zR))
'''
