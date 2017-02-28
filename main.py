import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy as sp
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
    estim_array = [i for i in range(50, 200, 25)]
    rndst_array = [i for i in range(4, 30, 3)]
    rfc = ensemble.RandomForestClassifier()
    grid = GridSearchCV(rfc, param_grid={'n_estimators': estim_array, 'random_state': rndst_array})
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
def randomForestRes(ne, rs, X_train, X_test, y_train):
    rf = ensemble.RandomForestClassifier(n_estimators=ne, random_state=rs)
    rf.fit(X_train, y_train)
    zR=rf.predict_proba(X_test)
    y_res=open('y_testRndFor.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zR))
#RANDOM FOREST END

#GRADIENT BOOSTING BEGIN
def gboostSearchParam(X_train, y_train):
    estim_array = [i for i in range(50, 200, 10)]
    rndst_array = [i for i in range(4, 30, 2)]
    rfc = ensemble.GradientBoostingClassifier()
    grid = GridSearchCV(rfc, param_grid={'n_estimators': estim_array, 'random_state': rndst_array}, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print 'CV error    = ', 1 - grid.best_score_
    print 'best estim  = ', grid.best_estimator_.n_estimators
    print 'best rndst = ', grid.best_estimator_.random_state
    bestne=grid.best_estimator_.n_estimators
    bestrs=grid.best_estimator_.random_state
    return bestne, bestrc
def gboostTest(ne, rs, X_tr, X_lt, y_tr, y_lt):
    gbt = ensemble.GradientBoostingClassifier(n_estimators=ne, random_state=rs)
    gbt.fit(X_tr, y_tr)
    zb=gbt.predict_proba(X_lt)
    print "Boost: "+str(log_loss(y_lt, zb[:,1]))
def gboostRes(ne, rs, X_train, X_test, y_train):
    gbt = ensemble.GradientBoostingClassifier(n_estimators=ne, random_state=rs)
    gbt.fit(X_train, y_train)
    zB=gbt.predict_proba(X_test)
    y_res=open('y_testGB.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zB))   
#GRADIENT BOOSTING END

#SVC RADIAN KERNEL BEGIN
def svcRadianKerSearchParam(X_train, y_train):
    C_array = np.logspace(-3, 3, num=7)
    gamma_array = np.logspace(-5, 2, num=8)
    svc = SVC(kernel='rbf', probability=True)
    grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
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
    print "Rad: "+str(logloss(y_lt,zs[:,1]))
def svcRadianKerRes(lC, lGamma, X_train, X_test, y_train):
    svc = SVC(kernel='rbf', C=lC, gamma=lGamma, probability=True)
    svc.fit(X_train, y_train)
    zS=svc.predict_proba(X_test)
    y_res=open('y_testSVCRK.csv', 'w')
    y_res.write('\n'.join(str(v[1]) for v in zS))   
#SVC RADIAN KERNEK END

def main():
    X_train, y_train, X_test = importTrainData()
    X_train, X_test = normalizeData(X_train, X_test)
    X_tr, X_lt, y_tr, y_lt = makeLocalData(X_train, y_train)
    bestneRnd, bestrsRnd = 100, 22
    #bestneRnd, bestrsRnd = randomForestSearchParam(X_train, y_train)
    bestneGB, bestrsGB = 100, 13
    #bestneGB, bestrsGB = randomForestSearchParam(X_train, y_train)
    bestC, bestG = 100.0, 0.01
    #bestC, bestG = svcRadianKerSearchParam(X_train, y_train)
    randomForestTest(bestneRnd, bestrsRnd, X_tr, X_lt, y_tr, y_lt)
    gboostTest(bestneGB, bestrsGB, X_tr, X_lt, y_tr, y_lt)
    svcRadianKerTest(bestC, bestG, X_tr, X_lt, y_tr, y_lt)
    randomForestRes(bestneRnd, bestrsRnd, X_train, X_test, y_train)
    gboostRes(bestneGB, bestrsGB,  X_train, X_test, y_train)
    svcRadianKerRes(bestC, bestG, X_train, X_test, y_train)
    
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
