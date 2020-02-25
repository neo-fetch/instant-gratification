import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split # Import train_test_split function
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#train  =  train.drop(['id'], axis = 1)

#train.head()
Y = train['target']
from sklearn.feature_selection import VarianceThreshold
tst = test.drop(['id', 'wheezy-copper-turtle-magic'],axis=1)
model = QuadraticDiscriminantAnalysis(0.1)#LogisticRegression()#SVC(probability=True,kernel='poly',degree=4,gamma='auto')

res = np.zeros(len(train))
pree = np.zeros(len(test))
#if label is not ['id','wheezy-copper-turtle-magic','target']:
# check = []
# for label in train.columns:    
#     if label not in ['id','wheezy-copper-turtle-magic','target']:
#         check.append(label)
check = tst.columns
        
for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    index1 = train2.index
    index2 = test2.index
    print(index1)
    train2.reset_index(drop=True,inplace=True)
    lowvardrop = VarianceThreshold(threshold=1.5).fit(train2[check])
    trainupdted = lowvardrop.transform(train2[check])
    testupdted = lowvardrop.transform(test2[check])
    splits = 11
    X_train, X_test, Y_train, Y_test = train_test_split(trainupdted, train2['target'], test_size=0.1, random_state=42)
    model = model.fit(X_train,Y_train)
    pree[index2] += (model.predict_proba(testupdted)[:,1])/splits
#     folds = StratifiedKFold(n_splits=splits)
#     for traindex, testdex in folds.split(trainupdted, train2['target']):
#         print('hi')
#         print(len(traindex))
#         model.fit(trainupdted[traindex,:],train2.loc[traindex]['target'])
#         res[index1[testdex]] = model.predict_proba(trainupdted[testdex,:])[:,1]
#         pree[index2] += (model.predict_proba(testupdted)[:,1])/splits
lmao = pd.read_csv('../input/sample_submission.csv')
lmao['target'] = pree.reshape((-1,1))
lmao.to_csv('submission.csv', index = False)
