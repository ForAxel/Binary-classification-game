import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy
from tqdm import tqdm_notebook as tqdm
from scipy.sparse import *
from sklearn.base import TransformerMixin
from itertools import repeat
import scipy

df_train=pd.read_csv("DataSet/train.csv")
df_test=pd.read_csv("DataSet/test.csv")

df_all=df_train.append(df_test, sort=False)
num_train=len(df_train)
df_all.head()

# drop_cols=["bin_0"]
drop_cols=[]

df_all["ord_5a"]=df_all["ord_5"].str[0]
df_all["ord_5b"]=df_all["ord_5"].str[1]
drop_cols.append("ord_5")

for col in ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]:
    train_vals = set(df_train[col].unique())
    test_vals = set(df_test[col].unique())
   
    xor_cat_vals=train_vals ^ test_vals
    if xor_cat_vals:
        df_all.loc[df_all[col].isin(xor_cat_vals), col]="xor"

X=df_all[df_all.columns.difference(["id", "target"] + drop_cols)]

X_oh=X[X.columns.difference(["ord_1", "ord_4", "ord_5a", "ord_5b", "day", "month"])]
oh1=pd.get_dummies(X_oh, columns=X_oh.columns, drop_first=True, sparse=True)
# ohc1=oh1.sparse.to_coo()
ohc1=oh1.to_coo()

'''对于部分有序特征进行编码'''
class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        possible_values = sorted(self.value_map_.values())
        idx1 = []
        idx2 = []
        all_indices = np.arange(len(X))
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
        return result

thermos=[]
for col in ["ord_1", "ord_2", "ord_3", "ord_4", "ord_5a", "day", "month"]:
    if col=="ord_1":
        sort_key=['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'].index
    elif col=="ord_2":
        sort_key=['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'].index
    elif col in ["ord_3", "ord_4", "ord_5a"]:
        sort_key=str
    elif col in ["day", "month"]:
        sort_key=int
    else:
        raise ValueError(col)
    
    enc=ThermometerEncoder(sort_key=sort_key)
    thermos.append(enc.fit_transform(X[col]))

ohc=scipy.sparse.hstack([ohc1] + thermos).tocsr()

X_train = ohc[:num_train]
X_test = ohc[num_train:]
y_train = df_train["target"].values

'''模型的构建，选用三个不同参数的逻辑回归进行预测取平均值'''
clf1=LogisticRegression(C=0.15, solver="lbfgs", max_iter=1200)  # MODEL
clf1.fit(X_train, y_train)
pred1=clf1.predict_proba(X_test)[:,1]

clf2=LogisticRegression(C=0.15, solver="sag", max_iter=1200)  # MODEL
clf2.fit(X_train, y_train)
pred2=clf2.predict_proba(X_test)[:,1]

clf3=LogisticRegression(C=0.15, solver="saga", max_iter=1200)  # MODEL
clf3.fit(X_train, y_train)
pred3=clf3.predict_proba(X_test)[:,1]

'''生成最终用于提交的文件'''
ans=(pred1+pred2+pred3)/3
pd.DataFrame({"id": df_test["id"], "target": ans}).to_csv("submission.csv", index=False)