import pandas as pd
import numpy as np
import pingouin as pg
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


def shapreg(df, target, features, itnum=500, depth=3, indicator=True, norm= True, seed=123):

    np.random.seed(seed)
    coef_all_iter = dict((el,[]) for el in features)
    for i in range(itnum):
        if indicator == True:
            if (i+1)%500 == 0:
                print("| "+str(i+1))
            elif (i+1)%10 == 0:
                print("-", end="")
        shuffle_feature = np.random.choice(features, size=depth, replace=False, p=None)
        r2_pre = 0
        features_in_model = []
        for f in shuffle_feature:
            features_in_model.append(f)
            X,y = df[features_in_model].values, df[target].values
            model = LinearRegression()
            model.fit(X, y)
            r2_modell = r2_score(y ,model.predict(X))
            r2_post = r2_modell - r2_pre
            r2_pre = r2_modell
            coef_all_iter[f].append(r2_post)

    out = pd.DataFrame(dict([ (k,pd.Series(v, dtype="float64")) for k,v in coef_all_iter.items() ]))
    if norm == True:
        out = out.mean() / out.mean().sum()
    return out



def kruskal(df, target, features):
    coef_all_iter = dict((f,[]) for f in features)
    for a, b in itertools.product(features, repeat=2):
        if a != b:
            cor1 = float(pg.corr(x=df[a], y=df[target], method='pearson').r.values)**2
            cor2 = float(pg.partial_corr(data=df, x= b, y= target, covar=[a], method='pearson').r.values)**2
            coef_all_iter[a].append(cor1)
            coef_all_iter[b].append(cor2)
    out = pd.DataFrame(dict([ (k,pd.Series(v, dtype="float64")) for k,v in coef_all_iter.items() ]))
    out = out.mean() / out.mean().sum()
    return out
        

def seqreg(df, target, features, itnum=500, depth=3, indicator=True, seed=123):
    np.random.seed(seed)
    coef_all_iter = dict((el,[]) for el in features)
    for i in range(itnum):
        if indicator == True:
            if (i+1)%500 == 0:
                print("| "+str(i+1))
            elif (i+1)%10 == 0:
                print("-", end="")
        shuffle_feature = np.random.choice(features, size=depth, replace=False, p=None)
        features_in_model = []
        for f in shuffle_feature:
            features_in_model.append(f)
            #print(features_in_model)
            X,y = df[features_in_model].values, df[target].values
            model = LinearRegression()
            model.fit(X, y)
            coef = model.coef_
            for j, mf in enumerate(features_in_model):
                #print(coef, coef[j])
                coef_all_iter[mf].append(coef[j])

    out = pd.DataFrame(dict([ (k,pd.Series(v, dtype="float64")) for k,v in coef_all_iter.items() ]))
    out = out.mean()
    return out


def r2(df, target, features):
    X,y = df[features].values, df[target].values
    model = LinearRegression()
    model.fit(X, y)
    r2_modell = r2_score(y ,model.predict(X))
    return r2_modell

def rfimpo(df, target, features, itnum=100, depth=10, seed=123 ):
    X,y = df[features].values, df[target].values
    model = RandomForestRegressor(n_estimators=itnum, random_state=seed, max_features=depth)
    model.fit(X, y)
    
    out = model.feature_importances_ / sum(model.feature_importances_)
    out = pd.Series(out)
    out.index = features
    out.to_frame()
    return out

def has_missings(df, target, features):
    return df[[target]+features].isnull().values.any()


def imp_median(df, target, features):
    tofill = df[[target]+features].median()
    out = df[[target]+features].fillna(tofill)
    return out

def count_missings(df, target, features):
    return df[[target]+features].isnull().values.sum().sum()

