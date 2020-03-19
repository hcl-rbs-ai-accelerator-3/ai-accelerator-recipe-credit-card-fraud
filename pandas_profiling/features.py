# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def important_features(df,target):
    try:
        df.dropna(inplace =True)
        X_train = df.drop(target,axis=1)
        y_train = df[target]
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        feature_imp = pd.DataFrame.from_dict({'Features':X_train.columns,'Importance':clf.feature_importances_}).sort_values(by ='Importance', ascending=False).reset_index(drop=True)
        dfStyler = df.style.set_properties(**{'text-align': 'left'})
        dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return feature_imp
    except Exception as e:
        print('Exception occurred in imp_features() in data quality module')
        print(e)

    
