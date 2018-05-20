import catboost as cb
cat_features_index = [0,1,2,3,4,5,6]

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'cat_features': [cat_features_index],
         'iterations': [300]}

cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring='rmse', cv = 1)
cb_model.fit(train, y_train)