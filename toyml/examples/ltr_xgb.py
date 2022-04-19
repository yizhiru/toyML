import os

import xgboost as xgb
from sklearn.datasets import load_svmlight_file

from toyml.ltr import metrics, helper

root_path = 'ltrc_yahoo'
X_train, y_train, qid_train = load_svmlight_file(os.path.join(root_path, 'train.txt'), query_id=True)
X_vali, y_vali, qid_vali = load_svmlight_file(os.path.join(root_path, 'vali.txt'), query_id=True)
X_test, y_test, qid_test = load_svmlight_file(os.path.join(root_path, 'test.txt'), query_id=True)

print(f'train: {X_train.shape}, vali: {X_vali.shape}, test: {X_test.shape}')

group_train = helper.group_count(qid_train)
group_vali = helper.group_count(qid_vali)
group_test = helper.group_count(qid_test)

model = xgb.sklearn.XGBRanker(objective='rank:pairwise',
                              learning_rate=0.1,
                              gamma=1.,
                              max_depth=6,
                              min_child_weight=1,
                              n_estimators=100)
model.fit(X_train,
          y_train,
          group_train,
          eval_set=[(X_vali, y_vali)],
          eval_group=[group_vali]
          )
model.save_model('xgb.bin')

model = xgb.sklearn.XGBRanker()
model.load_model('xgb.bin')

y_pred = model.predict(X_test)
ndcg = metrics.tf_ndcg(y_test, y_pred, group_test, top_n=5)
print(f'ndcg@5: {ndcg}')
ndcg = metrics.tf_ndcg(y_test, y_pred, group_test, top_n=10)
print(f'ndcg@10: {ndcg}')
ndcg = metrics.tf_ndcg(y_test, y_pred, group_test, top_n=20)
print(f'ndcg@20: {ndcg}')
