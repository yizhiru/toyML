import numpy as np
import xgboost as xgb
import xlearn as xl
from scipy import sparse


class GBDTLRClassifier:

    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(max_depth=5,
                                           learning_rate=0.01,
                                           subsample=0.9,
                                           colsample_bylevel=0.5,
                                           verbosity=2)
        self.lr_model = xl.LRModel(task='binary',
                                   init=0.1,
                                   epoch=100,
                                   lr=0.01,
                                   reg_lambda=0.1,
                                   opt='sgd',
                                   stop_window=4)

    def fit(self, X_train, y_train, X_test):
        self.xgb_model.fit(X_train, y_train)
        X_train = xgb.DMatrix(X_train)
        train_leaves = self.xgb_model.get_booster().predict(X_train, pred_leaf=True)
        X_test = xgb.DMatrix(X_test)
        test_leaves = self.xgb_model.get_booster().predict(X_test, pred_leaf=True)
        self.__check_leaves(train_leaves, test_leaves)
        xgb_feats = self.__transform_leaves(train_leaves)
        self.lr_model.fit(xgb_feats, y_train)

    def __check_leaves(self, train_leaves, test_leaves):
        leaves = np.concatenate((train_leaves, test_leaves), axis=0)
        self.col_max_values = np.max(leaves, axis=0)

    def __transform_leaves(self, leave_features):
        n_samples, n_trees = leave_features.shape
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_trees)
        indices = np.cumsum(np.hstack(([0], self.col_max_values + 1)))
        column_indices = (leave_features + indices[:-1]).ravel()
        data = np.ones(n_samples * n_trees)
        return sparse.coo_matrix((data, (row_indices, column_indices)),
                                 shape=(n_samples, indices[-1]),
                                 dtype=np.float64).tocsr()

    def predict(self, X_test):
        X_test = xgb.DMatrix(X_test)
        test_leaves = self.xgb_model.get_booster().predict(X_test, pred_leaf=True)
        xgb_feats = self.__transform_leaves(test_leaves)
        return self.lr_model.predict(xgb_feats)



