PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'is_unbalance': True,# Handle imbalance
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 63,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 7,
    'min_data_in_leaf': 40,
    'min_gain_to_split' : 0.0,
    'lambda_l1': 0.5,
    'lambda_l2': 0.1,
    'seed': 42
}
TEST_SIZE = 0.3

K_FOLD = 5