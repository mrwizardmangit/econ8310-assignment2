import pandas as pd
from xgboost import XGBClassifier

training_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train_df = pd.read_csv(training_url)
test_df = pd.read_csv(test_url)

train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
train_df['hour'] = train_df['DateTime'].dt.hour
train_df['dayofweek'] = train_df['DateTime'].dt.dayofweek

test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
test_df['hour'] = test_df['DateTime'].dt.hour
test_df['dayofweek'] = test_df['DateTime'].dt.dayofweek

y_train = train_df['meal']
x_train = train_df.drop(columns=['meal', 'id', 'DateTime'])
x_test = test_df.drop(columns=['meal', 'id', 'DateTime'])

imbalance_ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

model = XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.2, 
    scale_pos_weight = imbalance_ratio,
    random_state=42,
    n_jobs=-1
)

modelFit = model.fit(x_train, y_train)

pred = modelFit.predict(x_test).astype(float).tolist()
