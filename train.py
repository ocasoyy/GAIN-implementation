# Experiment
from utils import *
from gain import *

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import pandas as pd
import tensorflow as tf

parser = ConfigParser()
_ = parser.read('config.ini')

BATCH_SIZE = int(parser.get('setting', 'batch_size'))
EPOCHS = int(parser.get('setting', 'epochs'))
HIDDEN_SIZE = int(parser.get('setting', 'hidden_size'))
DROP_RATE = float(parser.get('setting', 'drop_rate'))
LEARNING_RATE = float(parser.get('setting', 'learning_rate'))
ALPHA = int(parser.get('setting', 'alpha'))

# 데이터 준비
cancer = load_breast_cancer()
original = cancer.data
scaler = MinMaxScaler()
original = scaler.fit_transform(original)

# 기록용
report = pd.DataFrame({'Miss Rate': [], 'Hint Rate': [], 'Model RMSE': [], 'Random RMSE': []})

# 조정
MISSING_RATE = 0.5
HINT_RATES = [0.5]

# 결측치 만들기
na_ids = np.random.uniform(0, 1, size=[*original.shape])
na_ids = 1 * (na_ids < MISSING_RATE)
missing_data = original.copy()
missing_data[na_ids == 1] = np.nan
missing_data = pd.DataFrame(missing_data)


# Hint Rate 조정하여 학습
for hint_rate in HINT_RATES:
    data_mat, mask_mat = get_data_and_mask_mat(missing_data)
    dataset = tf.data.Dataset.from_tensor_slices((data_mat, mask_mat)).shuffle(
        missing_data.shape[0]*2).batch(BATCH_SIZE)

    model = GAIN(drop_rate=DROP_RATE, hidden_size=HIDDEN_SIZE, num_features=missing_data.shape[1])
    d_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    g_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCHS):
        d, g = [], []
        for data_batch, mask_batch in dataset:
            model, discriminator_loss, generator_loss = train_step(
                model, data_batch, mask_batch, d_opt, g_opt, hint_rate)
            d.append(discriminator_loss)
            g.append(generator_loss)

        print("Epoch: {}, Dis Loss: {:.4f}, Gen Loss: {:4f}".format(epoch, np.mean(d), np.mean(g)))

    # RMSE 계산
    data_array, X_hat_array, random_array, model_rmse, random_rmse = calculate_rmse(
        model, original, data_mat, mask_mat, hint_rate, False)

    # 기록
    report = report.append(pd.Series([MISSING_RATE, hint_rate, model_rmse, random_rmse],
                                     index=report.columns), ignore_index=True)


def return_accuracy(xgb, id, input):
    X, Y = input, cancer.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    xgb.fit(X_train, Y_train)
    Y_pred = xgb.predict(X_test)
    test_acc = np.round(accuracy_score(Y_test, Y_pred), 4)
    return id, test_acc


ids = ['원본', '0채움', '모델채움']
inputs = [original, data_array, X_hat_array]
accuracy_report = pd.DataFrame({'id': [], 'test_acc': []})

for id, input in zip(ids, inputs):
    xgb = XGBClassifier(random_state=7)

    id, test_acc = return_accuracy(xgb, id, input)
    accuracy_report = accuracy_report.append(
        pd.Series([id, test_acc], index=accuracy_report.columns), ignore_index=True)


print(accuracy_report)
