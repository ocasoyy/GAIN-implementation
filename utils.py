import numpy as np
import tensorflow as tf
from configparser import ConfigParser

parser = ConfigParser()
_ = parser.read('config.ini')
ALPHA = int(parser.get('setting', 'alpha'))


def get_data_and_mask_mat(X):
    (n, p) = X.shape
    # 1) Data Matrix: 결측치를 0으로 채운 행렬
    data_mat = X.fillna(0.0).values

    # 2) Mask Matrix: 실제값 위치면 1, 결측치 위치면 0
    mask_mat = np.array(1 - np.isnan(X))

    # Conversion to Tensor
    data_mat = tf.convert_to_tensor(value=data_mat, dtype=tf.float32)
    mask_mat = tf.convert_to_tensor(value=mask_mat, dtype=tf.float32)
    return data_mat, mask_mat


def get_random_and_hint_mat_on_batch(data_batch, mask_batch, hint_rate):
    (n, p) = data_batch.shape

    # 1) Random Matrix: 균일 분포에서 추출한 Random 값
    random_batch = np.random.uniform(0, 1, size=[n, p])

    # 2) Hint Matrix: hint rate 만큼 mask batch 에서 discriminator 에게 힌트를 주는 행렬
    # 이 행렬의 값이 1이면 실제 값의 위치를, *0이면 알 수 없음을 의미함
    # *0: 논문에서는 이 값을 0, 0.5, 1 모두 사용해본 결과를 서술하고 있다.
    tmp = np.random.uniform(0, 1, size=[n, p])
    # B: 논문 4p 참조, hint rate 보다 작으면 1 (hint rate: 1이 될 확률)
    B = 1 * (tmp < hint_rate)

    # 위 무작위 추출 단계 때문에 원래 mask batch 보다 정보량이 감소한다.
    hint_batch = mask_batch * B + 0.0 * (1-B)

    random_batch = tf.convert_to_tensor(value=random_batch, dtype=tf.float32)
    hint_batch = tf.convert_to_tensor(value=hint_batch, dtype=tf.float32)
    return random_batch, hint_batch


def compute_discriminator_loss(mask_mat, estimated_mask_mat):
    # mask matrix M과 discriminator의 결과물인 estimate_mask_mat
    eps = 1e-8
    discriminator_loss = -tf.reduce_mean(
        tf.math.multiply(mask_mat, tf.math.log(estimated_mask_mat + eps)) +
        tf.math.multiply(1-mask_mat, tf.math.log(1.0 - estimated_mask_mat + eps)) )
    return discriminator_loss


def compute_generator_loss(mask_mat, estimated_mask_mat, data_mat, imputed_mat):
    # Reconsturcion Loss (Loss2는 MSE Loss)
    eps = 1e-8
    loss1 = -tf.reduce_mean(tf.math.multiply(1-mask_mat, tf.math.log(estimated_mask_mat + eps)))
    loss2 = tf.reduce_mean(
        tf.math.multiply(mask_mat, (tf.math.subtract(data_mat, imputed_mat))**2)) / tf.reduce_mean(mask_mat)
    generator_loss = loss1 + ALPHA*loss2
    return generator_loss


def train_step(model, data_batch, mask_batch, d_opt, g_opt, hint_rate):
    with tf.GradientTape() as tape:
        random_batch, hint_batch = get_random_and_hint_mat_on_batch(data_batch, mask_batch, hint_rate)
        data_with_random_batch = tf.math.multiply(
            data_batch, mask_batch) + tf.math.multiply(random_batch, (1-mask_batch))
        imputed_batch = model.generate(data_with_random_batch, mask_batch)
        X_hat = tf.math.multiply(data_batch, mask_batch) + tf.math.multiply(imputed_batch, 1-mask_batch)
        estimated_mask_batch = model.discriminate(X_hat, hint_batch)

        discriminator_loss = compute_discriminator_loss(mask_batch, estimated_mask_batch)
    d_gradients = tape.gradient(discriminator_loss, model.trainable_variables)
    d_opt.apply_gradients(zip(d_gradients, model.trainable_variables))

    for _ in range(2):
        with tf.GradientTape() as tape:
            random_batch, hint_batch = get_random_and_hint_mat_on_batch(data_batch, mask_batch, hint_rate)
            data_with_random_batch = tf.math.multiply(
                data_batch, mask_batch) + tf.math.multiply(random_batch, (1 - mask_batch))
            imputed_batch = model.generate(data_with_random_batch, mask_batch)
            X_hat = tf.math.multiply(data_batch, mask_batch) + tf.math.multiply(imputed_batch, 1 - mask_batch)
            estimated_mask_batch = model.discriminate(X_hat, hint_batch)

            generator_loss = compute_generator_loss(mask_batch, estimated_mask_batch, data_batch, imputed_batch)
        g_gradients = tape.gradient(generator_loss, model.trainable_variables)
        g_opt.apply_gradients(zip(g_gradients, model.trainable_variables))

    return model, discriminator_loss, generator_loss


def calculate_rmse(model, original_array, data_mat, mask_mat, hint_rate, verbose=True):
    # original_array, data_array, mask_array: np.array
    # 원본 행렬, 결측치를 0으로 채운 행렬, 마스크 행렬
    data_batch, mask_batch = tf.convert_to_tensor(data_mat), tf.convert_to_tensor(mask_mat)
    data_array, mask_array = data_batch.numpy(), mask_batch.numpy()

    random_batch, hint_batch = get_random_and_hint_mat_on_batch(data_batch, mask_batch, hint_rate)
    data_with_random_batch = tf.math.multiply(data_batch, mask_batch) + tf.math.multiply(random_batch, (1-mask_batch))

    imputed_batch = model.generate(data_with_random_batch, mask_batch)
    X_hat = tf.math.multiply(data_batch, mask_batch) + tf.math.multiply(imputed_batch, (1-mask_batch))

    X_hat_array = X_hat.numpy()
    model_rmse = np.sqrt((original_array - X_hat_array)**2).mean()

    random_array = np.random.uniform(0, 1, size=[*data_array.shape])
    # 원래 값은 그대로 두고, 결측값(0으로 채운)은 random 값으로 대치
    random_array = data_array*mask_array + random_array*(1-mask_array)
    random_rmse = np.sqrt((original_array - random_array )**2).mean()

    if verbose:
        print("Model RMSE: {:.4f}, Random RMSE: {:.4f}".format(model_rmse, random_rmse))

    return data_array, X_hat_array, random_array, model_rmse, random_rmse




