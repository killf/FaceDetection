import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, AvgPool2D


# Rapidly Digested Convolutional Layers
class RDCL(tf.keras.Model):
    def __init__(self):
        super(RDCL, self).__init__()

        self.conv1 = Conv2D(24, 7, strides=4, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(64, 5, strides=2, padding='same')
        self.bn2 = BatchNormalization()

    def __call__(self, x, training=False):
        x = tf.nn.crelu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME')

        x = tf.nn.crelu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME')

        return x


# Inception model
class Inception(tf.keras.Model):
    def __init__(self):
        super(Inception, self).__init__()

        self.conv1 = Conv2D(32, 1, strides=1, padding='same')
        self.bn1 = BatchNormalization()

        self.avg_pool = AvgPool2D((3, 3), strides=1, padding='same')
        self.conv2 = Conv2D(32, 1, strides=1, padding='same')
        self.bn2 = BatchNormalization()

        self.conv3_1 = Conv2D(24, 1, strides=1, padding='same')
        self.bn3_1 = BatchNormalization()

        self.conv3_2 = Conv2D(32, 3, strides=1, padding='same')
        self.bn3_2 = BatchNormalization()

        self.conv4_1 = Conv2D(24, 1, strides=1, padding='same')
        self.bn4_1 = BatchNormalization()

        self.conv4_2 = Conv2D(32, 3, strides=1, padding='same')
        self.bn4_2 = BatchNormalization()

        self.conv4_3 = Conv2D(32, 3, strides=1, padding='same')
        self.bn4_3 = BatchNormalization()

    def __call__(self, x, training=False):
        path1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))

        path2 = self.avg_pool(x)
        path2 = tf.nn.relu(self.bn2(self.conv2(path2), training=training))

        path3 = tf.nn.relu(self.bn3_1(self.conv3_1(x), training=training))
        path3 = tf.nn.relu(self.bn3_2(self.conv3_2(path3), training=training))

        path4 = tf.nn.relu(self.bn4_1(self.conv4_1(x), training=training))
        path4 = tf.nn.relu(self.bn4_2(self.conv4_2(path4), training=training))
        path4 = tf.nn.relu(self.bn4_3(self.conv4_3(path4), training=training))

        return tf.concat([path1, path2, path3, path4], axis=3)


# Multiple Scale Convolutional Layers
class MSCL(tf.keras.Model):
    def __init__(self):
        super(MSCL, self).__init__()

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = Conv2D(128, 1, strides=1, padding='same')
        self.bn3_1 = BatchNormalization()

        self.conv3_2 = Conv2D(256, 3, strides=2, padding='same')
        self.bn3_2 = BatchNormalization()

        self.conv4_1 = Conv2D(128, 1, strides=1, padding='same')
        self.bn4_1 = BatchNormalization()

        self.conv4_2 = Conv2D(256, 3, strides=2, padding='same')
        self.bn4_2 = BatchNormalization()

    def __call__(self, x, training=False):
        result = []

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        result.append(x)

        x = tf.nn.relu(self.bn3_1(self.conv3_1(x), training=training))
        x = tf.nn.relu(self.bn3_2(self.conv3_2(x), training=training))
        result.append(x)

        x = tf.nn.relu(self.bn4_1(self.conv4_1(x), training=training))
        x = tf.nn.relu(self.bn4_2(self.conv4_2(x), training=training))
        result.append(x)

        return result


class FaceBoxes(tf.keras.Model):
    def __init__(self):
        super(FaceBoxes, self).__init__()

        self.RDCL = RDCL()
        self.MSCL = MSCL()

        self.conv_loc_1 = Conv2D(4 * 21, 3, strides=1, padding='same')
        self.conv_conf_1 = Conv2D(2 * 21, 3, strides=1, padding='same')

        self.conv_loc_2 = Conv2D(4 * 1, 3, strides=1, padding='same')
        self.conv_conf_2 = Conv2D(2 * 1, 3, strides=1, padding='same')

        self.conv_loc_3 = Conv2D(4 * 1, 3, strides=1, padding='same')
        self.conv_conf_3 = Conv2D(2 * 1, 3, strides=1, padding='same')

    def __call__(self, x, training=False):
        batch_size = x.shape[0]

        x = self.RDCL(x, training=training)
        o1, o2, o3 = self.MSCL(x, training=training)

        loc_1 = self.conv_loc_1(o1)
        loc_1 = tf.reshape(loc_1, (batch_size, -1, 4))  # (-1, 32 * 32 * 21, 4)

        loc_2 = self.conv_loc_2(o2)
        loc_2 = tf.reshape(loc_2, (batch_size, -1, 4))  # (-1, 16 * 16 * 1, 4)

        loc_3 = self.conv_loc_3(o3)
        loc_3 = tf.reshape(loc_3, (batch_size, -1, 4))  # (-1, 8 * 8 * 1, 4)

        loc = tf.concat([loc_1, loc_2, loc_3], axis=1)

        conf_1 = self.conv_conf_1(o1)
        conf_1 = tf.reshape(conf_1, (batch_size, -1, 2))  # (-1, 32 * 32 * 21, 2)

        conf_2 = self.conv_conf_2(o2)
        conf_2 = tf.reshape(conf_2, (batch_size, - 1, 2))  # (-1, 16 * 16 * 1, 2)

        conf_3 = self.conv_conf_3(o3)
        conf_3 = tf.reshape(conf_3, (batch_size, -1, 2))  # (-1, 8 * 8 * 1, 2)

        conf = tf.concat([conf_1, conf_2, conf_3], axis=1)

        return loc, conf


if __name__ == '__main__':
    for device in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    input = tf.zeros((2, 1024, 1024, 3))
    model = FaceBoxes()
    output = model(input)
