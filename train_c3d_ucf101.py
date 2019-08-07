import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from alisuretool.Tools import Tools
import tensorflow.contrib.slim as slim


class DataUCF101(object):

    NUM_CLASSES = 101
    CROP_SIZE = 112
    CHANNELS = 3
    FRAMES = 16

    def __init__(self, train_filename, test_filename, batch_size, crop_mean='./list/crop_mean.npy'):
        self.batch_size = batch_size

        self.train_lines = list(open(train_filename, 'r'))
        self.train_batch_num = len(self.train_lines) // self.batch_size

        self.test_lines = list(open(test_filename, 'r'))
        self.test_batch_num = len(self.test_lines) // self.batch_size

        self.np_mean = np.load(crop_mean).reshape([self.FRAMES, self.CROP_SIZE, self.CROP_SIZE, 3])
        pass

    def video_generator(self, is_train=True):
        lines = self.train_lines if is_train else self.test_lines
        video_indices = list(range(len(lines)))

        while True:
            if is_train:
                random.seed(time.time())
                random.shuffle(video_indices)

            batch_data, batch_label = [], []
            for video_index in video_indices:
                if len(batch_data) == self.batch_size:
                    yield (np.array(batch_data).astype(np.float32), np.array(batch_label).astype(np.int64))
                    batch_data, batch_label = [], []
                    pass

                dir_name, tmp_label = lines[video_index].strip('\n').split()
                tmp_data, _ = self.get_frames_data(dir_name, self.FRAMES)
                if len(tmp_data) != 0:
                    img_data = self.deal_data(tmp_data, self.CROP_SIZE, self.np_mean)
                    batch_data.append(img_data)
                    batch_label.append(int(tmp_label))
                    pass
                pass

            if not is_train:
                break
            pass
        pass

    @staticmethod
    def deal_data(tmp_data, crop_size, np_mean):
        img_data = []
        for j in range(len(tmp_data)):
            height, width = tmp_data[j].shape[0], tmp_data[j].shape[1]
            if width > height:
                scale = float(crop_size) / float(height)
                now_data = np.array(cv2.resize(tmp_data[j], (int(width * scale + 1), crop_size)), np.float32)
            else:
                scale = float(crop_size) / float(width)
                now_data = np.array(cv2.resize(tmp_data[j], (crop_size, int(height * scale + 1))), np.float32)

            crop_x, crop_y = int((now_data.shape[0] - crop_size) / 2), int((now_data.shape[1] - crop_size) / 2)
            now_data = now_data[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean[j]
            img_data.append(now_data)
            pass

        return img_data

    @staticmethod
    def get_frames_data(filename, frames):
        ret_arr = []
        s_index = 0
        for parent, _, file_names in os.walk(filename):
            if len(file_names) < frames:
                return [], s_index
            file_names = sorted(file_names)
            s_index = random.randint(0, len(file_names) - frames)
            for i in range(s_index, s_index + frames):
                image_name = os.path.join(filename, file_names[i])
                img_data = np.array(Image.open(image_name))
                ret_arr.append(img_data)
        return ret_arr, s_index

    pass


class C3D(object):

    @staticmethod
    def conv3d(name, l_input, w, b):
        return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name), b)

    @staticmethod
    def max_pool(name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

    @staticmethod
    def _variable_with_weight_decay(name, shape, wd):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('weight_decay_losses', weight_decay)
        return var

    @classmethod
    def inference_c3d(cls, _x, _dropout, class_num):

        with tf.variable_scope('var_name'):
            _weights = {
                'wc1': cls._variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                'wc2': cls._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': cls._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': cls._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': cls._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': cls._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': cls._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': cls._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                'wd1': cls._variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': cls._variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': cls._variable_with_weight_decay('wout', [4096, class_num], 0.0005)
            }
            _biases = {
                'bc1': cls._variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': cls._variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': cls._variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': cls._variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': cls._variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': cls._variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': cls._variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': cls._variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': cls._variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': cls._variable_with_weight_decay('bd2', [4096], 0.000),
                'out': cls._variable_with_weight_decay('bout', [class_num], 0.000),
            }
            pass

        # Convolution Layer
        conv1 = cls.conv3d('conv1', _x, _weights['wc1'], _biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = cls.max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = cls.conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = cls.max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = cls.conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = cls.conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = cls.max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = cls.conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = cls.conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = cls.max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = cls.conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = cls.conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = cls.max_pool('pool5', conv5, k=2)

        # Fully connected layer
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        dense1 = tf.nn.dropout(dense1, _dropout)

        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
        dense2 = tf.nn.dropout(dense2, _dropout)

        # Output: class prediction
        out = tf.matmul(dense2, _weights['out']) + _biases['out']

        return out

    pass


class TrainC3D(object):

    def __init__(self, run_name, batch_size, max_epochs, data, net, log_dir='./logs', model_dir="./models",
                 model_name="./c3d_ucf_model", pre_train=None):
        self.run_name = run_name
        self.model_name = model_name
        self.model_dir = Tools.new_dir(os.path.join(model_dir, self.run_name))
        self.checkpoint_path = os.path.join(self.model_dir, self.model_name)
        self.log_dir = Tools.new_dir(os.path.join(log_dir, self.run_name))
        self.pre_train = pre_train

        self.data = data
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # Input
        _shape = (batch_size, self.data.FRAMES, self.data.CROP_SIZE, self.data.CROP_SIZE, self.data.CHANNELS)
        self.images_placeholder = tf.placeholder(tf.float32, _shape)
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(self.batch_size,))

        # Net
        self.logits = net(self.images_placeholder, 0.5, self.data.NUM_CLASSES)

        # Output
        self.pred = tf.argmax(self.logits, 1)
        self.loss = self.cal_loss(self.labels_placeholder, self.logits)
        self.accuracy = self.cal_acc(self.pred, self.labels_placeholder)
        self.now_epoch = tf.Variable(tf.constant(0), trainable=False)
        self.learning_rate = tf.train.piecewise_constant(self.now_epoch, [5, 10], [0.003, 0.001, 0.0001])
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.summary_op = tf.summary.merge_all()

        # Sess
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        pass

    def train(self):
        self.load_model()
        self.test()

        video_generator = self.data.video_generator(is_train=True)
        for epoch in range(self.max_epochs):
            total_acc = 0
            total_loss = 0
            for step in range(self.data.train_batch_num):
                train_images, train_labels = next(video_generator)
                _loss, summary, acc, _, _pred, _learning_rate = self.sess.run(
                    [self.loss, self.summary_op, self.accuracy, self.train_op, self.pred, self.learning_rate],
                    feed_dict={self.images_placeholder: train_images, self.labels_placeholder: train_labels})
                total_acc += acc
                total_loss += _loss
                self.summary_writer.add_summary(summary, epoch * self.data.train_batch_num + step)

                if step % 10 == 0:
                    Tools.print("{}/{} {}/{} acc={:.5f} avg_loss={:.5f} loss={:.5f} lr={}".format(
                        epoch, self.max_epochs, step, self.data.train_batch_num,
                        total_acc/(step + 1), total_loss/(step + 1), _loss, _learning_rate))
                    Tools.print("Train preds {}".format(_pred))
                    Tools.print("Train label {}".format(train_labels))
                    pass
                pass
            self.saver.save(self.sess, self.checkpoint_path, global_step=epoch)
            self.test()
            self.sess.run(self.learning_rate, feed_dict={self.now_epoch: epoch})
            pass

        pass

    def test(self):
        step = 0
        total_acc = 0
        video_generator = self.data.video_generator(is_train=False)
        for step in range(self.data.test_batch_num):
            try:
                val_images, val_labels = next(video_generator)
                acc, _pred = self.sess.run([self.accuracy, self.pred], feed_dict={self.images_placeholder: val_images,
                                                                                  self.labels_placeholder: val_labels})
                total_acc += acc

                if step % 10 == 0:
                    Tools.print("Test {}/{} {:.5f}".format(step, self.data.test_batch_num, total_acc/(step + 1)))
                    Tools.print("Test preds {}".format(_pred))
                    Tools.print("Test label {}".format(val_labels))
            except StopIteration:
                break
            pass
        Tools.print("Test accuracy: {:.5f}".format(total_acc / step))
        pass

    @staticmethod
    def cal_acc(preds, labels):
        correct_pred = tf.equal(preds, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @staticmethod
    def cal_loss(labels, logits):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        weight_decay_loss = tf.reduce_mean(tf.get_collection('weight_decay_losses'))

        total_loss = cross_entropy_mean + weight_decay_loss
        # total_loss = cross_entropy_mean

        tf.summary.scalar('loss_cross_entropy', cross_entropy_mean)
        tf.summary.scalar('loss_weight_decay_loss', weight_decay_loss)
        tf.summary.scalar('loss_total_loss', total_loss)

        return total_loss

    def load_model(self):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        pre_train = ckpt.model_checkpoint_path if ckpt and ckpt.model_checkpoint_path else self.pre_train
        if pre_train:
            # tf.train.Saver(var_list=tf.global_variables()).restore(sess, ckpt.model_checkpoint_path)
            slim.assign_from_checkpoint_fn(pre_train, var_list=tf.global_variables(),
                                           ignore_missing_vars=True)(self.sess)
            Tools.print("Restored model parameters from {}".format(pre_train))
        else:
            Tools.print('No checkpoint file found.')
            pass
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 1
    _batch_size = 10
    _data = DataUCF101(train_filename='./list/train_1.list',
                       test_filename='./list/test_1.list', batch_size=_batch_size)
    TrainC3D(run_name="split_1", batch_size=_batch_size,
             max_epochs=50, data=_data, net=C3D.inference_c3d).train()
