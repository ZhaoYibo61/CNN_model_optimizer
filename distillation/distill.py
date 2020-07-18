import os
import numpy as np
import tensorflow as tf
from utils.tools import load_checkpoint_model


class Distillation(object):

    def __init__(self, sess_student_dir, sess_teacher_dir, image="D:/Downloads/DIV2K_train_HR", temp=20, gama=0.5,
                 epoch=20, batch=10):
        self.temp = temp
        self.gama = gama
        self.epoch = epoch
        self.batch_size = batch
        self.images = image
        self.sess_student = load_checkpoint_model(sess_student_dir)
        self.sess_teacher = load_checkpoint_model(sess_teacher_dir)

    def get_total_loss(self, logit_name="student/logit/BiasAdd:0",
                       loss_name="student/softmax_cross_entropy_loss/value:0"):
        with self.sess_student.graph.as_default():
            # label_soft
            logit_teacher = tf.placeholder(dtype=tf.float32, shape=[None, None], name="logit_teacher")
            print(logit_teacher)
            label_soft = tf.nn.softmax(logit_teacher / self.temp)
            # logit_soft
            logit_student = tf.get_default_graph().get_tensor_by_name(logit_name)
            logit_soft = logit_student / self.temp
            # loss_1
            loss_1 = self.gama * tf.losses.softmax_cross_entropy(label_soft, logit_soft)
            # loss_2
            loss_2 = (1 - self.gama) * tf.get_default_graph().get_tensor_by_name(loss_name)
            # loss_total
            loss = loss_1 + loss_2
            #
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            train_op = optimizer.minimize(loss)
            self.sess_student.run(tf.variables_initializer(optimizer.variables()))
            return train_op

    def picture_read(self, data_file, height=224, width=224):
        """
        :param data_file:
        :param height:
        :param width:
        :return:
        """
        file_name = os.listdir(data_file)
        file_list = [os.path.join(data_file, name) for name in file_name]
        file_queue = tf.train.string_input_producer(file_list, shuffle=False)
        reader = tf.WholeFileReader()
        key, value = reader.read(file_queue)
        image = tf.image.decode_png(value)
        image_resize = tf.image.resize_images(image, [height, width])
        image_resize.set_shape([height, width, 3])
        image_resize_batch = tf.train.batch([image_resize], batch_size=self.batch_size, num_threads=10, capacity=32)
        return image_resize_batch

    def train(self,
              teacher_logit_name="teacher/logit/BiasAdd:0",
              teacher_input_name="teacher/input:0",
              student_input_name="student/input:0",
              student_label_name="student/label:0"):

        train_op = self.get_total_loss()

        teacher_logit = self.sess_teacher.graph.get_tensor_by_name(teacher_logit_name)
        teacher_input = self.sess_teacher.graph.get_tensor_by_name(teacher_input_name)
        student_label = self.sess_student.graph.get_tensor_by_name(student_label_name)
        student_input = self.sess_student.graph.get_tensor_by_name(student_input_name)

        teacher = self.sess_student.graph.get_tensor_by_name("logit_teacher:0")

        with tf.Session() as sess:
            image_batch = self.picture_read(self.images)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(self.epoch):
                for tem in range(self.batch_size):
                    print("epoch:{}-batch:{}".format(i, tem))
                    np_images_raw = np.squeeze(sess.run([image_batch]))
                    label = np.eye(10)
                    teacher_predict = self.sess_teacher.run(teacher_logit, feed_dict={teacher_input: np_images_raw})
                    self.sess_student.run(train_op, feed_dict={student_input: np_images_raw, student_label: label,
                                                               teacher: teacher_predict})
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    distill = Distillation(sess_student_dir="./student/", sess_teacher_dir="./teacher/")
    distill.train()
