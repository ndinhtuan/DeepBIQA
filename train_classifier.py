import tensornets as nets
import tensorflow as tf
import cv2
from load_data import load_and_get_iter_dataset, train_data_root, val_data_root
import numpy as np
import shutil
import os
from tensorflow.contrib.layers import fully_connected as fc

def train(num_classes, train_data_root, val_data_root, epochs=100, log_dir="log_summary"):

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    best_acc = 0.0

    inputs = tf.placeholder(tf.float32, [None, 224,224,3])
    outputs = tf.placeholder(tf.float32, [None, num_classes])

    with tf.device('/gpu:0'):
        model = nets.ResNet50(inputs, is_training=True, classes=num_classes)

    f = model.get_outputs()[-3]
    #print(model.get_outputs())
    f1 = tf.keras.layers.Dense(200, activation='relu')(f)
    f2 = tf.keras.layers.Dense(5, activation='relu')(f1)
    print(f, f2)
    loss = tf.losses.softmax_cross_entropy(outputs, model.get_outputs()[-1])
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(3e-4, global_step,
                                           30, 0.9, staircase=True)
    #train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    init=tf.global_variables_initializer()

    iter_dataset, len_data = load_and_get_iter_dataset(train_data_root)
    val_iter_dataset, val_len_data = load_and_get_iter_dataset(val_data_root)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'best_validation')


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        with tf.device("/gpu:0"):
            sess.run(init)
            sess.run(model.pretrained())
            summary = tf.Summary()
            summary1 = tf.Summary()
            step = 0

            for j in range(epochs):

                for i in range(int(np.ceil(len_data/16))):
                    step += 1
                    img, label = sess.run(iter_dataset)
                    print(len(img))
                    _loss, _ = sess.run([loss, train], {inputs: img, outputs: label})
                    summary.value.add(tag='loss/train', simple_value=_loss)
                    summary_writer.add_summary(summary, step)
                    print("Loss: ", _loss)
                acc = val(5, val_iter_dataset, val_len_data, model, sess, inputs)
                if best_acc < acc:
                    bes_acc = acc
                    saver.save(sess=sess, save_path=save_path)

                summary1.value.add(tag='acc/test', simple_value=acc)
                summary_writer.add_summary(summary1, j)


def val(num_classes, iter_dataset, len_data, model, sess, inputs):

    num_rights = 0
    num_samples = 0

    with tf.device("/gpu:0"):

        for i in range(int(np.ceil(len_data/16))):
            img, label = sess.run(iter_dataset)
            softmax = sess.run(model.get_outputs()[-1], {inputs: img})
            num_rights += sum(np.argmax(softmax, axis=1)==np.argmax(label, axis=1))
            num_samples += len(softmax)
            #print(mi)

    acc = num_rights/num_samples
    print(acc)
    return acc


def predict_resnet(img):

    _net = nets.ResNet50
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            _input = tf.placeholder(tf.float32, [None, 224, 224, 3])
            _model = _net(_input, is_training=False)

        graph = tf.get_default_graph()
        for op in graph.as_graph_def().node:
            print(op.name)
        node = graph.get_tensor_by_name("resnet50/avgpool:0")
        print("Operations: ", node)
        with tf.Session() as sess:
            #nets.pretrained(_model)
            sess.run(_model.pretrained())
            feed_dict={_input: [img]}
            print("Node: ", len(sess.run(node, feed_dict)[0]))
            return sess.run(_model, feed_dict)[0]

if __name__=="__main__":
    train(5, train_data_root, val_data_root)
    #print("Hello")
    #img = cv2.imread("img.jpeg")
    #img = cv2.resize(img, (224, 224))
    #a = predict_resnet(img)
    #print(len(a))
    #print(sum(a))
