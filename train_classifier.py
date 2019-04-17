import tensornets as nets
import tensorflow as tf
import cv2
from load_data import load_and_get_iter_dataset, train_data_root

def train(num_classes, data_root):
    
    inputs = tf.placeholder(tf.float32, [None, 224,224,3])
    outputs = tf.placeholder(tf.float32, [None, num_classes])
    
    with tf.device('/gpu:0'):
        model = nets.ResNet50(inputs, is_training=True, classes=num_classes)
    
    loss = tf.losses.softmax_cross_entropy(outputs, model.logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
    
    iter_dataset = load_and_get_iter_dataset(data_root)
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(model.pretrained())

            for i in range(10):
                img, label = sess.run(iter_dataset)
                print(img)



def predict_resnet(img):

    _net = nets.ResNet50
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            _input = tf.placeholder(tf.float32, [None, 224, 224, 3])
            _model = _net(_input, is_training=False)
        
        graph = tf.get_default_graph()
        for op in graph.as_graph_def().node:
            print(op.name)
        node = graph.get_tensor_by_name("resnet50/logits/BiasAdd:0")
        print("Operations: ", )
        with tf.Session() as sess:
            #nets.pretrained(_model)
            sess.run(_model.pretrained())
            feed_dict={_input: [img]}
            print("Node: ", len(sess.run(node, feed_dict)[0]))
            return sess.run(_model, feed_dict)[0]

if __name__=="__main__":
    train(5, train_data_root)
    #print("Hello")
    #img = cv2.imread("img.jpeg")
    #img = cv2.resize(img, (224, 224))
    #a = predict_resnet(img)
    #print(len(a))
    #print(sum(a))
