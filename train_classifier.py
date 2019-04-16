import tensornets as nets
import tensorflow as tf
import cv2

def predict_resnet(img):

    _net = nets.ResNet50
    with tf.Graph().as_default():
        with tf.device("gpu:0"):
            _input = tf.placeholder(tf.float32, [None, 224, 224, 3])
            _model = _net(_input, is_training=False)

        with tf.Session() as sess:
            #nets.pretrained([_model])
            sess.run(_model.pretrained())
            feed_dict={_input: [img]}
            return sess.run(_model, feed_dict)[0]

if __name__=="__main__":
    img = cv2.imread("img.jpeg")
    img = cv2.resize(img, (224, 224))
    a = predict_resnet(img)
    print(len(a))
    print(sum(a))
