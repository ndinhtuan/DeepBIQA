from __future__ import division

import tensorflow as tf
import random
import pathlib

train_data_root = "/home/robot/IQA/pytorch-image-quality-param-ctrl/deepbiq/dataset/train1"
val_data_root = "/home/robot/IQA/pytorch-image-quality-param-ctrl/deepbiq/dataset/val1"
NUM_CLASSES = 5

def get_img_paths(data_root):
    data_root = pathlib.Path(data_root)
    all_data_paths = list(data_root.glob("*/*"))
    all_data_paths = [str(path) for path in all_data_paths]
    random.shuffle(all_data_paths)
    print("#Data: ", len(all_data_paths))

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir() )
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_img_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_data_paths]
    tmp = [[0 for i in range(NUM_CLASSES)] for j in range(len(all_img_labels))]
    print(tmp[0][0])
    for i in range(len(tmp)):
        tmp[i][all_img_labels[i]] = 1

    print(tmp[0:2])
    print("#Label: ", len(all_img_labels))
    return all_data_paths, tmp

def preprocess_image(img):
    print(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.random_crop(img, [224, 224, 3])
    print(img.dtype)
    img = tf.cast(img, tf.float32)
    img = tf.divide(img, tf.constant(255.0))

    return img

def load_and_preprocess_img(path):
    img = tf.read_file(path)
    return preprocess_image(img)

def gen_dataset(img_paths, label_imgs, batch_size=4):
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    image_ds = path_ds.map(load_and_preprocess_img)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_imgs, tf.int64))

    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(img_paths))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=len(img_paths))
    return ds

def load_and_get_iter_dataset(data_root, batch_size=16):
    data_paths, img_labels = get_img_paths(data_root)
    len_data = len(data_paths)
    ds = gen_dataset(data_paths, img_labels, batch_size)
    return ds.make_one_shot_iterator().get_next(), len_data

if __name__=="__main__":

    val_data_paths, val_img_labels = get_img_paths(val_data_root)
    print(val_data_paths)
    ds = gen_dataset(val_data_paths, val_img_labels)
    print(ds.output_shapes[0])
    iter = ds.make_one_shot_iterator()
    img, label = iter.get_next()
    print(img)

    with tf.Session() as sess:
        img, label = sess.run([img, label])
        print(img[0].shape)
        print(label)
