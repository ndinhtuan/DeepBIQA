import cv2
import glob
import os
import shutil

def convert_format(root_data, dst_dir):
    
    label_dirs = glob.glob(root_data+"/*")
    imgs = []
    name_labels = []

    for label_dir in label_dirs:
        name_labels.append(label_dir.split('/')[-1])
        img_paths = glob.glob(label_dir+"/*")
        imgs.append(img_paths)

    print(name_labels)

    # create folder in dst_dir
    for name_label, img_paths in zip(name_labels, imgs):
        dst_label = dst_dir+'/'+name_label
        if os.path.exists(dst_label):
            shutil.rmtree(dst_label)
        
        os.makedirs(dst_label)

        for img_path in img_paths:
            img = cv2.imread(img_path)
            name_img = img_path.split('/')[-1].split('.')[0]
            cv2.imwrite(dst_label+'/{}.jpg'.format(name_img), img)

if __name__=="__main__":
    
    convert_format("/home/ndtuan/IQA/pytorch-image-quality-param-ctrl/deepbiq/dataset/val", "/home/ndtuan/IQA/pytorch-image-quality-param-ctrl/deepbiq/dataset/val1")
