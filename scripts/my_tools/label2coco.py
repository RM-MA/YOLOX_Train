# 命令行执行：  python labelme2coco.py --input_dir images --output_dir coco --labels labels.txt
# 输出文件夹必须为空文件夹

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import shutil
import sys
import uuid
# import imgviz
import numpy as np
import cv2
# import labelme
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from pathlib import Path, PurePath
import cv2 as cv
from tqdm import tqdm

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


# def train_test_split(features_files, labels_files, test_size):
#     train_size = int((1 - test_size) * len(features_files))
#     features_files.sort()
#     labels_files.sort()
#     # print(features_files)
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []
#     for i in range(len(features_files)):
#         if i < train_size:
#             x_train.append(features_files[i])
#             y_train.append(labels_files[i])
#         else:
#             x_test.append(features_files[i])
#             y_test.append(labels_files[i])
#     return x_train, x_test, y_train, y_test

def to_coco(args, label_files, train):
    # 创建 总标签data

    dataset = {'categories': [], 'annotations': [], 'images': []}
    now = datetime.datetime.now()

    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    class_name_to_id = {}
    classes = []
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i  # starts with -1
        class_name = line.strip()  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

        class_name_to_id[class_name] = class_id
        dataset["categories"].append(
            dict(supercategory='mark', id=class_id, name=class_name, )
        )
        classes.append(class_name)

    if train:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_train2017.json")
    else:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_val2017.json")

    ann_id_cnt = 0
    for image_id, filename in enumerate(tqdm(label_files)):

        filepath = Path(filename)
        img_file = filepath.parent / f'{filepath.stem}.jpg'
        if not img_file.exists():
            img_file = filepath.parent / f'{filepath.stem}.png'
            if not img_file.exists():
                print("! no jpg, png")

        # label_file = labelme.LabelFile(filename=filename)

        base = filepath.stem  # 文件名不带后缀
        if train:
            out_img_file = osp.join(args.output_dir, "train2017", base + img_file.suffix)
        else:
            out_img_file = osp.join(args.output_dir, "val2017", base + img_file.suffix)

        # print("| ", out_img_file)

        # ************************** 对图片的处理开始 *******************************************
        # 将标签文件对应的图片进行保存到对应的 文件夹。train保存到 train2017/ test保存到 val2017/
        im = cv.imread(str(img_file))

        height, width, _ = im.shape
        dataset['images'].append(
            dict(
                file_name=base + img_file.suffix,  # 只存图片的文件名
                height=height,
                width=width,
                id=image_id,
            )
        )

        # cv.imwrite(out_img_file, im)
        shutil.copy(img_file, out_img_file)

        # ************************** 对图片的处理结束 *******************************************

        # ************************** 对标签的处理开始 *******************************************
        num_colors = 2
        num_classes = 3
        f =  open(filename)
        for line in f.readlines():
            datas = line.split(" ")
            category_id = int(datas[0])
            color_id = category_id // num_classes  # 0: Blue, 1: Red
            cls_id = category_id % num_classes # 0: R, 1: no activate, 2: activate
            # category_id = num_classes * color_id + cls_id
            H, W, _ = im.shape
            points = datas[5:]
            x1 = float(points[0]) * W
            y1 = float(points[1]) * H
            x2 = float(points[2]) * W
            y2 = float(points[3]) * H
            x3 = float(points[4]) * W
            y3 = float(points[5]) * H
            x4 = float(points[6]) * W
            y4 = float(points[7]) * H
            x5 = float(points[8]) * W
            y5 = float(points[9]) * H
        # label_file = json.load(open(filename))



        # for index, armor in enumerate(label_file['list']):
            # {'ArmorOrEnergy': 1,
            #  'color': 0,
            #  'tags': 0,
            #  'x': 0.5555535554885864,
            #  'y': 0.3610963523387909,
            #  'w': 0.00870203971862793,
            #  'h': 0.0057083964347839355,
            #  'points': [{'x': 0.558269202709198, 'y': 0.3491906523704529},
            #             {'x': 0.5528890490531921, 'y': 0.34932249784469604},
            #             {'x': 0.5512025356292725, 'y': 0.35843759775161743},
            #             {'x': 0.5555318593978882, 'y': 0.36395055055618286},
            #             {'x': 0.5599045753479004, 'y': 0.3582421541213989}]}
            # if armor.get('ArmorOrEnergy', -1) != 1:
            #     print("isn't Energy!")
            #
            # color_id = armor.get('color')  # 0: Blue, 1: Red
            # cls_id = armor.get('tags')  # 0: R, 1: no activate, 2: activate
            # category_id = num_classes * color_id + cls_id
            # H, W, _ = im.shape
            # points = armor['points']
            #
            # x1 = points[0]['x'] * W
            # y1 = points[0]['y'] * H
            # x2 = points[1]['x'] * W
            # y2 = points[1]['y'] * H
            # x3 = points[2]['x'] * W
            # y3 = points[2]['y'] * H
            # x4 = points[3]['x'] * W
            # y4 = points[3]['y'] * H
            # x5 = points[4]['x'] * W
            # y5 = points[4]['y'] * H

            keypoints = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5])
            num_keypoints = int(len(keypoints) / 2)

            keypoints = keypoints.reshape(-1, 2)
            keypoints_type = 2 * np.ones((num_keypoints, 1))
            keypoints = np.concatenate((keypoints, keypoints_type), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            width = max(x1, x2, x3, x4, x5) - min(x1, x2, x3, x4, x5)
            height = max(y1, y2, y3, y4, y5) - min(y1, y2, y3, y4, y5)

            dataset['annotations'].append({
                'area': width * height,
                'bbox': [min(x1, x2, x3, x4), min(y1, y2, y3, y4), width, height],
                'category_id': category_id,
                'id': ann_id_cnt,
                'image_id': image_id,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]],
                'num_keypoints': num_keypoints,
                'keypoints': keypoints
            })
            ann_id_cnt += 1

    with open(out_ann_file, "w") as f:  # 将每个标签文件汇总成data后，保存总标签data文件
        json.dump(dataset, f)


# 主程序执行
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir",
                        default="/home/yiyu/datasets/power/dataset",
                        # "/home/www/datasets/power/all" ,
                        help="input annotated directory")
    parser.add_argument("--output_dir",
                        default="/home/yiyu/py_code/YOLOX/datasets/TUP-Armor-Dataset",
                        # "/home/www/datasets/power/COCO",
                        help="output dataset directory")
    parser.add_argument("--labels",
                        default="/home/yiyu/datasets/power/label.txt",
                        help="labels file")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        # sys.exit(1)
    else:
        os.makedirs(args.output_dir)
        print("| Creating dataset dir:", args.output_dir)

    # 创建保存的文件夹
    if not os.path.exists(osp.join(args.output_dir, "annotations")):
        os.makedirs(osp.join(args.output_dir, "annotations"))
    if not os.path.exists(osp.join(args.output_dir, "train2017")):
        os.makedirs(osp.join(args.output_dir, "train2017"))
    if not os.path.exists(osp.join(args.output_dir, "val2017")):
        os.makedirs(osp.join(args.output_dir, "val2017"))

    # 获取目录下所有的.jpg文件列表
    feature_files_png = glob.glob(osp.join(args.input_dir, "*.png"))
    feature_files_jpg = glob.glob(osp.join(args.input_dir, "*.jpg"))
    feature_files = feature_files_png + feature_files_jpg
    print('| Image number: ', len(feature_files))

    # 获取目录下所有的joson文件列表
    label_files = glob.glob(osp.join(args.input_dir, "*.txt"))
    print('| txt number: ', len(label_files))

    # print(feature_files)
    # feature_files:待划分的样本特征集合    label_files:待划分的样本标签集合    test_size:测试集所占比例
    # x_train:划分出的训练集特征      x_test:划分出的测试集特征     y_train:划分出的训练集标签    y_test:划分出的测试集标签
    x_train, x_test, y_train, y_test = train_test_split(feature_files, label_files, test_size=0.3)
    print("| Train number:", len(y_train), '\t Value number:', len(y_test))

    # 把训练集标签转化为COCO的格式，并将标签对应的图片保存到目录 /train2017/
    print("—" * 50)
    print("| Train images:")
    to_coco(args, y_train, train=True)

    # 把测试集标签转化为COCO的格式，并将标签对应的图片保存到目录 /val2017/
    print("—" * 50)
    print("| Test images:")
    to_coco(args, y_test, train=False)


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)
