import json
from tqdm import tqdm
import numpy as np
import os

#-------------------------------------------------#
#   将coco格式的标注转换为yolo格式的标注
#   不适用官方的coco json
#-------------------------------------------------#

# coco json文件
json_file = "D:/ml/code/datasets/coco/annotations/instances_val2017.json"
# 保存txt目录
save_dir  = "D:/ml/code/datasets/coco/annotations/val_txt"
# bbox or segmentation
tp        = "bbox"


def coco2yolo():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 打开coco json
    with open(json_file, mode="r", encoding="utf-8") as f:
        js = json.load(f)

    # 打印标签信息
    print("=" * 100)
    print("labels:")
    for category in js["categories"]:
        print(category["id"], "=>", category["name"])
    print("=" * 100)

    # 获取图片信息 {image_id: image_dict,...}
    images = {}
    for i in js["images"]:
        images[i["id"]] = i

    if tp == "bbox":
        bbox(js["annotations"], images)
    elif tp == "segmentation":
        sementation(js["annotations"], images)


def bbox(annotations, images):
    for annotation in annotations:
        image_id     = annotation["image_id"]
        category_id  = annotation["category_id"]
        bbox         = annotation["bbox"]           # coco: [x_min, ymin, width, height]
        bbox[0]     += int(bbox[2] / 2)             # yolo: [x_center, y_center, width, height]
        bbox[1]     += int(bbox[3] / 2)
        height       = images[image_id]["height"]
        width        = images[image_id]["width"]
        x_center     = bbox[0] / width              # 归一化
        y_center     = bbox[2] / width
        width        = bbox[1] / height
        height       = bbox[3] / height
        bbox_line    = [str(category_id), str(x_center), str(y_center), str(width), str(height)]
        if "bbox_lines" not in images[image_id].keys():
            images[image_id]["bbox_lines"] = [bbox_line]
        else:
            images[image_id]["bbox_lines"].append(bbox_line)

    num_pic = num_pic_bbox = num_bbox = 0
    empty_images = []
    # 将lines的标注批量保存进以图片名字命名的txt文件中
    for image in tqdm(images.values()):
        num_pic += 1
        if "bbox_lines" in image.keys(): # 有的图片没有box,就忽略lines
            num_pic_bbox += 1
            num_bbox += len(image["bbox_lines"])
            file_name = "".join(image["file_name"].split(".")[:-1]) + ".txt"
            with open(os.path.join(save_dir, file_name), mode="w", encoding="utf-8") as f:
                for line in image["bbox_lines"]:
                    line = [str(l) for l in line]
                    f.write(" ".join(line) + "\n")
        else:
            empty_images.append(image["file_name"])

    print("=" * 100)
    print(f"总图片数量为 {num_pic}, 有bbox的图片数量为 {num_pic_bbox}, 没有bbox的图片数量为 {num_pic - num_pic_bbox}, 总bbox数为 {num_bbox}")
    print("=" * 100)
    print("没有bbox的图片为:", empty_images)
    print("=" * 100)


def sementation(annotations, images):
    for annotation in annotations:
        image_id    = annotation["image_id"]
        category_id = annotation["category_id"]
        height      = images[image_id]["height"]
        width       = images[image_id]["width"]

        # 忽略 segmentation 为字典的数据,里面存储的为 "segmentation": {"counts": [], "size": []},含义不明
        if isinstance(annotation["segmentation"], dict):
            continue
        for segmentation in annotation["segmentation"]:
            segment = np.array(segmentation)
            segment[0::2] /= width          # 归一化
            segment[1::2] /= height
            segment = list(segment)
            segment.insert(0, category_id)  # 添加类别id
            if "segment_lines" not in images[image_id].keys():
                images[image_id]["segment_lines"] = [segment]
            else:
                images[image_id]["segment_lines"].append(segment)

    num_pic = num_pic_segment = num_segment = 0
    empty_images = []
    # 将lines的标注批量保存进以图片名字命名的txt文件中
    for image in tqdm(images.values()):
        num_pic += 1
        if "segment_lines" in image.keys(): # 有的图片没有box,就忽略lines
            num_pic_segment += 1
            num_segment += len(image["segment_lines"])
            file_name = "".join(image["file_name"].split(".")[:-1]) + ".txt"
            with open(os.path.join(save_dir, file_name), mode="w", encoding="utf-8") as f:
                for line in image["segment_lines"]:
                    line = [str(l) for l in line]
                    f.write(" ".join(line) + "\n")
        else:
            empty_images.append(image["file_name"])

    print("=" * 100)
    print(f"总图片数量为 {num_pic}, 有segmentation的图片数量为 {num_pic_segment}, 没有segmentation的图片数量为 {num_pic - num_pic_segment}, 总segmentation数为 {num_segment}")
    print("=" * 100)
    print("没有segmentation的图片为:", empty_images)
    print("=" * 100)


if __name__ == "__main__":
    coco2yolo()
