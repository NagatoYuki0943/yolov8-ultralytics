import argparse
from pathlib import Path
import numpy as np

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi.utils.cv import read_image, visualize_object_predictions


def run(weights='yolov8n.pt', source='../../ultralytics/assets/bus.jpg', view_img=False, save_img=False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): image file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
    """
    image : np.ndarray = read_image(source)

    yolov8_model_path = f'../../weights/{weights}'
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device='cpu'
    )

    results = get_sliced_prediction(image=image,
                                    detection_model=detection_model,
                                    slice_height=512,
                                    slice_width=512,
                                    overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.2)
    object_prediction_list = results.object_prediction_list

    visualize_result = visualize_object_predictions(image, object_prediction_list)
    image = visualize_result["image"]

    # boxes_list = []
    # id_list = []
    # clss_list = []
    # score_list = []
    # for ind, _ in enumerate(object_prediction_list):
    #     boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
    #             object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
    #     id = object_prediction_list[ind].category.id
    #     clss = object_prediction_list[ind].category.name
    #     score = object_prediction_list[ind].score.value
    #     boxes_list.append(boxes)
    #     id_list.append(id)
    #     clss_list.append(clss)
    #     score_list.append(score)

    # for box, cls, score in zip(boxes_list, clss_list, score_list):
    #     x1, y1, x2, y2 = box
    #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
    #     label = f"{cls} {round(score, 2)}"
    #     t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    #     cv2.rectangle(image, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1)
    #     cv2.putText(
    #         img=image,
    #         text=label,
    #         org=(int(x1), int(y1) - 2),
    #         fontFace=0,
    #         fontScale=0.6,
    #         color=[255, 255, 255],
    #         thickness=1,
    #         lineType=cv2.LINE_AA
    #     )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if view_img:
        cv2.imshow(Path(source).stem, image)
        cv2.waitKey(0)
    if save_img:
        cv2.imwrite("result.jpg", image)

    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--source', type=str, default='../../ultralytics/assets/bus.jpg', help='image file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    # python yolov8_sahi_img.py --weights yolov8n.pt --source ../../ultralytics/assets/bus.jpg  --view-img --save-img
