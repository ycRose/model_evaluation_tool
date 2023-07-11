import sys
import json
import requests
import os
import cv2
import numpy as np
import colorama
import argparse
import time
from Inference_apis import safetyhat,vest,soil
from Data_loader.downloadimgsfromjson import downloadimgsfromjson, localimages
import utils.converter as converter
from utils.enumerators import BBFormat, BBType, CoordinatesType
from utils.folder_utils import folder_creat
from src.pascal_voc_evaluator import (get_pascalvoc_metrics, plot_precision_recall_curve,
                                                 plot_precision_recall_curves)
from src.coco_evaluator import (get_coco_metrics, get_coco_summary)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_mode', type=str, default='local', help='local service or remote service')
    parser.add_argument('--server_ip', type=str, default='10.4.34.0', help='device ip')
    parser.add_argument('--data_type', type=str, default='helmet', help='test data type,like helmet / vest or other support type')            # 源模型文件
    parser.add_argument('--conf', type=float, default=0.6, help='confThreshold for model')
    parser.add_argument('--test_data_mode', type=int, default=0, help='0-json file; 1-local images')  # 测试数据类型
    parser.add_argument('--test_data', type=str, default="./Test_dataset", help='test data source')  # 测试数据来源
    parser.add_argument('--gtdir', type=str, default="./GT", help='folder of ground truth files')  # gt路径
    opt = parser.parse_args()
    return opt
 
#preUrl = "http://10.4.34.2:8211/ai/"
headers = {"Content-Type": "application/json"}
if __name__ == '__main__':
    
    opt = parse_opt()
    preUrl = "http://" + opt.server_ip + ":8211/ai/"
    
    if opt.test_data_mode == 0:
        imageurl, imagedir = downloadimgsfromjson(opt.test_data)
    elif opt.test_data_mode == 1:
        imageurl, imagedir = localimages(opt.test_data)
    
    if opt.data_type == "helmet":
        name = "SafetyHatAndVestDetection"
        safetyhat.test_safetyhat_det_api(preUrl+name, headers, imageurl, opt.conf)
    elif opt.data_type == "vest":
        name = "SafetyHatAndVestDetection"
        det_dir = vest.test_vest_det_api(preUrl+name, headers, imageurl, opt.conf, opt.server_mode)
    elif opt.data_type == "head":
        name = "HeadCountDetPipeline"
        #safetyhat.test_safetyhat_det_api(preUrl+name, headers, imageurl, opt.conf)
    elif opt.data_type == "person":
        name = "PersonCountDetection"
        #return new HeadCountDet()
    elif opt.data_type == "AreaIntrusion":
        name = "AreaIntrusionDetPipeline"
        #return new AreaIntrusionDet()
    elif opt.data_type == "soil":
        name = "SoilSegmentation"
        soil.test_soil_segmentation_api(preUrl+name, headers, imageurl, opt.conf)
    elif opt.data_type == "HumanPose":
        name = "HumanPoseEstimationPipeline"
        #return new HumanPoseDet()
    elif opt.data_type == "belt":
        name = "TFCameraBeltPipeline"
        #return new SafetyBeltDet()
    elif opt.data_type == "fire":
        name = "FireSmogPipeline"
        #return new FireSmogDet()
    elif opt.data_type == "wheel":
        name = "WheelPipeline"
        #return new WheelCleanDet()
    elif opt.data_type == "CarClose":
        name = "CarClose"
        #return new CarCloseDet()
    else:
        print("error!!! unsupport type!!!")

    print("Loading predictions and GTs......\n")
    dir_images_gt = imagedir#"/home/lynxi/Documents/model_accuracy_evaluate_tool_0620/Test_dataset/vest_20230626153057"
    dir_dets = det_dir#"/home/lynxi/Documents/model_accuracy_evaluate_tool_0620/GT/helmet_results_22_0.4_20230621174044"
    det_annotations = converter.text2bb(dir_dets,
                                    bb_type=BBType.DETECTED,
                                    bb_format=BBFormat.XYX2Y2,
                                    type_coordinates=CoordinatesType.RELATIVE,
                                    img_dir=dir_images_gt)
    print("len(det_annotations):", len(det_annotations))

    dir_annotations_gt = opt.gtdir
    gt_annotations = converter.text2bb(dir_annotations_gt, bb_type=BBType.GROUND_TRUTH,bb_format=BBFormat.YOLO,
                                    type_coordinates=CoordinatesType.RELATIVE,
                                    img_dir=dir_images_gt)
    print("len(gt_annotations):", len(gt_annotations))
    
    print("Evaluating results......\n")
    pascal_res = get_pascalvoc_metrics(gt_annotations,
                                        det_annotations,
                                        iou_threshold=0.5,
                                        generate_table=True)
    
    dir_save_results = os.path.join(os.getcwd(), "eval_results")#"/home/lynxi/Documents/model_accuracy_evaluate_tool_0701/eval_results"
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_dir = os.path.join(dir_save_results, opt.data_type + "_" + current_time)
    os.makedirs(file_dir, exist_ok=True)
    file_name = opt.data_type + ".txt"
    file_path = os.path.join(file_dir, file_name)
    
    mAP = pascal_res['mAP']
    print("data-type:", opt.data_type)
    print("map:", mAP)
    with open(file_path, "w") as f:
        f.write("map: {}\n".format(mAP))
    
    if 'per_class' in pascal_res:
        #print(pascal_res['per_class'],"\n\n")
        dict1 = pascal_res['per_class']
        total_gt = 0
        total_tp = 0
        for c, v in dict1.items():
            dict_res = dict1[c]
            total_gt += dict_res['total positives']
            total_tp += dict_res['total TP']
            recall = dict_res['total TP'] / dict_res['total positives']
            precision = dict_res['total TP'] / (dict_res['total TP'] + dict_res['total FP'])
            F1_score = 2 * (precision * recall) / (precision + recall)
            average_tp_iou = sum(dict_res['total tp iou']) / dict_res['total TP']
            content = "class: {}, precision: {}, recall: {}, ap: {}, F1_score: {}, average_tp_iou: {}\n".format(c, precision, recall, dict_res['AP'], F1_score, average_tp_iou)
            print(content)
            with open(file_path, "a") as f:
                f.write(content)
        #print("dataset ap:",total_tp/total_gt)
        
        # Save a single plot with all classes
        plot_precision_recall_curve(pascal_res['per_class'],
                                    mAP=mAP,
                                    savePath=file_dir,
                                    showGraphic=False)
        # Save plots for each class
        plot_precision_recall_curves(pascal_res['per_class'],
                                        showAP=True,
                                        savePath=file_dir,
                                        showGraphic=False)
    
    coco_res1 = get_coco_summary(gt_annotations, det_annotations)
    coco_res2 = get_coco_metrics(gt_annotations, det_annotations)
    print("coco_res1:")
    print(coco_res1)
    with open(file_path, "a") as f:
        for key, value in coco_res1.items():
            line = "{}: {}\n".format(key, value)
            f.write(line)
    # print("coco_res2:")
    # print(coco_res2)
    
    print("test over")
