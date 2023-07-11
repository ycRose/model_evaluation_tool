import json
import requests
import os
import cv2
import numpy as np
import base64
from utils import folder_utils

def downloadimgsfromjson(jsonfile):
    with open(jsonfile, "r") as f:
        lines = json.load(f)#f.readlines()
        #print(lines[0])
    
    # # 获取当前脚本文件所在的目录
    # current_file_path = os.path.abspath(__file__)

    # # 获取当前脚本文件所在的目录的上一级目录，即工程的根目录
    # project_root = os.path.dirname(os.path.dirname(current_file_path))
    # print("project_root:",project_root)
    # #img_dir = "helmet_results_" +str(confThreshold)
    # img_dir = os.path.join(project_root,"Test_dataset/" +os.path.basename(jsonfile)[:-5])
    # #img_dir = "./Test_dataset/" +os.path.basename(jsonfile)[:-5]
    # if not os.path.isdir(img_dir):
    # # 文件夹不存在，则创建一个新的文件夹
    #     os.makedirs(img_dir)
    folder_name = "Test_dataset/" +os.path.basename(jsonfile)[:-5]
    img_dir = folder_utils.folder_creat(folder_name)

    name_dict = {}#检查有没有同名图片而创建
    images = []
    for i in range(len(lines)):
        line = lines[i]
        image_url = line['imageUrl']
        
        img_name = os.path.basename(image_url)
        if img_name in name_dict:
            name_dict[img_name].append(i)
        else:
            name_dict[img_name] = [i]
        
        # 下载对应的图像数据
        response = requests.get(image_url, stream=True)
        image_data = np.frombuffer(response.content, np.uint8)

        # 将图像数据解码成 OpenCV 格式
        img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        img_name = os.path.basename(image_url)
        if img_name.endswith('.png'):
            img_name = img_name[:-4] + '.jpg'  # 切片操作替换字符串
        
        save_path = os.path.join(img_dir, img_name)
        print(i,save_path)   
        images.append(save_path)
    
        cv2.imwrite(save_path, img)
        
        # #test_carclose_det_api(preUrl, headers, save_path)
        # #test_vest_det_api(preUrl, headers, save_path, 0.5)
        # #test_safetyhat_det_api(preUrl, headers, save_path, 0.5)
        # #test_soil_segmentation_api(preUrl, headers, save_path,0.3)
        # test_soil_segmentation_api(preUrl, headers, save_path,0.1)
        
    duplicated_names = [name for name in name_dict if len(name_dict[name]) > 1]
    if len(duplicated_names) == 0:
        print('所有测试图片均不相同！')
    else:
        print('存在重复的测试图片：')
        for name in duplicated_names:
            print(f'{name}，出现位置：{name_dict[name]}')
    
    return images, img_dir


def localimages(path):
    # with open(jsonfile, "r") as f:
    #     lines = json.load(f)#f.readlines()
    #     #print(lines[0])
    
    # # 获取当前脚本文件所在的目录
    # current_file_path = os.path.abspath(__file__)

    # # 获取当前脚本文件所在的目录的上一级目录，即工程的根目录
    # project_root = os.path.dirname(os.path.dirname(current_file_path))
    # print("project_root:",project_root)
    # #img_dir = "helmet_results_" +str(confThreshold)
    # img_dir = os.path.join(project_root,"Test_dataset/" +os.path.basename(jsonfile)[:-5])
    # #img_dir = "./Test_dataset/" +os.path.basename(jsonfile)[:-5]
    # if not os.path.isdir(img_dir):
    # # 文件夹不存在，则创建一个新的文件夹
    #     os.makedirs(img_dir)
    #folder_name = "Test_dataset/" +os.path.basename(jsonfile)[:-5]
    img_dir = os.path.abspath(path)
    #img_dir = path#folder_utils.folder_creat(folder_name)
    file_list = os.listdir(img_dir)

    name_dict = {}#检查有没有同名图片而创建
    images = []
    for i, file in enumerate(file_list):
        
        img_name = os.path.basename(file)
        if img_name in name_dict:
            name_dict[img_name].append(i)
        else:
            name_dict[img_name] = [i]
            
        save_path = os.path.join(img_dir, img_name)
        print(i,save_path)   
        images.append(save_path)
        
        # # 下载对应的图像数据
        # response = requests.get(image_url, stream=True)
        # image_data = np.frombuffer(response.content, np.uint8)

        # # 将图像数据解码成 OpenCV 格式
        # img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        # img_name = os.path.basename(image_url)
        # if img_name.endswith('.png'):
        #     img_name = img_name[:-4] + '.jpg'  # 切片操作替换字符串
        
        # save_path = os.path.join(img_dir, img_name)
        # print(i,save_path)   
        # images.append(save_path)
    
        # cv2.imwrite(save_path, img)
        
        # # #test_carclose_det_api(preUrl, headers, save_path)
        # # #test_vest_det_api(preUrl, headers, save_path, 0.5)
        # # #test_safetyhat_det_api(preUrl, headers, save_path, 0.5)
        # # #test_soil_segmentation_api(preUrl, headers, save_path,0.3)
        # # test_soil_segmentation_api(preUrl, headers, save_path,0.1)
        
    duplicated_names = [name for name in name_dict if len(name_dict[name]) > 1]
    if len(duplicated_names) == 0:
        print('所有测试图片均不相同！')
    else:
        print('存在重复的测试图片：')
        for name in duplicated_names:
            print(f'{name}，出现位置：{name_dict[name]}')
    
    return images, img_dir

###np 图片编码成base64形式
def image_to_base64(image_path):
    img = cv2.imread(image_path)#
    image = cv2.imencode('.jpg', img)[1]
    image_code = str(base64.b64encode(image), 'utf-8')
    return image_code