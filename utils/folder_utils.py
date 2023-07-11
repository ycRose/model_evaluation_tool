import os
import time

def folder_creat(folder_name):
    # 获取当前脚本文件所在的目录
    current_file_path = os.path.abspath(__file__)

    # 获取当前脚本文件所在的目录的上一级目录，即工程的根目录
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    print("project_root:",project_root)
    #img_dir = "helmet_results_" +str(confThreshold)
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    img_dir = os.path.join(project_root,folder_name + "_" + current_time)
    if not os.path.isdir(img_dir):
    # 文件夹不存在，则创建一个新的文件夹
        os.makedirs(img_dir)
        
    return img_dir