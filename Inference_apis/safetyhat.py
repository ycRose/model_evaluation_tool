import requests
import os
import cv2
import json
import time
from utils import folder_utils

def test_safetyhat_det_api(preUrl, headers, imageurl, confThreshold):
    # headers = json.loads(headers)
    # data = json.loads(data)
    folder_name = "Prediction/helmet_results_" + str(confThreshold)
    img_dir = folder_utils.folder_creat(folder_name)
    det_folder_name = "GT/helmet_results_" + str(confThreshold)
    det_dir = folder_utils.folder_creat(det_folder_name)
    print(det_dir)
    gt_folder_name = "GT/gt_helmet_results_" + str(confThreshold)
    gt_dir = folder_utils.folder_creat(gt_folder_name)
    print(gt_dir)
    
    url = preUrl #+ "SafetyHatAndVestDetection"
    for i in range(len(imageurl)):
        print(f"\nProcess:{i+1}/{len(imageurl)}\n")
        data = dict()
        data["image"] = imageurl[i]#"/home/lynxi/repo/dist/onepic/bus.jpg"
        data["parameters"] = [{"confThreshold": confThreshold}, {}]
        #print("\n")
        print(url)
        print(data)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # print(response.text)
        result = response.json()
        print(result)
        print("\n")
        
        assert result['resultCode'] == 0
        assert result['errorString'] == 'success'

        assert 'results' in result
        assert 'usingTime' in result['results'][0]
        assert 'imageSize' in result['results'][0]
        
        assert 'title' in result['results'][0]
        assert result['results'][0]["title"] == \
            ["label_id", "confidence_score", "xmin", "ymin", "xmax", "ymax"]

        assert "categories" in result["results"][0]
        assert result['results'][0]["categories"] == \
            [{"label_id": 1, "name": "noVest"},
            {"label_id": 2, "name": "vest"},
            {"label_id": 3, "name": "noSafetyhat"},
            {"label_id": 4, "name": "safetyhat"}]

        assert "category_color_map" in result["results"][0]
        assert result["results"][0]["category_color_map"] == \
            {
                "noSafetyhat": [0, 0, 255],
                "noVest": [0, 0, 255],
                "safetyhat": [0, 255, 0],
                "vest": [0, 255, 0]
            }

        assert 'height' in result['results'][0]['imageSize']
        assert 'width' in result['results'][0]['imageSize']  
        assert 'depth' in result['results'][0]['imageSize']
        
        height = result["results"][0]["imageSize"]["height"]
        width = result["results"][0]["imageSize"]["width"]
        image = cv2.imread(imageurl[i])
        
        label_pred_res = []
        conf_pred_res = []
        if os.path.basename(imageurl[i]).endswith('.jpg'):
            txt_name = os.path.basename(imageurl[i])[:-4] + '.txt'  # 切片操作替换字符串
        det_save_path = os.path.join(det_dir, txt_name)
        gt_save_path = os.path.join(gt_dir, txt_name)
        print(det_save_path)
        for bbox_info in result["results"][0]["prediction"]:
            [classid, conf, xmin, ymin, xmax, ymax] = bbox_info
            gt_bbox_info = [classid, xmin, ymin, xmax, ymax]
            classid = int(classid)
            
            if classid < 3:
                continue
            xmin = int(xmin*float(width))
            ymin = int(ymin*float(height))
            xmax = int(xmax*float(width))
            ymax = int(ymax*float(height))
            
            with open(det_save_path, "at") as file:
                # 将文本写入文件
                #for item in bbox_info:
                for item in bbox_info:
                    file.write(str(item) + " ")
                file.write('\n')
            with open(gt_save_path, "at") as file:
                # 将文本写入文件
                #for item in bbox_info:
                file.write(' '.join(str('{:.3f}'.format(item)) for item in gt_bbox_info))
                file.write('\n')
            
            print("classid:",classid)
            if classid == 3 :
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
                cv2.putText(image, f'{conf:.2f}', (xmin , ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
                cv2.putText(image, f'{conf:.2f}', (xmin , ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
        
        save_path = os.path.join(img_dir, os.path.basename(imageurl[i]))    
        cv2.imwrite(save_path, image)
        print("saved img")