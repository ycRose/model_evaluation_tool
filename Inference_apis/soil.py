import requests
import os
import cv2
import json
from utils import folder_utils

def test_soil_segmentation_api(preUrl, headers, imageurl, confThreshold):
    # headers = json.loads(headers)
    # data = json.loads(data)
    folder_name = "Prediction/soil_results_" + str(confThreshold)
    img_dir = folder_utils.folder_creat(folder_name)
    
    url = preUrl #+ "SafetyHatAndVestDetection"
    for i in range(len(imageurl)):
        print(f"\nProcess:{i+1}/{len(imageurl)}\n")
        data = dict()
        data["image"] = imageurl[i]#os.path.join(imagePath,"soil.jpg")
        data["parameters"] = [{"score": confThreshold},{"type": 1000}]
        print(url)
        print(data)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # print(response.text)
        if (response.status_code != 200):
            print(f"Error code: {response.status_code}")
            return
        
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
            ["label_id", "area_score", "xmin", "ymin", "xmax", "ymax"]
        
        assert result['results'][0]["categories"] == \
            [{"label_id": 0, "name": "hard"},
            {"label_id": 1, "name": "soil"},
            {"label_id": 2, "name": "mask"},
            {"label_id": 3, "name": "steel"},
            {"label_id": 4, "name": "other"}]

        assert 'height' in result['results'][0]['imageSize']
        assert 'width' in result['results'][0]['imageSize']  
        assert 'depth' in result['results'][0]['imageSize']
        
        height = result["results"][0]["imageSize"]["height"]
        width = result["results"][0]["imageSize"]["width"]
        image = cv2.imread(imageurl[i])
        for bbox_info in result["results"][0]["prediction"]["detection"]:
            [classid, conf, xmin, ymin, xmax, ymax] = bbox_info
            classid = int(classid)
            if classid > 3:
                continue
            xmin = int(xmin*float(width))
            ymin = int(ymin*float(height))
            xmax = int(xmax*float(width))
            ymax = int(ymax*float(height))
            
            print("classid:",classid)
            if classid == 1 :
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
                cv2.putText(image, f'{conf:.2f}', (xmin , ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # else:
            #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 1)

        save_path = os.path.join(img_dir, os.path.basename(imageurl[i]))
        #save_path = "./vest_results_0.8/" + os.path.basename(imageurl)    
        cv2.imwrite(save_path, image)
        print("saved img")