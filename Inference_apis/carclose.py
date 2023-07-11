import requests
import os
import cv2
import json

def test_carclose_det_api(preUrl, headers, imageurl, confThreshold):
    # headers = json.loads(headers)
    # data = json.loads(data)
    url = preUrl + "CarClose"
    data = dict()
    data["image"] = imageurl#"/home/lynxi/repo/dist/onepic/carclose4.jpg"
    data["parameters"] = [{"confThreshold":0.8}]
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
        [{"label_id": 0, "name": "truck"},
         {"label_id": 1, "name": "close"},
         {"label_id": 2, "name": "open"}]

    # assert "category_color_map" in result["results"][0]
    # assert result["results"][0]["category_color_map"] == \
    #     {
    #         "noSafetyhat": [0, 0, 255],
    #         "noVest": [0, 0, 255],
    #         "safetyhat": [0, 255, 0],
    #         "vest": [0, 255, 0]
    #     }

    assert 'height' in result['results'][0]['imageSize']
    assert 'width' in result['results'][0]['imageSize']  
    assert 'depth' in result['results'][0]['imageSize']
    
    print("test over")
    
    height = result["results"][0]["imageSize"]["height"]
    width = result["results"][0]["imageSize"]["width"]
    image = cv2.imread(imageurl)
    for bbox_info in result["results"][0]["prediction"]:
        [classid, conf, xmin, ymin, xmax, ymax] = bbox_info
        classid = int(classid)
        print(classid)
        if classid > 2:
            continue
        xmin = int(xmin*float(width))
        ymin = int(ymin*float(height))
        xmax = int(xmax*float(width))
        ymax = int(ymax*float(height))
        
        if classid == 2 :
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
            cv2.putText(image, f'{conf:.2f}', (xmin , ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
            cv2.putText(image, f'{conf:.2f}', (xmin , ymin ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    img_dir = "./carclose_results_" +str(confThreshold)
    if not os.path.isdir(img_dir):
    # 文件夹不存在，则创建一个新的文件夹
        os.makedirs(img_dir)
    save_path = os.path.join(img_dir, os.path.basename(imageurl))    
    cv2.imwrite(save_path, image)
    print("saved img")