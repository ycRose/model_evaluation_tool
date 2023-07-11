import os

with open('/home/lynxi/Documents/model_accuracy_evaluate_tool_0620/gt.txt', 'r') as f:
    file_names = f.readlines()
    
source_dir = '/home/lynxi/Documents/model_accuracy_evaluate_tool_0620/GT/helmet_results_0.4_20230621174044/'
target_dir = '/home/lynxi/Documents/model_accuracy_evaluate_tool_0620/GT/helmet_results_22_0.4_20230621174044/'

import shutil

for file_name in file_names:
    file_name = file_name.strip()  
    if os.path.basename(file_name).endswith('.jpg'):
        file_name = os.path.basename(file_name)[:-4] + '.txt'  # 切片操作替换字符串
    print(file_name)
    source_path = source_dir + file_name
    target_path = target_dir + file_name

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
    else:
        print(f"{source_path} 不存在")
        
print("over")