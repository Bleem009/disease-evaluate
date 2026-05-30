import requests
import json

url_stage1 = "http://127.0.0.1:8000/predict/stage1/"
image_path = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images\bell_pepper_frogeye_leaf_spot_Google_0049.jpg"  # 替换为真实图片路径

with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url_stage1, files=files)
    print("Status:", response.status_code)
    print("Response:", response.json())