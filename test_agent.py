import requests
import json

url = "http://127.0.0.1:8000/chat"
files = {
    "message": (None, "请分析这张水稻叶片图片"),
    "image": open(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images\rice_blast_google_0062.jpg", "rb")
}
response = requests.post(url, files=files)
print(response.json())