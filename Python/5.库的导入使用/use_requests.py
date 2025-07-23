# TODO 使用库：requests
# use_requests.py
import requests

def fetch_data():
    response = requests.get('https://api.github.com')
    if response.status_code == 200:
        print("GitHub API 响应数据:", response.json())
    else:
        print("请求失败，状态码:", response.status_code)

fetch_data()
