# 完整的实用程序
import os
import sys
import json
import requests
from collections import Counter
from typing import List, Dict

def get_user_data() -> Dict:
    """获取示例数据"""
    return {"name": "张三", "age": 25, "city": "北京"}

def main():
    # 使用os模块
    print(f"当前工作目录: {os.getcwd()}")
    
    # 使用json模块
    user_data = get_user_data()
    json_str = json.dumps(user_data, ensure_ascii=False)
    print(f"JSON数据: {json_str}")
    
    # 使用Counter
    words = ['苹果', '香蕉', '苹果', '橙子', '香蕉', '苹果']
    word_count = Counter(words)
    print(f"单词统计: {word_count}")

if __name__ == "__main__":
    main()
