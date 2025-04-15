# 收集到的法律知识数据csv文件，格式为<title, reply>
# 根据通义千问的SFT格式来生成微调数据并保存在json当中
# 千问的数据微调格式：
'''
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是一个语言模型，我叫通义千问。"
      }
    ]
  }
]
'''

import pandas as pd 
import json

# 生成微调格式数据
def SFT_data(data_path):
    df = pd.read_csv(data_path)

    assert 'title' in df.columns and 'reply' in df.columns

    data = []
    for idx, (_, row) in enumerate(df.iterrows()):   
        if pd.isna(row['title']) or pd.isna(row['reply']):
            continue
    
        prompt_template = {
            "id": f"identity_{idx}",
            "conversations": [
                {
                    "from": "user",
                    "value": str(row["title"])
                },
                {
                    "from": "assistant",
                    "value": str(row["reply"])
                }
            ]
        }
        data.append(prompt_template)

    # 保存到json文件中
    with open('qwen_sft_chat.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



if __name__ == '__main__':
    data_path = './law_faq.csv'
    SFT_data(data_path)
