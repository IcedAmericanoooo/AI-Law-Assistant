# 客户端调用服务，流式返回
# 多个请求过去后，服务端会把请求放入队列攒批，然后一起推理next token，并且返回给用户
import requests  # HTTP的库
import json 

def clear_lines():
    print('\033[2J')
        
history=[]

while True:
    query=input('问题:')  # 从命令行得到一个问题
    
    # 调用api_server
    response=requests.post('http://localhost:8000/chat',json={
        'query':query,
        'stream': True,
        'history':history,
    },stream=True)
    
    # 流式读取http response body, 按\0分割
    # 下面这个函数会迭代不断接受服务端的推理结果，然后以一个字一个字的形式返回给用户，只有一次的结果才是完整的。
    for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"): # \0是在服务端专门设计的一个间隔符，由于unicode里面没有这个符号，所以在这里要指定decode_unicode=false
        if chunk:
            data=json.loads(chunk.decode('utf-8'))
            text=data["text"].rstrip('\r\n') # 确保末尾无换行
            
            # 清空前一次的内容
            clear_lines()
            # 打印最新内容
            print(text)
    
    # 对话历史
    history.append((query,text))
    history=history[-5:] 
    