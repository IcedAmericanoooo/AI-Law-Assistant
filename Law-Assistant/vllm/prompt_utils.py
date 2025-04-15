# 按照chatml格式构造Qwen的prompt
# prompt的拼装格式为：
'''
    输入：
        头部：<|im_start|>system\nyou are a helper assitant.\n<|im_end|>
        中间：（历史信息）
        \n<|im_start|>user\n历史提问A？\n<|im_end|><|im_start|>assitant:历史回答A\n<|im_end|>
        \n<|im_start|>user\n历史提问B？\n<|im_end|><|im_start|>assitant:历史回答B\n<|im_end|>
        尾部：（use发言和assistant引导）
        \n<|im_start|>user\n寻衅滋事？\n<|im_end|><|im_start|>assitant:\n<|endoftext|>
    '''
import copy
from langchain_community.embeddings import ModelScopeEmbeddings

def _build_prompt(
                generation_config,
                tokenizer,
                query,
                history=None,
                system="you are a helpful assistant"):
    if history is None:
        history=[]
    
    # 每一句话都需要有一个<|im_start|>作为开头，一个|im_end|>作为结尾
    im_start,im_start_tokens='<|im_start|>',[tokenizer.im_start_id]
    im_end,im_end_tokens='<|im_end|>',[tokenizer.im_end_id]

    # 换行符
    nl_tokens=tokenizer.encode("\n")

    # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
    def _tokenize_str(role,content): # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}",tokenizer.encode(role)+nl_tokens+tokenizer.encode(content)
    # 编码成{user}\n{content}，后面是对应的input_ids

    # 剩余token数，不能超长
    left_token_space=generation_config.max_window_size

    # prompt头部: system发言 ： <|im_start|>system\nyou are a helper assitant.\n<|im_end|>
    system_text_part, system_tokens_part=_tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
    system_text=f'{im_start}{system_text_part}{im_end}'  # 《—————————————————— 组装
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
    left_token_space -= len(system_tokens)

    # prompt尾部: user发言和assistant引导  \n<|im_start|>user\n了解英国航空么？\n<|im_end|><|im_start|>assitant:\n<|endoftext|>
    query_text_part, query_tokens_part = _tokenize_str('user', query)
    query_tokens_prefix = nl_tokens + im_start_tokens
    query_tokens_suffix = im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode('assistant') + nl_tokens

    # 如果用户的输入太长，应该做截断
    if len(query_tokens_prefix)+len(query_tokens_part)+len(query_tokens_suffix)>left_token_space: # query太长截断
        query_token_len=left_token_space-len(query_tokens_prefix)-len(query_tokens_suffix)
        query_tokens_part=query_tokens_part[:query_token_len]
        query_text_part=tokenizer.decode(query_tokens_part)
    query_tokens=query_tokens_prefix+query_tokens_part+query_tokens_suffix
    query_text=f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"  # 《—————————————————— 组装
    left_token_space-=len(query_tokens)
    
     # prompt腰部: 历史user+assitant对话（中间那部分）
    history_text, history_tokens= '' ,[]        # 《—————————————————— 组装
    for hist_query,hist_response in reversed(history):    # 优先采用最近的对话历史
        hist_query_text,hist_query_tokens_part=_tokenize_str("user",hist_query) # user\n历史提问
        hist_response_text,hist_response_tokens_part=_tokenize_str("assistant",hist_response) # assistant\n历史回答
        # 生成本轮对话
        cur_history_tokens=nl_tokens+im_start_tokens+hist_query_tokens_part+im_end_tokens+nl_tokens+im_start_tokens+hist_response_tokens_part+im_end_tokens
        cur_history_text=f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
        # 储存多轮对话
        if len(cur_history_tokens)<=left_token_space:
            history_text=cur_history_text+history_text      
            history_tokens=cur_history_tokens+history_tokens
            left_token_space-=len(cur_history_tokens)
        else:
            break 
    
    # 生成完整Prompt
    prompt_str=f'{system_text}{history_text}{query_text}'
    prompt_tokens=system_tokens+history_tokens+query_tokens
    return prompt_str,prompt_tokens

# 停用词清理，最后模型给出的文本回答可能包括 <|im_end|> 这样的停止词，所以我们需要把这些词移除掉
def remove_stop_words(token_ids,stop_words_ids):
    token_ids=copy.deepcopy(token_ids)
    while len(token_ids)>0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids
