# lora对Qwen进行微调
from dataclasses import dataclass, field
import json
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate.utils import DistributedType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
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


def process_function(
sources,
tokenizer: transformers.PreTrainedTokenizer,
max_len: int,
system_message: str = "You are a helpful assistant."
) -> Dict:
    
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



if __name__ == "__main__":
    # 加载数据集
    data_dir = "./qwen_sft_chat.json"
    ds = json.load(open(data_dir, 'r'))

    # 加载tokenizer
    model_dir = 'qwen/Qwen-1_8B-Chat-Int8'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 处理输入数据
    sources = [example['conversations'] for example in ds]
    tokenized_ds = process_function(sources=sources, tokenizer=tokenizer, max_len=512)
    train_dataset = dict(
        input_ids = tokenized_ds["input_ids"],
        attention_mask = tokenized_ds["attention_mask"],
        labels = tokenized_ds["labels"],
    )

    # 创建dataset
    dl = DataLoader(train_dataset, batch_size=2, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("E:\LLM\Law-Assistant\pretrained_model\Qwen\Qwen-1_8B-Chat-Int8", trust_remote_code = True, low_cpu_mem_usage = True,
                                                 torch_dtype = torch.bfloat16, device_map = "auto")
    
    # 配置lora
    config = LoraConfig(task_type=TaskType.get_peft_model)
    model = get_peft_model(model = model, config = config)
    model.enable_input_require_grads()

    # 配置训练参数
    args = TrainingArguments(
        output_dir="./output/lora_adapter/",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=10,
        num_train_epochs=1,
    )

    # 创建训练器，并开始训练
    trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds.select(range(6000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # 模型合并，由于vLLM不支持未合并的LoRA模型，所以必须要先合并之后才能加载
    base_model = AutoModelForCausalLM.from_pretrained("E:\LLM\Law-Assistant\pretrained_model\Qwen\Qwen-1_8B-Chat-Int8")
    lora_model = PeftModel.from_pretrained(base_model, "./output/lora_adapter/")
    merge_model = lora_model.merge_and_unload()
    merge_model.save_pretrained("./pretrained_model/lora_model")

    


    