# -*- encoding: utf-8 -*-
# here put the import lib
import json
import torch
import copy
import sys
class qwen2_train(object):
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department

    def __call__(self, examples):
        model_inputs = {"input_ids": [], "labels": []}
        
        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("data/task_dataset.json", "r"))["str2id"]
       
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query = examples[self.prompt_column][i]
                answer = examples[self.response_column][i]
                
                # 1. 构造 Qwen2 格式的对话输入
                messages = []
                if self.history_column and examples[self.history_column][i]:
                    print('this')
                    sys.exit(0)
                    for old_query, response in examples[self.history_column][i]:
                        messages.append({"role": "user", "content": old_query})
                        messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": query})
                messages.append({"role": "assistant", "content": answer})
              
                conversation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                model_inputs['input_ids'].append(torch.tensor(self.tokenizer(conversation, add_special_tokens=False).input_ids, dtype=torch.long))
                model_inputs["labels"].append(copy.deepcopy(model_inputs['input_ids'][-1]))
                tmp_message = [
                    {"role": "user", "content": query}, 
                    {"role": "assistant", "content": answer}
                ]
                cur = 0
                message = []
                instruction = self.tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = self.tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)
                instruction_len = len(torch.tensor(self.tokenizer(instruction, add_special_tokens=False).input_ids, dtype=torch.long))
                conversation_len = len(torch.tensor(self.tokenizer(conversation, add_special_tokens=False).input_ids, dtype=torch.long))
            
                model_inputs["labels"][-1][cur:instruction_len] = -100

                cur = conversation_len
                message += tmp_message
                if self.task:
                    task_id = task_dict[examples['task_dataset'][i]]
                    model_inputs["task_id"].append(task_id)

        return model_inputs
        
class qwen2_eval(object):
    
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department

        

    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []
        model_inputs = {"input_ids": [], "labels": [], 'attention_mask': []}
        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("data/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[self.response_column][i])

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                answer = examples[self.response_column][i]
                messages = []
                messages.append({"role": "user", "content": query})
                messages.append({"role": "assistant", "content": answer})
                conversation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                model_inputs['input_ids'].append(torch.tensor(self.tokenizer(conversation, add_special_tokens=False, padding='max_length',
                    truncation=True,
                    max_length=self.data_args.max_source_length,
                    return_tensors='pt').input_ids, dtype=torch.long).squeeze(0))
                model_inputs['attention_mask'].append(torch.tensor(self.tokenizer(conversation, add_special_tokens=False, padding='max_length',
                    truncation=True,
                    max_length=self.data_args.max_source_length,
                    return_tensors='pt').attention_mask, dtype=torch.long).squeeze(0))
                model_inputs["labels"].append(copy.deepcopy(model_inputs['input_ids'][-1]))
                tmp_message = [
                    {"role": "user", "content": query}, 
                    {"role": "assistant", "content": answer}
                ]
                cur = 0
                message = []
                instruction = self.tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                
                instruction_len = len(torch.tensor(self.tokenizer(instruction, add_special_tokens=False).input_ids, dtype=torch.long))
             
            
                model_inputs["labels"][-1][cur:instruction_len] = -100
            if self.task:
                task_id = task_dict[examples['task_dataset'][i]]
                model_inputs["task_id"].append(task_id)

        return model_inputs