import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import os
import numpy as np
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM , Trainer , TrainingArguments ,BitsAndBytesConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from peft import LoraConfig , get_peft_model , TaskType
import gc
from datasets import Dataset

project='lora-flant5'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)



learning_rate = 2e-5
num_iterations = 2000
batch_size = 4
max_length = 512
gradient_accumulation_steps = 4
warmup_steps = 100
weight_decay = 0.01
adam_epsilon = 1e-8
max_grad_norm = 1.0
beta1=0.9
beta2=0.95
eval_interval=200
train_loss_log_iteration=50

# Load the dataset
data=pd.read_csv('HealthCare.csv')

device='cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16",quantization_config=bnb_config,device_map="auto")


tokenized_data=[]

for question , answer in zip(data['Question'],data['Answer']):
    
    if answer is np.nan or question is np.nan:
        continue
    
    tokenized_inputs = tokenizer(question,padding='max_length',max_length=max_length,truncation=True)
    output=tokenizer(answer,padding='max_length',max_length=max_length,truncation=True)
    output['labels']=[
         (i_d if i_d != tokenizer.pad_token_id else -100)   for i_d in output['input_ids']
        ]
    tokenized_inputs['labels']=output['labels']
    
        
    tokenized_data.append(tokenized_inputs)
        
data=pd.DataFrame(tokenized_data)  


train , test = train_test_split(data, test_size=0.2, random_state=42)

train_dataset=Dataset.from_pandas(train,preserve_index=False)
test_dataset=Dataset.from_pandas(test,preserve_index=False)

      
lora_config=LoraConfig(
    r=16, # dimension of the A & B eg: 100*16 for A and 16*100 for B
    lora_alpha=1,# the scale factor of the Lora coefficients to adjust the magnitute in research paper its 1 /16 multiplied A*B
    target_modules=["q","v"], # which weights we aare appling the  lora on 
    lora_dropout=0.05, # dropout rate for the Lora coefficients
    bias="none", # the bias of the Lora coefficients
    task_type=TaskType.SEQ_2_SEQ_LM  # type of task for the model
    
)


model=get_peft_model(model,lora_config)


model.print_trainable_parameters()


args = TrainingArguments(
       output_dir ='./output',
       do_train=True,
       do_eval=True,
       evaluation_strategy='steps',
       per_device_train_batch_size = batch_size,
       per_device_eval_batch_size=batch_size*2,
       gradient_accumulation_steps=gradient_accumulation_steps, # acuumulate  batches before the update 
       max_steps=num_iterations,
       warmup_steps=warmup_steps,
       weight_decay = weight_decay,
       metric_for_best_model= 'loss',
       optim='adamw_torch',
       logging_strategy='steps',
       logging_steps=50,
       lr_scheduler_type='linear',
       save_strategy='steps',
       save_steps=100
       )

wandb.init(project=project, entity='vmunagal',config=args) 

trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )
trainer.train()
    
del model
gc.collect()
torch.cuda.empty_cache()
wandb.finish()
