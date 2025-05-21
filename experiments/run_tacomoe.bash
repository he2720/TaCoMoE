lora_rank=16
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
#lora_trainable='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=45000
SAVE_STEPS=5000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
model_name_or_path="resourses/chatglm3-6b"   
your_data_path="data/en/task"  
your_checkpopint_path="saved/chatglm3/en/task"  
MAX_SOURCE_LENGTH=2048

DS_SKIP_CUDA_CHECK=1

# # Training Command

deepspeed --num_gpus=1 --master_port $MASTER_PORT run_tacomoe.py \
    --deepspeed src/ds.config \
    --do_train \
    --train_file $your_data_path/train.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 100 \
    --save_steps ${SAVE_STEPS} \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --lora_name moelora \
    --expert_num 8

deepspeed --num_gpus=1 --master_port $MASTER_PORT run_tacomoe.py \
    --do_predict \
    --test_file $your_data_path/test.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/checkpoint-45000 \
    --output_dir results/pred/moelora/qwen_en/45000 \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 8
