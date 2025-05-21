# TaCoMoE
The overall architecture of TaCoMoE. The proposed TaCoMoE consists of three maincomponents: dialogue input engineering, task-oriented mixture of experts layer,and contrastive loss.<br>
<img src="imgs/asq_model.png" alt="" style="width:630px; background-color:white; padding:10px; border-radius:8px;" />
# Requirements
Here are the software and hardware environments I used in my experiments:<br>
• Software:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• Python: 3.8.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;• Pytorch: 2.2.0+cu121<br>
&nbsp;&nbsp;&nbsp;&nbsp;• transformers: 4.28.1<br>
&nbsp;&nbsp;&nbsp;&nbsp;• deepspeed: 0.13.1<br>
• Hardware:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• GPU: NVIDIA GeForce RTX 4090<br>
&nbsp;&nbsp;&nbsp;&nbsp;• CUDA: 12.1<br>
&nbsp;&nbsp;&nbsp;&nbsp;• Driver: 535.230.02<br>

# Data & Model<br>
You can download the dataset used in this project from the following link: [Download Dataset](https://github.com/unikcc/DiaASQ).<br>
We provide the implementation for ChatGLM3; the other models can be adapted by modifying their model loading code in the same way as for ChatGLM3.<br>
| Model Name |
|----------|
| [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)  | 
| [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)  | 
| [ChatGLM3-6B ](https://huggingface.co/THUDM/chatglm3-6b) |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)  |



# Running TaCoMoE
• Install the required packages:<br>
```bash
cd TaCoMoE/
pip install -r requirements.txt
```
• Train && Evaluate the TaCoMoE on the Chinese and English dataset:<br>
```bash
bash experiments/run_tacomoe.bash
```
