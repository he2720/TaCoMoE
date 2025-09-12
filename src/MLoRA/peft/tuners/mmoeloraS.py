# -*- encoding: utf-8 -*-
# here put the import lib
import re
import importlib
import warnings
from dataclasses import dataclass, field
from .mmoelora import MMOELoraModel, MMOELoraLinear, MMOELoraLayer
from .lora import LoraConfig
import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import _get_submodules, transpose, PeftType
import torch.optim as optim
def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None

from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
@dataclass
class MMOELoraConfigS(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """
    task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MMOELORAS



class MMOELoraModelS(MMOELoraModel):

    def __init__(self, model, config, adapter_name):

        super().__init__(model, config, adapter_name)
        print('====loading MMOELoraModelS=====')



    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            # find the corresponding modules. target module has been split into list.
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, MMOELoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        raise NotImplementedError
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = MMOELoraLinearS(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )


def NCELoss(A_out, expert, temperature):
    temperature = 0.07
   
   
   
    expert=expert.unsqueeze(0).unsqueeze(-1)
    
    Gate = A_out.mean(dim=-1, keepdim=True)*expert
    
    seq_len, batch_size,  n, dim = A_out.shape  
    experts_select = (Gate > 1e-4).bool()
    A_out = A_out / A_out.norm(dim=-1, keepdim=True)
    experts_select = experts_select.permute(2, 1, 0, 3 ).reshape(n, batch_size * seq_len)
    p_mask = experts_select.unsqueeze(2) * experts_select.unsqueeze(1)
   
    A_out = A_out.permute(2, 1, 0, 3).reshape(n, batch_size * seq_len, dim)                       
    mask = ~torch.eye(batch_size * seq_len, dtype=bool, device=A_out.device)
    product = torch.matmul(A_out, A_out.transpose(1, 2))
    clip_value = 10.0
  
    product = torch.exp((product / temperature).clamp(-clip_value, clip_value) )*p_mask.float() 
    denominator = product.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p_product = (product / denominator).clamp(min=1e-8, max=1e8)
   # p_product = product
   
    #loss = -torch.log(p_product / product.sum(dim=-1, keepdim=True))
    loss = -torch.log(p_product)
    return loss.mean()
   
    
    return loss.mean()
class MMOELoraLinearS(MMOELoraLinear):

    def __init__(self, 
                 adapter_name: str, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 0, 
                 lora_alpha: int = 1, 
                 lora_dropout: float = 0, 
                 fan_in_fan_out: bool = False, 
                 **kwargs):
        
        super().__init__(adapter_name, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, **kwargs)


    def unmerge(self, expert_weight):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
                self.weight.data -= (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
                    * expert_weight[..., i]
                )
            self.merged = False


    def forward(self, x: torch.Tensor, **kwargs):
        
        expert_weight = kwargs["task_id"]
        #print('x.dtype',x.dtype)
        previous_dtype = x.dtype
        seq_len, batch_size, _ = x.size()
       
        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge(expert_weight)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:   # general lora process
            #print('general lora process')
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias).permute(1, 0, 2)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            nce_loss = 1e-8
            E = []
            for i in range(self.expert_num):
                A_out = self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)).reshape(seq_len, batch_size,  1, -1)
                #print('175',result)
                
                E.append(A_out)
                result += ( # lora process
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * expert_weight[..., i].unsqueeze(-1).unsqueeze(0)
                )
              
            
            E = torch.cat(E, dim=2)
        #    nce_loss = NCELoss(E, expert_weight, 0.07)
           
            
            
            

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
       
        return result,expert_weight,E
