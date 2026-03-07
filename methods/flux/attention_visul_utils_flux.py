import torch
import numpy as np
import string
import abc
from PIL import Image
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
import os
import cv2
import re

class FluxAttnProcessor2_0_modified:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attnstore, name_in_transform):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0_modified requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attnstore = attnstore
        self.name_in_transform = name_in_transform

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)


        #below change to 1.0

        query = query.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)
        key = key.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)
        value = value.transpose(1, 2).contiguous().view(batch_size, -1, inner_dim)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)
        self.attnstore(attention_probs, self.name_in_transform)#!!!for single input wo clg

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)  

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

def register_attention_control(model, controller):
    attn_procs = {}
    att_count = 0
    for name in model.transformer.attn_processors.keys():
        att_count += 1
        attn_procs[name] = FluxAttnProcessor2_0_modified(attnstore=controller, name_in_transform=name)
    model.transformer.set_attn_processor(attn_procs)
    controller.num_att_layers = att_count



class AttentionControl(abc.ABC):

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, name_in_transform: str):
        raise NotImplementedError

    def __call__(self, attn, name_in_transform: str):
        self.forward(attn, name_in_transform)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.between_steps()

    def reset(self):
        self.cur_att_layer = 0

    def __init__(self):
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {f"block_{i}": [[],[]] for i in range(57)}
    @staticmethod
    def get_empty_gpu_store():
        return {f"block_{i}": [] for i in range(57)}
    
    def crop_attn_into_cross_part(self, attn):
        head_avg_attn = torch.mean(attn, dim=0)
        cross_attn_1 = head_avg_attn[0:self.text_token_len, self.text_token_len::].detach().cpu()
        cross_attn_2 = head_avg_attn[self.text_token_len::, 0:self.text_token_len].detach().cpu()
        return cross_attn_1, cross_attn_2

    def get_gpu_attn_score(self, attn):
        head_avg_attn = torch.mean(attn, dim=0)
        gpu_cross_attn_1 = head_avg_attn[0:self.text_token_len, self.text_token_len::]
        gpu_cross_attn_2 = head_avg_attn[self.text_token_len::, 0:self.text_token_len]
        gpu_cross_attn = ((gpu_cross_attn_1 + gpu_cross_attn_2.transpose(0,1)) / 2).reshape(-1, 64, 64)
        return gpu_cross_attn

    def forward(self, attn, name_in_transform: str):
        this_idx = int(re.search(r'transformer_blocks\.(\d+)\.attn\.processor', name_in_transform).group(1))
        if 'single' in name_in_transform:
            this_idx += 19
        key = 'block_'+str(this_idx)
        cross_attn_1, cross_attn_2 = self.crop_attn_into_cross_part(attn)
        self.step_store[key][0].append(cross_attn_1)
        self.step_store[key][1].append(cross_attn_2)
  
        if this_idx in self.gpu_store_block_range:
            gpu_cross_attn = self.get_gpu_attn_score(attn)
            self.gpu_step_store[key].append(gpu_cross_attn)

        return attn

    def between_steps(self):
        self.cur_att_layer = 0
        self.attention_store = self.step_store
        self.gpu_attention_store = self.gpu_step_store

        self.step_store = self.get_empty_store()
        self.gpu_step_store = self.get_empty_gpu_store()

    def save_global_store(self):
        with torch.no_grad():
            if len(self.global_store) == 0:
                self.global_store = self.attention_store
            else:
                for key in self.global_store:
                    self.global_store[key][0].append(self.attention_store[key][0][0])
                    self.global_store[key][1].append(self.attention_store[key][1][0])

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.gpu_step_store = self.get_empty_gpu_store()
        self.attention_store = {}
        self.gpu_attention_store = {}
        self.global_store = {}

    def __init__(self, gpu_store_start_idx=0):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.gpu_step_store = self.get_empty_gpu_store()
        self.attention_store = {}
        self.gpu_attention_store = {}
        self.global_store = {}
        self.text_token_len = 512
        self.gpu_store_block_range = list(range(gpu_store_start_idx,28))

def get_avg_attn_for_select_step(attn_list, tokenizer, prompt_input, select_step=0):
    cross_attn_1_list = []
    cross_attn_2_list= []
    for each_key in attn_list.keys():
        cross_attn_1_list.append(attn_list[each_key][0][select_step])
        cross_attn_2_list.append(attn_list[each_key][1][select_step])
    avg_attn_1 = torch.mean(torch.stack(cross_attn_1_list), dim=0)
    avg_attn_2 = torch.mean(torch.stack(cross_attn_2_list), dim=0)
    cross_avg_attn = ((avg_attn_1 + avg_attn_2.transpose(0,1)) / 2).reshape(-1, 64, 64)

    t5_tokens = tokenizer.encode(prompt_input)
    t5_token_len = len(t5_tokens)
    t5_tokens_str = [tokenizer.decode(int(each_t5_token)) for each_t5_token in t5_tokens]
    cross_avg_attn_t5 = cross_avg_attn[0:t5_token_len,:,:]
    return [cross_avg_attn_t5, t5_tokens_str]

def gen_select_attn_for_select_step(attn_list, tokenizer, prompt_input, select_step=0, select_block=0):
    select_attn_1 = attn_list[f'block_{select_block}'][0][select_step]
    select_attn_2 = attn_list[f'block_{select_block}'][1][select_step]
    cross_select_attn = ((select_attn_1 + select_attn_2.transpose(0,1)) / 2).reshape(-1, 64, 64)

    t5_tokens = tokenizer.encode(prompt_input)
    t5_token_len = len(t5_tokens)
    t5_tokens_str = [tokenizer.decode(int(each_t5_token)) for each_t5_token in t5_tokens]
    cross_select_attn_t5 = cross_select_attn[0:t5_token_len,:,:]
    return [cross_select_attn_t5, t5_tokens_str]


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    image = cv2.resize(image, (256,256))
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def concat_images(images):
    img_height, img_width, img_channels = images[0].shape
    num_images = len(images)
    row_images = []
    if num_images <= 10:
        row_image = np.concatenate(images, axis=1)
        row_images.append(row_image)
    else:
        row_image = np.concatenate(images[:10], axis=1)
        row_images.append(row_image)
    
    for i in range(1, (num_images + 9) // 10):
        row_images_list = images[i*10:(i+1)*10]
        if len(row_images_list) < 10:
            white_image = np.ones((img_height, img_width, img_channels), dtype=np.uint8) * 255
            row_images_list.extend([white_image] * (10 - len(row_images_list)))
        row_image = np.concatenate(row_images_list, axis=1)
        row_images.append(row_image)

    final_image = np.concatenate(row_images, axis=0)
    return final_image

def visual_avg_cross_attn(cross_avg_attn, token_strs):
    assert cross_avg_attn.shape[0] == len(token_strs), f"Shapes do not match"
    result_img_list = []
    for idx in range(cross_avg_attn.shape[0]):
        single_cross_attn = cross_avg_attn[idx]
        single_cross_attn = (single_cross_attn - single_cross_attn.min()) / (single_cross_attn.max() - single_cross_attn.min())
        single_cross_attn = np.repeat(np.uint8(255 * single_cross_attn.numpy())[..., np.newaxis], repeats=3, axis=-1)
        single_cross_attn_with_str = text_under_image(single_cross_attn, token_strs[idx])
        result_img_list.append(single_cross_attn_with_str)

    final_img = Image.fromarray(concat_images(result_img_list))
    return final_img



