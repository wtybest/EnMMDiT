import torch
from pipeline_sd3point5_enmmdit import StableDiffusion3Pipeline
import numpy as np
from PIL import Image
import os
import string
from transformer_sd3point5_enmmdit import SD3Transformer2DModel
import torch.nn.functional as F
from attention_visul_utils_sd3point5 import AttentionStore, register_attention_control
import ast
import argparse
import json
import copy
import re
import random
def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--scale_factor', default=30, type=int)
    parser.add_argument('--use_tea_loss', default=False, action="store_true")
    parser.add_argument('--use_so_loss', default=False, action="store_true")
    parser.add_argument('--prompt_json', default="dataset/concise_prompt.json", type=str)
    parser.add_argument('--output_dir', default="outputs/sd3point5/concise_prompt_results", type=str)
    parser.add_argument('--gen_num_for_each_prompt', default=100, type=int)
    parser.add_argument('--refine_step_list', type=ast.literal_eval, default=[2, 3, 4, 5])
    parser.add_argument('--tea_lambda', default=0.2, type=float)
    parser.add_argument('--so_lambda', default=0.001, type=float)
    parser.add_argument('--json_start_idx', default=0, type=int)
    parser.add_argument('--json_end_idx', default=300, type=int)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()


    def sanitize_folder_name(prompt, max_length=60):
        safe_name = prompt.strip()
        safe_name = re.sub(r'[,\.\'"!\?:;]', '', safe_name)
        safe_name = safe_name.replace(" ", "_")
        safe_name = re.sub(r'[^\w\-]', '', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)
        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length]
            safe_name = safe_name.rstrip('_')
        return safe_name

    def load_and_transform_json(json_path):
        if not os.path.exists(json_path):
            print(f"Error: can't find {json_path}")
            return []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        transformed_list = []

        for item in data:
            prompt_text = item.get('prompt', '')
            subject_list = item.get('subject', [])
            entry = [prompt_text, [], subject_list]
            transformed_list.append(entry)
        return transformed_list

    

    def avg_gap_in_list(lst):
        gaps = [lst[i+1] - lst[i] for i in range(len(lst) - 1)]
        return sum(gaps) / len(gaps)

    def find_tightest_list(lists):
        avg_gaps = [avg_gap_in_list(lst) for lst in lists]
        tightest_list_index = avg_gaps.index(min(avg_gaps))
        return lists[tightest_list_index]

    def find_ascending_sequence(lists):
        result = []
        solutions_found = 0
        first_solution = None
        other_solution = []

        def backtrack(index, prev_value):
            nonlocal solutions_found, first_solution
            if index == len(lists):
                solutions_found += 1
                if solutions_found == 1:
                    first_solution = result[:]
                elif solutions_found > 1:
                    other_solution.append(copy.deepcopy(result))
                    # raise ValueError("Find the more than one solutioin for t-5!")
                return
            for num in sorted(lists[index]):
                if prev_value is None or num > prev_value:
                    result.append(num)
                    backtrack(index + 1, num)
                    result.pop()

        backtrack(0, None)
        if solutions_found == 1:
            return first_solution
        elif solutions_found > 1:
            other_solution.append(first_solution)
            tight_solutioin = find_tightest_list(other_solution)
            print(f'for t_5 token list {lists} finds {other_solution}, but return the tightest {tight_solutioin}')
            return tight_solutioin
        else:
            raise ValueError("There is no right solution for t-5!")

    def get_index_from_subject(pipe, input_prompt, subject_list):
        clip_prompt_token_list = pipe.tokenizer.encode(input_prompt)
        t5_prompt_token_list = pipe.tokenizer_3.encode(input_prompt)

        final_index_list = []
        for subject_idx, subject_word in enumerate(subject_list):
            final_index_list.append([[],[]])#first clip, second t5
            clip_word_token_list = pipe.tokenizer.encode(subject_word)[1:-1]
            t5_word_token_list = pipe.tokenizer_3.encode(subject_word)[0:-1]
            t5_word_token_list = [each_t5_token for each_t5_token in t5_word_token_list if each_t5_token not in [0,1,2,3]]
            for each_clip_word_token in clip_word_token_list:
                final_index_list[subject_idx][0].append(clip_prompt_token_list.index(each_clip_word_token))
            t_5_token_candidate_list = []
            for each_t5_word_token in t5_word_token_list:
                t_5_token_candidate_list.append([i+77 for i, x in enumerate(t5_prompt_token_list) if x == each_t5_word_token])
            final_index_list[subject_idx][1] = find_ascending_sequence(t_5_token_candidate_list)
        return final_index_list

    def concat_images_vertically(image_list, output_dir, pre_fix):
        for i in range(0, len(image_list), 4):
            images = image_list[i:i+4]
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)

            new_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                new_image.paste(img, (0, y_offset))
                y_offset += img.height
            output_path = os.path.join(output_dir, f'{pre_fix}_block_{i}-{i+len(images)-1}.jpg')
            new_image.save(output_path)


    def fill_list(lst, target_length):
        while len(lst) < target_length:
            new_number = random.randint(0, 60000)
            if new_number not in lst:
                lst.append(new_number)
        return lst



    prompt_seed_list = load_and_transform_json(args.prompt_json)[args.json_start_idx:args.json_end_idx]
    for each_prompt_seed in prompt_seed_list:
        each_prompt = each_prompt_seed[0]
        seed_list = each_prompt_seed[1]
        seed_list = fill_list(seed_list, args.gen_num_for_each_prompt)
        subject_list = each_prompt_seed[2]
        prompt_dir = sanitize_folder_name(each_prompt, max_length=70)
        output_dir = os.path.join(args.output_dir, prompt_dir)
        os.makedirs(output_dir, exist_ok=True)
        exist_seed_list = os.listdir(output_dir)
        exist_seed_list = [int(re.search(r'\d+', seed_name).group()) for seed_name in exist_seed_list]
        seed_list = [item for item in seed_list if item not in exist_seed_list][0:(args.gen_num_for_each_prompt-len(exist_seed_list))]
        if len(seed_list) == 0:
            print(output_dir, f'is already has {args.gen_num_for_each_prompt} results!')

        for seed in seed_list:
            result_img_list = []
            torch.cuda.empty_cache()

            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
            sd3_transfomer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch.float16)
            pipe.transformer = sd3_transfomer
            pipe = pipe.to("cuda")
            tokenizer_list = [pipe.tokenizer, pipe.tokenizer_3]
            controller = AttentionStore()
            register_attention_control(pipe, controller)

            controller.reset()
            indices_to_alter_list = get_index_from_subject(pipe, each_prompt, subject_list)

            image_0, _ = pipe(prompt=each_prompt, attention_store=controller, num_inference_steps=28,guidance_scale=7.0,seed=seed,\
                        indices_to_alter=indices_to_alter_list, run_standard_sd=True, smooth_attentions=True,\
                            early_return_block_idx=12, use_ba_loss=True, scale_factor=args.scale_factor, use_tea_loss=args.use_tea_loss,\
                                  use_so_loss=args.use_so_loss, refine_step_list=args.refine_step_list, tea_lambda=args.tea_lambda,\
                                      so_lambda=args.so_lambda, derive_restrict_mask=False)
            image_0 = image_0.images[0].resize((256,256))
            result_img_list.append(np.array(image_0))

            torch.cuda.empty_cache()

            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
            sd3_transfomer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch.float16)
            pipe.transformer = sd3_transfomer
            pipe = pipe.to("cuda")
            tokenizer_list = [pipe.tokenizer, pipe.tokenizer_3]
            controller = AttentionStore()
            register_attention_control(pipe, controller)

            controller.reset()      
            image_2, _ = pipe(prompt=each_prompt, attention_store=controller, num_inference_steps=28,guidance_scale=7.0,seed=seed,\
                        indices_to_alter=indices_to_alter_list, run_standard_sd=False, smooth_attentions=True,\
                            early_return_block_idx=12, use_ba_loss=True, scale_factor=args.scale_factor, use_tea_loss=args.use_tea_loss,\
                                  use_so_loss=args.use_so_loss, refine_step_list=args.refine_step_list, tea_lambda=args.tea_lambda,\
                                      so_lambda=args.so_lambda, derive_restrict_mask=True)
            image_2 = image_2.images[0].resize((256,256))
            result_img_list.append(np.array(image_2))

            final_img = Image.fromarray(np.concatenate(result_img_list, axis=1))
            final_img.save(f'{output_dir}/seed_{str(seed)}.jpg')



