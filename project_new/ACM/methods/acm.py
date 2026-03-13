import torch
import copy
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import re
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

import seaborn as sns
import logging
from copy import deepcopy
import json
# from scipy.stats import entropy

from utils.utils import *


def get_models(finetuned_model_names, pretrained_model_name):
    models_to_merge, finetuned_tokenizers, finetuned_configs = [], [], []
    for finetuned_model_name in finetuned_model_names:
        try:
            finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, device_map='cpu',
                                                                   torch_dtype=torch.bfloat16)
            finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
            finetuned_config = AutoConfig.from_pretrained(finetuned_model_name)
            models_to_merge.append(finetuned_model)
            finetuned_tokenizers.append(finetuned_tokenizer)
            finetuned_configs.append(finetuned_config)
        except Exception as e:
            print(f"Model {finetuned_model_name} could not be loaded.")
            print(f"Reason: {e}")

    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                            torch_dtype=torch.bfloat16)

    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
    pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)

    return models_to_merge, finetuned_tokenizers, finetuned_configs, pretrained_model, pretrained_tokenizer, pretrained_config


@torch.no_grad()
def get_calib_feat(model, tokenizer, dataset=None):
    input_dict = dict()

    def max_hook(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]

        y_max = y.view(-1, y.shape[-1]).cpu().detach()

        if name not in input_dict:
            input_dict[name] = [y_max]
        else:
            input_dict[name] += [y_max]

    hooks = []
    for name, m in model.named_modules():
        if 'down_proj' in name or "embed" in name or name == "model.norm" or "lm_head" in name:
        # if 'down_proj' in name or name == "model.norm":
            hooks.append(
                m.register_forward_hook(
                    partial(max_hook, name=name)))

    device = model.device

    samples = get_calib_dataset(tokenizer, dataset=dataset)
    pbar = tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()

    return input_dict

def get_calib_dataset(tokenizer=None, dataset=None):
    data_size = len(dataset)

    target_cluster_ids = {str(i) for i in range(20)}
    counter = {cid: 0 for cid in target_cluster_ids}

    samples = []

    for data in dataset:
        cid = data["cluster_id"]

        if cid not in target_cluster_ids:
            continue
        if counter[cid] >= 5: # items for each cluster
            continue

        line = data["sentence"].strip()
        line_encoded = tokenizer.encode(line)

        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue

        samples.append(sample)
        counter[cid] += 1

        if all(count >= 5 for count in counter.values()):
            break

    return samples

def align_tokenizers_and_embeddings(pretrained_model, pretrained_tokenizer, pretrained_config,
                                    finetuned_models, finetuned_tokenizers,
                                    finetuned_configs, logger: logging.Logger):

    pretrained_vocab_size = pretrained_config.vocab_size
    try:
        models_vocab_size = [pretrained_vocab_size]
        logger.info(f"Vocab size of pretrained model is {pretrained_vocab_size}.")
        pretrained_token_dict = json.loads(pretrained_tokenizer._tokenizer.to_str())
        pretrained_added_pad_tokens = [token_dict for token_dict in pretrained_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
        assert pretrained_added_pad_tokens == []
        models_added_pad_tokens_list = [(True, pretrained_added_pad_tokens)]

        added_pad_tokens_set = set()
        for index, (finetuned_tokenizer, finetuned_config) in enumerate(zip(finetuned_tokenizers, finetuned_configs)):
            finetuned_vocab_size = finetuned_config.vocab_size
            models_vocab_size.append(finetuned_vocab_size)
            finetuned_token_dict = json.loads(finetuned_tokenizer._tokenizer.to_str())
            finetuned_added_pad_tokens = [token_dict for token_dict in finetuned_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
            logger.info(f"Vocab size of index {index} finetuned model is {finetuned_vocab_size}.")
            logger.info(f"Added pad tokens of index {index} finetuned model is {finetuned_added_pad_tokens}.")
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            if finetuned_vocab_size - pretrained_vocab_size < len(finetuned_added_pad_tokens):
                logger.warning(f"Vocab size in index {index} finetuned model's config mismatches (less than) number of added tokens.")
                logger.warning(f"Before removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                for _ in range(len(finetuned_added_pad_tokens) - (finetuned_vocab_size - pretrained_vocab_size)):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning(f"Remove pad token {removed_pad_token}.")
                    assert removed_pad_token["content"] in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(f"After removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                is_matched = False
            else:
                assert finetuned_vocab_size - pretrained_vocab_size == len(finetuned_added_pad_tokens)
                is_matched = True
            for token_dict in finetuned_added_pad_tokens:
                added_pad_tokens_set.add(token_dict["content"])
            models_added_pad_tokens_list.append((is_matched, [token_dict["content"] for token_dict in finetuned_added_pad_tokens]))
        logger.info(f"All added pad tokens of finetuned models are {added_pad_tokens_set}.")

        # align the tokenizers
        aligned_models_vocab_size_set = set()
        for index, (model, tokenizer, model_vocab_size) in enumerate(zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers, models_vocab_size)):
            is_matched = models_added_pad_tokens_list[index][0]
            model_added_pad_tokens_list = models_added_pad_tokens_list[index][1]
            for added_pad_token in added_pad_tokens_set:
                if is_matched and added_pad_token in model_added_pad_tokens_list:
                    logger.info(f"Skip added pad token {added_pad_token} of index {index} model since its original added pad tokens and token embeddings are matched.")
                    continue
                num_new_tokens = tokenizer.add_special_tokens({"pad_token": added_pad_token})
                if num_new_tokens > 0:
                    assert num_new_tokens == 1
                    model_vocab_size = model_vocab_size + num_new_tokens

                    model.resize_token_embeddings(new_num_tokens=model_vocab_size)

                    # shape (new_num_tokens, embed_dim)
                    input_embeddings = model.get_input_embeddings().weight.data
                    output_embeddings = model.get_output_embeddings().weight.data

                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

            logger.info(f"Aligned index {index} model: input token embedding shape {model.get_input_embeddings().weight.shape}, "
                        f"output token embedding shape {model.get_output_embeddings().weight.shape}, "
                        f"tokenizer added tokens {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}.")
            aligned_models_vocab_size_set.add(model.model.embed_tokens.weight.shape)
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.error(e)
        logger.warning(f"Unable to align tokenizers by default function, using alternative smart_tokenizer_and_embedding_resize function.")
        for model, tokenizer in zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers):
            smart_tokenizer_and_embedding_resize(special_tokens_dict={"pad_token": "<special_pad>"},
                                                 tokenizer=tokenizer, model=model, pretrained_vocab_size=pretrained_vocab_size)


def smart_tokenizer_and_embedding_resize(special_tokens_dict: dict, tokenizer, model, pretrained_vocab_size: int):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(pretrained_vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def acm_merge(merged_model_name, pretrained_model_name, theta):
    try:
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
        base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        base_config = AutoConfig.from_pretrained(pretrained_model_name)
    except Exception as e:
        print(f"Model {pretrained_model_name} could not be loaded.")
        print(f"Reason: {e}")

    try:
        merged_model = AutoModelForCausalLM.from_pretrained(merged_model_name, torch_dtype=torch.bfloat16)
        # merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_name)
        merged_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        merged_config = AutoConfig.from_pretrained(merged_model_name)
    except Exception as e:
        print(f"Model {merged_model_name} could not be loaded.")
        print(f"Reason: {e}")

    logger = logging.getLogger(__name__)
    align_tokenizers_and_embeddings(pretrained_model=base_model, pretrained_tokenizer=base_tokenizer,
                                    pretrained_config=base_config, finetuned_models=[merged_model],
                                    finetuned_tokenizers=[merged_tokenizer], finetuned_configs=[merged_config],
                                    logger=logger)

    print('loading calibration dataset...')
    #dataset = load_dataset("/leonardo/home/userexternal/miacobel/project_new/ACM/train.jsonl", split="train")
    dataset = load_dataset(
        "json",  
        data_files="/leonardo/home/userexternal/miacobel/project_new/ACM/calibration/train.jsonl",
        split="train"
    )
    dataset = dataset.shuffle(seed=54)

    print('getting calibration features...')
    base_model = base_model.to('cuda')

    pretrained_scale_dict = get_calib_feat(base_model, base_tokenizer, dataset)
    base_model = base_model.to('cpu')

    merged_model = merged_model.to('cuda')
    merged_scale_dict = get_calib_feat(merged_model, merged_tokenizer, dataset)
    merged_model = merged_model.to('cpu')


    base_layer_dict = {}
    for name, param in base_model.named_modules():
        if hasattr(param, 'weight'):
            base_layer_dict[name] = param

    merged_layer_dict = {}
    for name, param in merged_model.named_modules():
        if hasattr(param, 'weight'):
            merged_layer_dict[name] = param

    final_weight_dict, final_bias_dict = {}, {}

    count,sum_tx = 0,[]
    t_x = {}
    for name, param in merged_model.named_modules():
        if 'down_proj' in name or "embed" in name or name == "model.norm" or "lm_head" in name:
            if "embed" in name:
                number = -1
            elif name == "model.norm":
                number = -2
            elif "lm_head" in name:
                number = -3
            else:
                number = re.findall(r'\d+', name)
                number = int(number[0])

            mi, nmi, nmi_max = [],[],[]
            for k,v in zip(pretrained_scale_dict[name],merged_scale_dict[name]):
                mii = mutual_info_score(k.float().numpy().flatten(),v.float().numpy().flatten())
                mi.append(mii)

                H_X = entropy(k.float().numpy().flatten())
                H_Y = entropy(v.float().numpy().flatten())

                nmi_max.append(mii / max(H_X, H_Y))
                nmi.append(mii / (H_X + H_Y - mii))

            t = theta
            x = max(mi)
            scaled_x = []
            for m in mi:
                scaled_x.append(1 / (1 + np.exp(-t * (m - x))))


            t_x[number] = sum(scaled_x) / len(scaled_x)

            count+=1


    for name, param in merged_model.named_modules():
        if hasattr(param, 'weight'):
            if "embed" in name:
                number = -1
            elif name == "model.norm":
                number = -2
            elif "lm_head" in name:
                number = -3
            else:
                number = re.findall(r'\d+', name)
                number = int(number[0])

            total_delta = merged_layer_dict[name].weight.data - base_layer_dict[name].weight.data

            delta_final = total_delta * (1 - t_x[number])
            new_name = name + '.weight'

            final_weight_dict[new_name] = (base_layer_dict[name].weight.data    + delta_final).to(torch.bfloat16)


            if hasattr(param, 'bias') and param.bias != None:
                bias_delta = merged_layer_dict[name].bias.data - base_layer_dict[name].bias.data
                bias_final = bias_delta * (1 - t_x[number])
                new_name = name + '.bias'
                final_bias_dict[new_name] = (base_layer_dict[name].bias.data   + bias_final).to(torch.bfloat16)


    new_merged_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
    new_merged_model = copy_params_to_model(final_weight_dict,final_bias_dict,new_merged_model)

    return new_merged_model, merged_tokenizer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--merged_model", type=str, default= "Your model path", help="Name of the fine-tuned model to use for merging.")
    argparser.add_argument("--pretrained_model_name", type=str, default= "Your model path", help="Name of the pretrained model.")
    argparser.add_argument("--save_path", type=str,
                        default="Your save path")
    argparser.add_argument("--theta", type=float,
                           default=0.7)

    args = argparser.parse_args()

    print(f"Merging {args.merged_model}...")
    merged_model, merged_tokenizer = acm_merge(args.merged_model,pretrained_model_name = args.pretrained_model_name, theta=args.theta)
    print(f"Saving to {args.save_path}...")
    merged_model.save_pretrained(args.save_path,torch_dtype=torch.bfloat16)
    merged_tokenizer.save_pretrained(args.save_path)
