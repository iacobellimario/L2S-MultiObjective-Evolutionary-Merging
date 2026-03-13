import os
import re
import copy
import json
import torch
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def weighted_task_vector(self, coeffs):
        """
        Apply layer-wise coefficients to task vector parameters
        :param coeffs: numpy array of coefficients for each layer
        """
        for param_name, param_value in self.task_vector_param_dict.items():
            if "layers" in param_name:
                layer_index = param_name.split(".")[2]
                coe = coeffs[int(layer_index)]
                self.task_vector_param_dict[param_name] = coe * param_value
    
    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict, tokenizer, model
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_llm(model_path, device="cpu"):
    """
    Load a large language model from a specified path
    :param model_path: path to the model files
    :param device: device to load the model onto (cpu or cuda)
    :return: loaded model
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict = True,
        low_cpu_mem_usage = True,
    )
    model.seq_len = model.config.max_position_embeddings
    model.to(device)
    model.eval()
    return model


def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def softmax(x, T):
    """
    Apply softmax with temperature parameter T.
    Higher T makes the distribution more uniform, lower T makes it more peaked.
    
    :param x: input vector
    :param T: temperature parameter
    :return: softmax output
    """
    x_scaled = x / T
    exp_x = np.exp(x_scaled)
    return exp_x / np.sum(exp_x, axis=0)


def load_sensitivity_scores(json_path):
    """
    Load sensitivity scores from a JSON file.
    
    :param json_path: Path to the JSON file containing sensitivity scores
    :return: numpy array of sensitivity scores
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if "sensitivity_scores" in data:
            return np.array(data["sensitivity_scores"])
        else:
            raise KeyError(f"No 'sensitivity_scores' found in {json_path}")
            
    except Exception as e:
        print(f"Error loading sensitivity scores from {json_path}: {e}")
        return None


def weighted_task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, 
                            math_coeffs, distill_coeffs, scaling_coefficient=1.0):
    """
    Perform weighted task arithmetic between base model and distilled model
    
    :param merged_model: base math model
    :param models_to_merge: list of models to merge [math_model, distill_model]
    :param exclude_param_names_regex: list of regex patterns to exclude parameters
    :param math_coeffs: coefficients for math model layers
    :param distill_coeffs: coefficients for distill model layers
    :param scaling_coefficient: global scaling factor for distill task vector
    :return: merged parameters
    """
    distill_model = models_to_merge[1]

    distill_task_vector = TaskVector(pretrained_model=merged_model, finetuned_model=distill_model, exclude_param_names_regex=exclude_param_names_regex) 
    distill_vector_param_dict = copy.deepcopy(distill_task_vector.task_vector_param_dict)
    
    merged_params = dict()

    with torch.no_grad():
        for param_name, param_value in merged_model.named_parameters():
            if "layers" in param_name:
                layer_index = param_name.split(".")[2]
                math_coe = math_coeffs[int(layer_index)]
                distill_coe = distill_coeffs[int(layer_index)]
                merged_params[param_name] = math_coe * param_value + distill_coe * scaling_coefficient * distill_vector_param_dict[param_name] 
            else:
                merged_params[param_name] = param_value + scaling_coefficient * distill_vector_param_dict[param_name]
            
    return merged_params


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Layer-wise Weighted Model Merging Tool")
    
    # Model paths
    parser.add_argument("--math_model", type=str, required=True, 
                        help="Path to the math-specialized model")
    parser.add_argument("--distill_model", type=str, required=True, 
                        help="Path to the distilled model")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the merged model")
    
    # Sensitivity scores
    parser.add_argument("--math_sensitivity_file", type=str, required=True,
                        help="Path to JSON file with math model sensitivity scores")
    parser.add_argument("--distill_sensitivity_file", type=str, required=True,
                        help="Path to JSON file with distill model sensitivity scores")
    
    # Merging parameters
    parser.add_argument("--softmax_temperature", type=float, default=3.0,
                        help="Temperature parameter for softmax normalization")
    parser.add_argument("--scaling_coefficient", type=float, default=0.7,
                        help="Global scaling coefficient for distill task vector")
    parser.add_argument("--coeff_multiplier", type=float, default=2.0,
                        help="Number of models to be merged")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for computation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID if using CUDA")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set device
    device = args.device
    if device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Load model paths from arguments
    math_model_path = args.math_model
    distill_model_path = args.distill_model
    merged_model_path = args.output_model
    
    # Load tokenizers
    print("Loading tokenizers...")
    math_tokenizer = AutoTokenizer.from_pretrained(math_model_path)
    
    # Load models
    print(f"Loading models to {device}...")
    math_model = load_llm(math_model_path, device)
    distill_model = load_llm(distill_model_path, device)
    
    models_to_merge = [math_model, distill_model]
    
    # Load sensitivity scores from JSON files
    print("Loading sensitivity scores...")
    math_l2_norm = load_sensitivity_scores(args.math_sensitivity_file)
    distill_l2_norm = load_sensitivity_scores(args.distill_sensitivity_file)
    
    if math_l2_norm is None or distill_l2_norm is None:
        print("Failed to load sensitivity scores. Exiting.")
        return
    
    # Check if the arrays have the same length
    if len(math_l2_norm) != len(distill_l2_norm):
        print(f"Warning: Sensitivity score arrays have different lengths: {len(math_l2_norm)} vs {len(distill_l2_norm)}")
        # Use the minimum length to avoid index errors
        min_length = min(len(math_l2_norm), len(distill_l2_norm))
        math_l2_norm = math_l2_norm[:min_length]
        distill_l2_norm = distill_l2_norm[:min_length]
        print(f"Using the first {min_length} scores from each file.")
    
    # Prepare layer-wise coefficients based on L2 norms
    alpha = args.scaling_coefficient
    norm_vectors = np.vstack([
        (1 - alpha) * math_l2_norm, 
        alpha * distill_l2_norm
    ])
    num_layers = len(math_l2_norm)
    
    # Apply softmax with temperature to get per-layer weights
    print(f"Calculating layer coefficients with temperature {args.softmax_temperature}...")
    softmax_results = np.zeros((2, num_layers))
    for i in range(num_layers):
        softmax_results[:, i] = softmax(norm_vectors[:, i], T=args.softmax_temperature)
    
    math_coeff = softmax_results[0, :]
    distill_coeff = softmax_results[1, :]

    # Apply coefficient multiplier
    math_coeff = args.coeff_multiplier * math_coeff
    distill_coeff = args.coeff_multiplier * distill_coeff
    
    # Print coefficient statistics
    print(f"Math coefficients (mean, min, max): {math_coeff.mean():.4f}, {math_coeff.min():.4f}, {math_coeff.max():.4f}")
    print(f"Distill coefficients (mean, min, max): {distill_coeff.mean():.4f}, {distill_coeff.min():.4f}, {distill_coeff.max():.4f}")
    
    # Merge models with weighted coefficients
    print("Merging models...")
    merged_params = weighted_task_arithmetic(
        merged_model=math_model, 
        models_to_merge=models_to_merge, 
        exclude_param_names_regex=[], 
        math_coeffs=math_coeff, 
        distill_coeffs=distill_coeff, 
        scaling_coefficient=args.scaling_coefficient
    )
    
    # Copy merged parameters to the model
    print("Copying parameters to output model...")
    copy_params_to_model(params=merged_params, model=math_model)
    
    # Save the merged model
    print(f"Saving merged model to {merged_model_path}...")
    math_model.save_pretrained(save_directory=merged_model_path)
    math_tokenizer.save_pretrained(save_directory=merged_model_path)
    
    print("Merging complete!")


if __name__ == "__main__":
    main()