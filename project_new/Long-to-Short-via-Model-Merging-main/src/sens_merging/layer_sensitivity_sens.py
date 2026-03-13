import os
import gc
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.nn import functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM


class SensitivityAnalyzer:
    """Analyzes layer sensitivity in large language models using gradient information."""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the sensitivity analyzer with a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text for the model.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tokenized input ready for model consumption
        """
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors='pt'
        ).to(self.device)

        result["labels"] = result["input_ids"].clone()
        return result
    
    def calculate_sensitivity(self, sample: Dict, 
                             prompt_format: str = "instruction_output", 
                             loss_on_output: bool = True) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Calculate sensitivity for a single sample.
        
        Args:
            sample: Dictionary containing input data
            prompt_format: Format of the prompt ("instruction_output", "question_dpsk", "question_cot")
            loss_on_output: Whether to calculate loss only on the output tokens
            
        Returns:
            Tuple of (loss, sensitivity dictionary)
        """
        sample_sensitivity = {}
        
        # Format prompt based on the specified format
        if prompt_format == "instruction_output":
            full_prompt = f"<|instruction|>\n{sample['instruction'].strip()}\n\n<|output|>\n{sample['output'].strip()}{self.tokenizer.eos_token}"
            user_prompt = f"<|instruction|>\n{sample['instruction'].strip()}\n\n<|output|>\n" if loss_on_output else None
            
        elif prompt_format == "question_dpsk":
            full_prompt = f"<|question|>\n{sample['question'].strip()}\n\n<|dpsk_res|>\n{sample['dpsk_res'].strip()}{self.tokenizer.eos_token}"
            user_prompt = f"<|question|>\n{sample['question'].strip()}\n\n<|dpsk_res|>\n" if loss_on_output else None
            
        elif prompt_format == "question_cot":
            full_prompt = f"<|question|>\n{sample['question'].strip()}\n\n<|cot_solution|>\n{sample['cot_solution'].strip()}{self.tokenizer.eos_token}"
            user_prompt = f"<|question|>\n{sample['question'].strip()}\n\n<|cot_solution|>\n" if loss_on_output else None
            
        else:
            raise ValueError(f"Unsupported prompt format: {prompt_format}")

        tokenized_full_prompt = self.tokenize(full_prompt)

        # Mask loss for prompt tokens if needed
        if loss_on_output and user_prompt:
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = tokenized_user_prompt["input_ids"].size(1)
                
            tokenized_full_prompt["labels"] = torch.cat([
                torch.full((1, user_prompt_len), -100, dtype=torch.long, device=self.device),
                tokenized_full_prompt["labels"][:, user_prompt_len:]
            ], dim=-1)

        # Forward pass with gradient tracking
        self.model.zero_grad()
        outputs = self.model(**tokenized_full_prompt)
        loss = outputs.loss
        loss.backward()

        loss_value = loss.item()

        # Calculate sensitivity for each parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                cur_sensitivity = torch.abs(param * param.grad)
                cur_sensitivity = cur_sensitivity.detach()
                sample_sensitivity[name] = cur_sensitivity

        # Clear memory
        del full_prompt, tokenized_full_prompt, outputs, loss
        self.model.zero_grad()
        gc.collect()
        torch.cuda.empty_cache() if self.device == "cuda" else None

        return loss_value, sample_sensitivity
    
    def select_top_k_layers(self, sensitivity: Dict[str, torch.Tensor], k: int = 5) -> torch.Tensor:
        """
        Select the top-k most sensitive layers.
        
        Args:
            sensitivity: Dictionary mapping parameter names to sensitivity values
            k: Number of layers to select
            
        Returns:
            Normalized layer sensitivity scores
        """
        layer_sensitivity = defaultdict(float)
        
        # Aggregate sensitivity by layer
        for name in sensitivity:
            if "layers" in name:
                # Extract layer index (adapt this for different model architectures)
                layer_parts = name.split(".")
                layer_idx_pos = layer_parts.index("layers") + 1 if "layers" in layer_parts else -1
                
                if layer_idx_pos >= 0 and layer_idx_pos < len(layer_parts):
                    layer_index = layer_parts[layer_idx_pos]
                    cur_sensitivity = sensitivity[name].sum().item()
                    layer_sensitivity[layer_index] += cur_sensitivity
        
        # Normalize sensitivity
        layer_sensitivity_value = list(layer_sensitivity.values())
        x_normalized = F.normalize(torch.tensor(layer_sensitivity_value), p=2, dim=0)
        
        # Sort layers by sensitivity
        sorted_layers = sorted(layer_sensitivity.items(), key=lambda item: item[1], reverse=True)
        
        # Select top-k layers, then sort by layer index
        sorted_layers = sorted_layers[:k]
        sorted_layers = sorted(sorted_layers, key=lambda item: int(item[0]))
        
        top_k_layers = [int(layer) for layer, _ in sorted_layers]
        print(f'Top {k} sensitive layers (sorted by layer index): {top_k_layers}')
        
        return x_normalized
    
    def analyze_dataset(self, data_path: str, 
                       prompt_format: str = "instruction_output", 
                       num_samples: int = 100, 
                       k: int = 5) -> np.ndarray:
        """
        Analyze layer sensitivity on a dataset.
        
        Args:
            data_path: Path to the dataset file
            prompt_format: Format of the prompts in the dataset
            num_samples: Number of samples to analyze
            k: Number of top layers to select
            
        Returns:
            Normalized layer sensitivity scores
        """
        # Load data
        data = self._load_data(data_path)
        if num_samples > 0:
            data = data[:num_samples]
        
        print(f'Starting sensitivity analysis on {len(data)} samples using {prompt_format} format')
        
        # Calculate sensitivity
        sensitivity = {}
        losses = []
        
        for sample in tqdm(data, desc="Analyzing samples"):
            try:
                loss, sample_sensitivity = self.calculate_sensitivity(
                    sample, prompt_format=prompt_format)
                losses.append(loss)
                
                # Aggregate sensitivities
                for name, cur_sensitivity in sample_sensitivity.items():
                    if name in sensitivity:
                        sensitivity[name] += cur_sensitivity
                    else:
                        sensitivity[name] = cur_sensitivity
                        
                del sample_sensitivity
                gc.collect()
                torch.cuda.empty_cache() if self.device == "cuda" else None
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Report average loss
        avg_loss = sum(losses) / len(losses) if losses else float('nan')
        print(f"Average loss: {avg_loss:.4f}")
        
        # Select top-k layers
        x_normalized = self.select_top_k_layers(sensitivity, k=k)
        
        return x_normalized.numpy()
    
    def _load_data(self, path: str) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            path: Path to the data file
            
        Returns:
            List of data samples
        """
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, dict):
                    # If the JSON root is a dict, extract the main data array
                    # Adjust this based on your specific JSON structure
                    for key in data:
                        if isinstance(data[key], list):
                            return data[key]
                    # If no list found, return the dict as a single-item list
                    return [data]
                return data
            except json.JSONDecodeError:
                # Handle JSONL format
                data = []
                f.seek(0)  # Reset file pointer
                for line in f:
                    data.append(json.loads(line))
                return data


def main():
    """Main function to run the sensitivity analysis."""
    parser = argparse.ArgumentParser(description="Analyze layer sensitivity in LLMs")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--format", type=str, default="question_cot", 
                        choices=["question_dpsk", "question_cot"],
                        help="Format of the prompts in the dataset")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to analyze")
    parser.add_argument("--k", type=int, default=5, help="Number of top layers to select")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--output", type=str, default="sensitivity_results.json",
                        help="Path to save the results")
    
    args = parser.parse_args()
    
    # Set CUDA device if applicable
    if args.device.startswith("cuda") and ":" in args.device:
        device_id = args.device.split(":")[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        args.device = "cuda"
        
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(args.model, device=args.device)
    
    # Run analysis
    sensitivity_scores = analyzer.analyze_dataset(
        args.data, prompt_format=args.format, num_samples=args.samples, k=args.k)
    
    # Save results
    results = {
        "model": args.model,
        "dataset": args.data,
        "format": args.format,
        "samples_analyzed": args.samples,
        "top_k": args.k,
        "sensitivity_scores": sensitivity_scores.tolist()
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print("Layer sensitivity scores:", sensitivity_scores)


if __name__ == "__main__":
    main()