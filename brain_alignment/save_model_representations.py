import argparse
from abc import ABC
import logging
from pathlib import Path
import json

import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer

class ModelRepresentations(ABC):
    def __init__(self, device, context_len):
        self.device = device
        self.context_len = context_len

        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m",
            revision="main",
            cache_dir="/proj/inductive-bias.shadow/abakalov.data/pythia",
        ).to(self.device)
        self.model.eval()
        self.n_layers = len(self.model.gpt_neox.layers)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            revision="main",
            cache_dir="/proj/inductive-bias.shadow/abakalov.data/pythia",
        )
        logging.info(f"Created model and tokenizer of device {self.device} with context {self.context_len}")

    @torch.inference_mode()
    def get_model_layer_representations(self, words):
        logging.info("Getting model representations")
        words_layer_representations = None

        for i, word in enumerate(words):
            if i % 100 == 0:
                logging.info(f"Getting representations for word {i}: {word}")
            text_before_word = " ".join(words[:i])
            if self.tokenizer(text_before_word, return_tensors="pt")["input_ids"].shape[-1] <= self.context_len:
                continue
            else:
                if words_layer_representations is None:
                    logging.info("Creating storage")
                    first_taken_word = i
                    words_layer_representations = np.zeros((self.n_layers + 1, len(words) - i, self.model.gpt_neox.embed_in.weight.shape[-1]))
                text = " ".join(words[:i + 1])
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs_truncated_input_ids = inputs["input_ids"][:, -self.context_len:].to(self.device)
                inputs_truncated_attention_mask = inputs["attention_mask"][:, -self.context_len:].to(self.device)
                outputs = self.model(input_ids=inputs_truncated_input_ids,
                                     attention_mask=inputs_truncated_attention_mask,
                                     output_hidden_states=True)
                for layer in range(self.n_layers + 1):
                    words_layer_representations[layer, i - first_taken_word, :] = outputs.hidden_states[layer][0, -1, :].cpu().numpy()
            
        return words_layer_representations, first_taken_word
    
    def compute_and_save_model_representations(self, words, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        words_layer_representations, first_taken_word = self.get_model_layer_representations(words)
        np.save(str(output_dir.joinpath("representations.npy")), words_layer_representations)
        with open(output_dir.joinpath("words_skipped.json"), "w", encoding="utf-8") as file:
            json.dump(first_taken_word, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_words_file", type=str, default="/proj/inductive-bias.shadow/abakalov.data/words_fmri.npy")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--context_len", type=int, default=50)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)


    words = np.load(args.path_to_words_file)
    representations_calculator = ModelRepresentations(args.device, args.context_len)
    representations_calculator.compute_and_save_model_representations(words, args.output_dir)