from abc import ABC
import logging
from pathlib import Path
import json

import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer

class ModelRepresentations(ABC):
    LINGUISTIC_TASKS = [
        "sentence_length", #  length of a sentence in words
        "tree_depth", # depth of the sentence’s syntactic tree
        "top_constituents", # the sequence of top-level constituents in the syntactic tree
        "tense", # whether the main verb of the sentence is marked as being in the present (PRES class)
        # or past (PAST class) tense
        "subject_number", # the number of the subject of the main clause of a sentence
        "object_number" # the object number in the main clause of a sentence
    ]
    LINGUISTIC_TASKS_TO_CLASSES_NUM = {
        "sentence_length": 3, # [<=5, 5-8, >=9]
        "tree_depth": 3, # [5, 6-7, >=8]
        "top_constituents": 2, # [1, >=2], originally 20-class classification on ADVP_NP_NP_, CC_ADVP_NP_VP_, etc
        "tense": 2, # [past, present]
        "subject_number": 2, # [singular, plural]
        "object_number": 2 # [singular, plural]
    }

    def __init__(self, device, context_len, model_step, save_dir):
        self.device = device
        self.context_len = context_len
        self.model_step = model_step
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m",
            revision=f"step{self.model_step}",
            cache_dir=f"/proj/inductive-bias.shadow/abakalov.data/pythia/{self.model_step}",
        ).to(self.device)
        self.model.eval()
        self.n_layers = len(self.model.gpt_neox.layers)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            revision=f"step{self.model_step}",
            cache_dir=f"/proj/inductive-bias.shadow/abakalov.data/pythia/{self.model_step}",
        )
        logging.info(f"Created model and tokenizer from step {self.model_step} on device {self.device} with context {self.context_len}")

    @torch.inference_mode()
    def get_model_layer_representations(self, words):
        if self.save_dir.joinpath("representations.npy").exists() and self.save_dir.joinpath("words_skipped.json").exists():
            logging.info("Returning saved representations")
            words_layer_representations = np.load(str(self.save_dir.joinpath("representations.npy")))
            with open(self.save_dir.joinpath("words_skipped.json"), "r", encoding="utf-8") as file:
                first_taken_word = int(file.read())
            return words_layer_representations, first_taken_word
        

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


        logging.info("Saving model representations to file")
        np.save(str(self.save_dir.joinpath("representations.npy")), words_layer_representations)
        with open(self.save_dir.joinpath("words_skipped.json"), "w", encoding="utf-8") as file:
            json.dump(first_taken_word, file)
        return words_layer_representations, first_taken_word