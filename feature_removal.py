import sys
import logging
from pathlib import Path
import numpy as np

def run():
    from alignment.model_representations.feature_remover import FeatureRemover

    SENT_LEVEL_FEATURES = ["sentence_length", "tree_depth", "top_constituents",
                            "tense", "subject_number", "object_number"]
    RANDOM = ["random_binary"]
    WORD_LEVEL_FEATURES = ["pos_tags", "smallest_constituents"] # "word_depth"
    ALL_FEATURES = SENT_LEVEL_FEATURES + RANDOM + WORD_LEVEL_FEATURES

    save_dir = Path("/proj/inductive-bias.shadow/abakalov.trash/removed_features")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for feature in ALL_FEATURES:
        for step_name in ["random", "step-1000", "step-143000", "step-43000", "step-0", "step-100000", "step-15000", "step-72000"]:
            for layer in [0, 2, 4, 6, 8, 10, 12]:
                feature_remover = FeatureRemover(words_file="/proj/inductive-bias.shadow/abakalov.data/words_fmri.npy",
                                                representations_folder=f"/proj/inductive-bias.shadow/abakalov.trash/model_representations/{step_name}",
                                                layer=layer)

                cur_save_dir = save_dir.joinpath(feature).joinpath(step_name).joinpath(str(layer))
                cur_save_dir.mkdir(parents=True, exist_ok=True)
                removed_feat = feature_remover.remove_feature(feature, save_dir=str(cur_save_dir))

if __name__ == "__main__":
    sys.path.append("/proj/inductive-bias/llama")
    logging.basicConfig(level=logging.INFO)
    print("Starting program")
    run()
