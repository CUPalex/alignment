import sys
import logging
from pathlib import Path
import numpy as np

def run():
    from alignment.model_representations.feature_remover import FeatureRemover
    from alignment.brain_alignment.computation import AlignmentComputation
    from alignment.data.data import HarryPotterData

    SENT_LEVEL_FEATURES = ["sentence_length", "tree_depth", "top_constituents",
                            "tense", "subject_number", "object_number"]
    RANDOM = ["random_binary"]
    WORD_LEVEL_FEATURES = ["pos_tags", "smallest_constituents"] # "word_depth"
    ALL_FEATURES = SENT_LEVEL_FEATURES + RANDOM + WORD_LEVEL_FEATURES
    step_names = ["random", "step-1000", "step-143000", "step-43000", "step-0", "step-100000", "step-15000", "step-72000"]
    layers = [0, 2, 4, 6, 8, 10, 12]
    all_subjects = ["H", "F", "M"]

    save_dir = Path("/proj/inductive-bias.shadow/abakalov.trash/with_removed_features")
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_remover = FeatureRemover(words_file="/proj/inductive-bias.shadow/abakalov.data/words_fmri.npy")

    feature_dir = Path("/proj/inductive-bias.shadow/abakalov.trash/feature_removed")
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    for feature in ALL_FEATURES:
        print(f"Starting feature {feature}")
        for step_name in step_names:
            print(f"Starting step {step_name}")
            for layer in layers:
                print(f"Starting layer {layer}")
                removed_feat = feature_remover.remove_feature(feature,
                                                              representations_folder=f"/proj/inductive-bias.shadow/abakalov.trash/model_representations/{step_name}",
                                                              layer=layer, save_dir=str(feature_dir))
                
                data = HarryPotterData(data_directory="/proj/inductive-bias.shadow/abakalov.data", features_folder=str(feature_dir))

                for subject in all_subjects:
                    print(f"Starting subject {subject}")

                    alignment_computer = AlignmentComputation()
                    corrs, all_preds, all_tests = alignment_computer.run(data, subject, n_folds=5, num_delays=4, layers=[0])

                    cur_save_path = save_dir.joinpath(feature).joinpath(step_name).joinpath(subject).joinpath(str(layer))
                    cur_save_path.mkdir(parents=True, exist_ok=True)
                    np.save(str(cur_save_path.joinpath("corrs.npy")), corrs[0])
                    np.save(str(cur_save_path.joinpath("preds.npy")), all_preds[0])
                    np.save(str(cur_save_path.joinpath("tests.npy")), all_tests[0])
                feature_dir.joinpath("representations.npy").unlink()
                feature_dir.joinpath("words_skipped.json").unlink()

if __name__ == "__main__":
    sys.path.append("/proj/inductive-bias/llama")
    logging.basicConfig(level=logging.INFO)
    print("Starting program")
    run()
