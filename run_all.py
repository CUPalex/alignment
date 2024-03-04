import sys
import logging
from pathlib import Path
import numpy as np

def run():
    from alignment.brain_alignment.computation import AlignmentComputation
    from alignment.data.data import HarryPotterData

    step_names = ["step-72000"] # "random", "step-1000", "step-143000", "step-43000", "step-0", "step-100000", "step-15000"
    layers = [0, 2, 4, 6, 8, 10, 12]
    all_subjects = ["H"] # "F", "M"
    save_path = Path("/proj/inductive-bias.shadow/abakalov.trash/corrs/plain_model")
    save_path.mkdir(parents=True, exist_ok=True)
    for step_name in step_names:
        print(f"Starting step {step_name}")
        data = HarryPotterData(data_directory="/proj/inductive-bias.shadow/abakalov.data",
                            features_folder=f"/proj/inductive-bias.shadow/abakalov.trash/model_representations/{step_name}")
    
        for subject in all_subjects:
            print(f"Starting subject {subject}")
            alignment_computer = AlignmentComputation()
            corrs, all_preds, all_tests = alignment_computer.run(data, subject, n_folds=5, num_delays=4, layers=layers)
            for layer in layers:
                cur_save_path = save_path.joinpath(str(step_name)).joinpath(str(subject)).joinpath(str(layer))
                cur_save_path.mkdir(parents=True, exist_ok=True)
                np.save(str(cur_save_path.joinpath("corrs.npy")), corrs[layer])
                np.save(str(cur_save_path.joinpath("preds.npy")), all_preds[layer])
                np.save(str(cur_save_path.joinpath("tests.npy")), all_tests[layer])


if __name__ == "__main__":
    sys.path.append("/proj/inductive-bias/llama")
    logging.basicConfig(level=logging.INFO)
    print("Starting program")
    run()
