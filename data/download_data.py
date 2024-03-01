import sys
import gdown
import numpy as np
import argparse
from pathlib import Path
import logging

from alignment import logger

def download_data_from_drive(save_folder):
    urls = dict(
        data_subject_F = "https://drive.google.com/file/d/1olS7TuykmJOlnIvCvn4PxXUm-k9KGCYi/view?usp=share_link",
        data_subject_H = "https://drive.google.com/file/d/12cjnpyyEVUIZX4y_LZL5Hanwfzy3ppzB/view?usp=share_link",
        data_subject_I = "https://drive.google.com/file/d/1QNQQ0xnbYEMTqnMf3_-wF0dbbfuuhNu1/view?usp=share_link",
        data_subject_J = "https://drive.google.com/file/d/1GQmzSuaBCseWI3m-tb1CRaiU63jD3Bx-/view?usp=share_link",
        data_subject_K = "https://drive.google.com/file/d/1o54oxuTM-p9Nmcw2UQ5IPtnKENNROp8s/view?usp=share_link",
        data_subject_L = "https://drive.google.com/file/d/1C6UEa3hlm-3U82zJ8jZnGmDm1_tgp28o/view?usp=share_link",
        data_subject_M = "https://drive.google.com/file/d/1xeB42pbVGO9X8kSBVuVrJYpGIoSuNe8v/view?usp=share_link",
        data_subject_N = "https://drive.google.com/file/d/1TuJLRXPMLhN4am9dleU1x8xYzkOOSTpi/view?usp=share_link",
        runs_fmri = "https://drive.google.com/file/d/1lMGJ6V2oy60axMW7wmh_a77EQOxX28YC/view?usp=share_link",
        time_fmri = "https://drive.google.com/file/d/1BHaMgbs78EQJ4MFWEelQAbwaXcDXLSGd/view?usp=share_link",
        time_words_fmri = "https://drive.google.com/file/d/1cGjm4vVuMF5BduJitvY05KGfueYYo4Fu/view?usp=share_link",
        words_fmri = "https://drive.google.com/file/d/1EBF2nFcQiSLt6Vk_Yg8qrXQSOmcHogRa/view?usp=share_link"
    )

    for file_name, url in urls.items():
        save_path = str(Path(save_folder).joinpath(f"{file_name}.npy"))
        id = url[url.find("d/") + len("d/"):url.find("/view?")]
        logger.info(f"downloading {id} ...")
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, save_path, quiet=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Brain datasets argparser")
    parser.add_argument("--data_path", help="Directory where to save files", type=str)
    parser.add_argument("--verbosity", help="Verbosity of the logger", type=str, default="DEBUG")
    
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.verbosity))
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)


    data_path = Path(args.data_path)

    assert not data_path.exists(), "--data_path already exists, provide non-existent directory or do not try to download data"
    data_path.mkdir()
    download_data_from_drive(args.data_path)

