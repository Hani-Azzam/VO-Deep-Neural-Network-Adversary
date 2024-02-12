from experiments.loaders import *
from experiments import pgd_results_loaders, apgd_ResultLoader

import subprocess
from sys import stdout
import argparse
from tqdm import tqdm


def run_pgd_MI_FGSM(run_args, val_set_idx):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, choices={"I_FGSM", "MI_FGSM", "AB_FGSM", "M_AB_FGSM"})
    parser.add_argument("--attack", choices={"pgd", "apgd"}, default="pgd")
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()
    global_run_args = [
        "python3",
        "run_attacks.py",
        "--seed", "42",
        "--model-name", "tartanvo_1914.pkl",
        "--test-dir", "VO_adv_project_train_dataset_8_frames",
        "--max_traj_len", "8",
        "--batch-size", "1",
        "--worker-num", "1",
        "--save_csv",
        "--attack", "const",
        "--attack_k", "20" if args.attack == "pgd" else str(120),
        "--preprocessed_data",
        "--test_set_idx", "4",
        "--save_report_run",
        "--save_best_pert",
        "--experiment_name", f"{args.attack}_{args.optimizer}",
    ]

    if args.slurm:
        global_run_args = ["srun", "-c", "2", "--gres=gpu:1", "-w", "lambda1"] + global_run_args

    if args.attack == "pgd":
        results_loader = pgd_results_loaders[args.optimizer]()
    else:
        assert args.optimizer in {"I_FGSM", "MI_FGSM"}, args.optimizer
        results_loader = apgd_ResultLoader(args.optimizer)

    for val_set_idx in tqdm(range(4)):
        val_args = ([
            "--val_set_idx", f"{val_set_idx}",
            "--load_attack", str(results_loader.get_evaluation_png_path(val_set_idx)),
        ])
        subprocess.run(global_run_args + val_args, cwd="./", stderr=stdout)
        # globals()[f"run_{args.attack}_{args.optimizer}"](global_run_args + val_arg, val_set_idx)