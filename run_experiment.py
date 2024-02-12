import subprocess
import argparse
from tqdm import tqdm

import numpy as np
from sys import stdout


def run_pgd_MI_FGSM(run_args):
    for mu in tqdm(np.arange(0, 2.2, 0.2)):
        mu_args = [
            "--mu", f"{mu}",
        ]
        subprocess.run(run_args + mu_args, cwd="./", stderr=stdout)


def run_apgd_MI_FGSM(run_args):
    mu_alpha_args = [
        "--mu", f"{0.6}",
        "--alpha", f"{1.0}"
    ]
    for threshold in tqdm([0.85, 0.9, 0.95]):
        threshold_args = [
            "--sparsity_ratio_threshold", f"{threshold}"
        ]
        subprocess.run(run_args + mu_alpha_args + threshold_args, cwd="./", stderr=stdout)


def run_apgd_I_FGSM(run_args):
    return run_apgd_MI_FGSM(run_args)


def run_pgd_M_AB_FGSM(run_args):
    for beta_1 in tqdm(list(np.arange(0, 1, 0.2)) + [0.9, 0.99, 0.999]):
        for beta_2 in tqdm(list(np.arange(0, 1, 0.2)) + [0.9, 0.99, 0.999]):
            betas_args = [
                "--beta_1", f"{beta_1}",
                "--beta_2", f"{beta_2}",
            ]
            subprocess.run(run_args + betas_args, cwd="./", stderr=stdout)
    for beta_1 in tqdm(np.arange(0, 1, 0.2)):
        for beta_2 in tqdm([0.9, 0.99, 0.999]):
            betas_args = [
                "--beta_1", f"{beta_1}",
                "--beta_2", f"{beta_2}",
            ]
            subprocess.run(run_args + betas_args, cwd="./", stderr=stdout)


def run_pgd_AB_FGSM(run_args):
    return run_pgd_M_AB_FGSM(run_args)


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
            "--attack", args.attack,
            "--attack_k", "20" if args.attack == "pgd" else str(120),
            "--preprocessed_data",
            "--test_set_idx", "4",
            "--save_report_run",
            "--save_best_pert",
            "--attack_optimizer", args.optimizer,
        ]
    if args.slurm:
        global_run_args = ["srun", "-c", "2", "--gres=gpu:1", "-w", "lambda1"] + global_run_args

    for val_set_idx in tqdm(range(4)):
        val_arg = ([
            "--val_set_idx", f"{val_set_idx}",
        ])
        globals()[f"run_{args.attack}_{args.optimizer}"](global_run_args + val_arg)
    # python3 run_attacks.py --seed 42 --model-name tartanvo_1914.pkl --test-dir "VO_adv_project_train_dataset_8_frames"  --max_traj_len 8 --batch-size 1 --worker-num 1 --save_csv --attack pgd --attack_k 20 --attack_optimizer MI_FGSM --mu 1  --preprocessed_data --val_set_idx 3 --test_set_idx 4 --save_report_run --save_best_pert
