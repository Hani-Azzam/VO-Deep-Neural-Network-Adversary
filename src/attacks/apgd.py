import time
from typing import Collection

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pgd import PGD
from Datasets.tartanTrajFlowDataset import extract_traj_data


class APGD(PGD):
    def __init__(self, *args, sparsity_ratio_threshold, checkpoints: Collection[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoints = checkpoints
        self.sparsity_ratio_threshold = sparsity_ratio_threshold

    def perturb(self, data_loader, y_list, eps,
                targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing PGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))
        print(f"attack optimizer: {self.optimizer}")

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        with torch.no_grad():
            total_loss_track = torch.empty(self.n_iter, self.n_restarts)

        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.perf_counter()

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            pert = self.project(pert, eps)
            step_size = a_abs
            sparsity = 0.2
            min_step_size = 0.05
            best_momentum = 0

            eval_loss_tot_list = []

            # for k in K
            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.perf_counter()

                # optimization step:
                pert = self._optimization_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                               multiplier, step_size, eps, device)

                with torch.no_grad():
                    if k in self.checkpoints:
                        previous_sparsity = sparsity
                        sparsity = torch.count_nonzero(best_pert - pert) / (1.5 * pert.numel())
                        if sparsity / previous_sparsity >= self.sparsity_ratio_threshold:
                            step_size = max(step_size/1.5, min_step_size)
                        else:
                            step_size = a_abs
                        if a_abs == step_size:
                            pert = best_pert
                            if self.optimizer == "MI_FGSM":
                                self.grad_momentum = best_momentum


                step_runtime = time.perf_counter() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.perf_counter()

                # evaluate patch:
                with torch.no_grad():
                    # Loss: eval_loss_tot
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)
                    eval_loss_tot_list.append(eval_loss_tot)
                    # if Loss > Loss_best
                    if eval_loss_tot > best_loss_sum:
                        best_pert = pert.clone().detach()
                        if self.optimizer == "MI_FGSM":
                            best_momentum = self.grad_momentum
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.perf_counter() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.perf_counter() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
            with torch.no_grad():
                total_loss_track[:, rest] = torch.tensor(eval_loss_tot_list)
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss, total_loss_track

    
    def _optimization_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                           multiplier, a_abs, eps, device=None):
        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        # for i in N_train
        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad

            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()

        with torch.no_grad():
            grad = grad_tot
            if self._pert_update_aux is None:
                self._pert_update_aux = self.pert_update_dict[self.optimizer]
            pert += self._pert_update_aux(grad, pert, multiplier, a_abs, device)
            pert = self.project(pert, eps)
        return pert
