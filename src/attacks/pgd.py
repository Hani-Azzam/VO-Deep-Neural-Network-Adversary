import math
from typing import Dict, Callable

import numpy as np
import torch

from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2

import pandas as pd
from torch.nn import functional as F


class PGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None,
            optimizer=None,
            beta_1=None,
            beta_2=None,
            mu=None,
    ):
        super(PGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding)

        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)

        if optimizer is None:
            optimizer = "I_FGSM"
        self.optimizer = optimizer
        # I_FGSM
        def init_i_fgsm():
            pass

        # MI-I_FGSM
        def init_mi_fgsm():
            self.grad_momentum = None
            self.mu = mu

        # AI-FGSM
        def init_ai_fgsm():
            self.adam_m = None
            self.adam_v = None
            self.adam_beta_1_t = torch.tensor([1])
            self.adam_beta_2_t = torch.tensor([1])
            self.adam_beta_1 = beta_1
            self.adam_beta_2 = beta_2

        # AB-FGSM
        def init_ab_fgsm():
            self.ab_m = None
            self.ab_s = None
            # self.ab_beta_1 = torch.tensor([0.99])
            # self.ab_beta_2 = torch.tensor([0.999])
            # self.ab_beta_1_t = torch.tensor([1.0])
            # self.ab_beta_2_t = torch.tensor([1.0])
            self.ab_beta_1 = beta_1
            self.ab_beta_2 = beta_2
            self.ab_beta_1_t = 1.0
            self.ab_beta_2_t = 1.0
            self.ab_accumulative_gamma = 0.0

        optimizer_init_dict = {
            "I_FGSM": init_i_fgsm,
            "MI_FGSM": init_mi_fgsm,
            "AI_FGSM": init_ai_fgsm,
            "AB_FGSM": init_ab_fgsm,
            "M_AB_FGSM": init_ab_fgsm,
        }
        try:
            optimizer_init_dict[optimizer]()
        except KeyError:
            assert False, f"optimizer {optimizer} not in {optimizer_init_dict.keys()}"

        self._pert_update_aux = None
        self._pert_update_dict = None

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                                scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2,
                                device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                                   intrinsic_I0,
                                                                   img1_delta, img2_delta,
                                                                   scale,
                                                                   mask1, mask2,
                                                                   perspective1,
                                                                   perspective2,
                                                                   device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                               scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2,
                               device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window = self.calc_sample_grad_single(pert_window,
                                                       img1_I0_window,
                                                       img2_I0_window,
                                                       intrinsic_I0_window,
                                                       img1_delta_window,
                                                       img2_delta_window,
                                                       scale_window,
                                                       y_window,
                                                       clean_flow_window,
                                                       target_pose_window,
                                                       perspective1_window,
                                                       perspective2_window,
                                                       mask1_window,
                                                       mask2_window,
                                                       device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)

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

    @property
    def pert_update_dict(self) -> Dict[str, Callable]:
        if self._pert_update_dict is None:
            self._pert_update_dict = {
                "I_FGSM": self._i_fgsm_update,
                "MI_FGSM": self._mi_fgsm_update,
                "AI_FGSM": self._ai_fgsm_update,
                "AB_FGSM": self._ab_fgsm_update,
                "M_AB_FGSM": self._m_ab_fgsm_update,
            }
        return self._pert_update_dict

    def _i_fgsm_update(self, grad, pert, multiplier, a_abs, device):
        return multiplier * a_abs * self.normalize_grad(grad)

    def _mi_fgsm_update(self, grad, pert, multiplier, a_abs, device):
        if self.grad_momentum is None:
            self.grad_momentum = F.normalize(grad.view(grad.shape[0], -1), p=1, dim=-1).view(grad.shape)
        else:
            self.grad_momentum = self.mu * self.grad_momentum\
                                 + F.normalize(grad.view(grad.shape[0], -1), p=1, dim=-1).view(grad.shape)
        return multiplier * a_abs * self.normalize_grad(self.grad_momentum)

    def _ai_fgsm_update(self, grad, pert, multiplier, a_abs, device):
        g = self.normalize_grad(grad)
        if self.adam_m is None or self.adam_v is None:
            self.adam_m = (1-self.adam_beta_1) * g
            self.adam_v = (1-self.adam_beta_2) * g**2
        else:
            self.adam_m = self.adam_beta_1 * self.adam_m + (1-self.adam_beta_1) * g
            self.adam_v = self.adam_beta_2 * self.adam_v + (1-self.adam_beta_2) * g**2
        self.adam_beta_1_t *= self.adam_beta_1
        self.adam_beta_2_t *= self.adam_beta_2

        s = self.adam_m/(torch.sqrt(self.adam_v))
        # alpha = a_abs *
        raise NotImplementedError

    def _ab_fgsm_update(self, grad, pert, multiplier, a_abs, device):
        self.ab_beta_1_t *= self.ab_beta_1
        self.ab_beta_2_t *= self.ab_beta_2
        if self.ab_m is None:
            self.ab_m = (1-self.ab_beta_1) * grad
            self.ab_s = (1-self.ab_beta_2) * (grad - self.ab_m) ** 2
        else:
            self.ab_m = self.ab_beta_1 * self.ab_m + (1-self.ab_beta_1) * grad
            self.ab_s = torch.maximum(self.ab_beta_2 * self.ab_s + (1-self.ab_beta_2) * (grad - self.ab_m) ** 2,
                                      self.ab_s)
        m_hat = self.ab_m/(1-self.ab_beta_1_t)
        s_hat = self.ab_s/(1-self.ab_beta_2_t) + 10**-14
        self.ab_accumulative_gamma += math.sqrt(1-self.ab_beta_2_t)/(1-self.ab_beta_1_t)
        return multiplier * (a_abs / self.ab_accumulative_gamma) * torch.sign(m_hat/(1-torch.sqrt(10**-14 + s_hat)))

    def _m_ab_fgsm_update(self, grad, pert, multiplier, a_abs, device):
        grad = F.normalize(grad.view(grad.shape[0], -1), p=1, dim=-1).view(grad.shape)
        self.ab_beta_1_t *= self.ab_beta_1
        self.ab_beta_2_t *= self.ab_beta_2
        if self.ab_m is None:
            self.ab_m = (1-self.ab_beta_1) * grad
            self.ab_s = (1-self.ab_beta_2) * (grad - self.ab_m) ** 2
        else:
            self.ab_m = self.ab_beta_1 * self.ab_m + (1-self.ab_beta_1) * grad
            self.ab_s = torch.maximum(self.ab_beta_2 * self.ab_s + (1-self.ab_beta_2) * (grad - self.ab_m) ** 2,
                                      self.ab_s)
        m_hat = self.ab_m/(1-self.ab_beta_1_t)
        s_hat = self.ab_s/(1-self.ab_beta_2_t) + 10**-14
        self.ab_accumulative_gamma += math.sqrt(1-self.ab_beta_2_t)/(1-self.ab_beta_1_t)
        return multiplier * (a_abs / self.ab_accumulative_gamma) * torch.sign(m_hat/(1-torch.sqrt(10**-14 + s_hat)))


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

            eval_loss_tot_list = []

            # for k in K
            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.perf_counter()

                # optimization step:
                pert = self._optimization_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                               multiplier, a_abs, eps, device)

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
