import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict
from abc import abstractmethod
from functools import reduce


def float_to_path_str(value: float):
    return str(value).replace(".", "_")


class ResultsLoader:
    total_clean_loss = np.array([
        [
            17.86737,
            17.867386,
            17.867367,
            17.867376,
            17.867382,
            17.86739,
            17.867384,
            17.867378,
            17.867357,
            17.867353,
            17.86737,
            17.867369,
            17.867374,
            17.867365,
            17.867373,
            17.867376,
            17.867382,
            17.86737,
            17.867373,
            17.86737,
        ],
        [
            35.247936,
            35.247913,
            35.247955,
            35.247932,
            35.247925,
            35.247944,
            35.24789,
            35.2479,
            35.247917,
            35.247883,
            35.247932,
            35.247917,
            35.247894,
            35.247906,
            35.247883,
            35.247902,
            35.24795,
            35.247894,
            35.24787,
            35.247948,
        ],
        [
            14.63778,
            14.637785,
            14.637785,
            14.637795,
            14.637775,
            14.637798,
            14.63778,
            14.63778,
            14.637784,
            14.637782,
            14.637796,
            14.637779,
            14.637777,
            14.637782,
            14.637778,
            14.637795,
            14.637793,
            14.637781,
            14.63778,
            14.637783,
        ],
        [
            17.423504,
            17.423487,
            17.423504,
            17.423521,
            17.42351,
            17.42352,
            17.423515,
            17.4235,
            17.423504,
            17.423515,
            17.423515,
            17.423492,
            17.423529,
            17.42351,
            17.423521,
            17.423517,
            17.423502,
            17.423498,
            17.423513,
            17.423523,
        ],
    ])
    total_clean_loss_avg = np.average(total_clean_loss, axis=1)
    _improvements = None
    _apgd = False

    @classmethod
    def _get_run_root_path(cls, optimizer, optimizer_params: OrderedDict, attack_iter, validation_idx=None, test_idx=4,
                           seed=None, experiment_name=None):
        path = Path(
            f"results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames"
            f"/train_attack/"
            f"universal_attack/"
            f"{optimizer}/"
            f"{f'{experiment_name}/' if experiment_name is not None else '' }"
            f"attack_{'apgd' if cls._apgd else 'pgd'}_norm_Linf/"
            f"opt_whole_trajectory/"
            f"opt_t_crit_none_factor_1_0_rot_crit_none_flow_crit_none_target_t_crit_none"
            f"/eval_rms"
        )
        path = path.joinpath(Path(reduce(
            lambda a, b: a + b,
            [f"_{param_name}_{float_to_path_str(param_value)}"
             for param_name, param_value in optimizer_params.items()],
            (f'seed_{seed}_' if seed is not None else '') + f"eps_1_attack_iter_{attack_iter}"
        )))

        if validation_idx is not None:
            path = path.joinpath(f"test_set_idx_{test_idx}_val_set_idx_{validation_idx}")
        return path

    @classmethod
    def get_loss_track(cls, optimizer, optimizer_params: OrderedDict, attack_iter, validation_idx=None):
        return pd.read_csv(
            cls._get_run_root_path(optimizer, optimizer_params, attack_iter, validation_idx, seed=42)
            .joinpath("results_total_loss_track.csv")
        )["0"].to_numpy()

    _results = None

    @property
    def results(self):
        if self._results is None:
            self._results = self._get_results()
        return self._results

    @property
    def improvements(self):
        if self._improvements is None:
            self._improvements = self._calc_improvements(self.results)
        return self._improvements

    @abstractmethod
    def _get_results(self):
        pass

    @abstractmethod
    def _calc_improvements(self, results):
        pass

    experiment_settings = {}

    def get_evaluation_path(self, validation_idx, test_idx):
        path = Path(
            f"results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames"
            f"/train_attack/"
            f"universal_attack/"
            f"attack_const_img_adv_best_pert/"
            f"{'apgd' if self._apgd else 'pgd'}_{self.experiment_settings['optimizer']}/"
            f"test_set_idx_{test_idx}_val_set_idx_{validation_idx}/"
            f"results_rms.csv"
        )
        return path

    def __get_evaluation_improvements(self):
        clean_loss = np.empty((4, 10))
        adv_loss = np.empty((4, 10))
        for val_idx in range(4):
            results = pd.read_csv(self.get_evaluation_path(val_idx, 4))
            clean_loss[val_idx] = results[results["frame_idx"] == 7]["clean_rms"].to_numpy()
            adv_loss[val_idx] = results[results["frame_idx"] == 7]["adv_rms"].to_numpy()

        return np.average(adv_loss, axis=1)/np.average(clean_loss, axis=1) - 1

    __evaluation_improvements = None

    @property
    def evaluation_improvements(self):
        """ average adv l_VO / average clean l_VO"""
        if self.__evaluation_improvements is None:
            self.__evaluation_improvements = self.__get_evaluation_improvements()
        return self.__evaluation_improvements


class I_FGSM_ResultsLoader(ResultsLoader):
    experiment_settings = {
        "optimizer": "I_FGSM",
        "attack_iter": 20,
    }

    def get_evaluation_png_path(self, validation_idx):
        return self._get_run_root_path(
            optimizer_params=self.params_dict(alpha=0.05, mu=0.0),
            validation_idx=validation_idx,
            seed=42,
            **{"optimizer": "MI_FGSM", "attack_iter": self.experiment_settings["attack_iter"]}
        ).joinpath("adv_best_pert/adv_best_pert.png")

    @staticmethod
    def params_dict(alpha, mu):
        params = OrderedDict()
        params["alpha"] = alpha
        params["mu"] = mu
        return params

    @classmethod
    def _get_results(cls):
        loss_tracks = np.empty((4, 20))
        for val_set_idx in range(4):
            loss_track = cls.get_loss_track(
                optimizer_params=cls.params_dict(mu=0.0, alpha=0.05),
                validation_idx=val_set_idx,
                **{"optimizer": "MI_FGSM", "attack_iter": cls.experiment_settings["attack_iter"]}
            )
            loss_tracks[val_set_idx] = loss_track
        return loss_tracks

    @classmethod
    def _calc_improvements(cls, results):
        return (results.transpose(1, 0) / cls.total_clean_loss_avg - 1).transpose(1, 0)


class MI_FGSM_ResultsLoader(ResultsLoader):
    experiment_settings = {
        "optimizer": "MI_FGSM",
        "attack_iter": 20,
    }
    mu_range = np.arange(0, 2.2, 0.2)

    def get_evaluation_png_path(self, validation_idx):
        return self._get_run_root_path(
            optimizer_params=self.params_dict(alpha=0.05, mu=0.2*3),
            validation_idx=validation_idx,
            seed=42,
            **self.experiment_settings
        ).joinpath("adv_best_pert/adv_best_pert.png")

    @staticmethod
    def params_dict(alpha, mu):
        params = OrderedDict()
        params["alpha"] = alpha
        params["mu"] = mu
        return params

    @classmethod
    def _get_results(cls):
        loss_tracks = np.empty((4, 11, 20))
        for mu_num, mu in enumerate(cls.mu_range):
            for val_set_idx in range(4):
                loss_track = cls.get_loss_track(
                    optimizer_params=cls.params_dict(mu=mu, alpha=0.05),
                    validation_idx=val_set_idx,
                    **cls.experiment_settings
                )
                loss_tracks[val_set_idx, mu_num, :] = loss_track
        return loss_tracks

    @classmethod
    def _calc_improvements(cls, results):
        return (results.transpose(2, 1, 0) / cls.total_clean_loss_avg - 1).transpose(2, 1, 0)


class AB_FGSM_ResultsLoader(ResultsLoader):
    beta_1_range = list(np.arange(0, 1, 0.2)) + [0.9, 0.99, 0.999]
    beta_2_range = list(np.arange(0, 1, 0.2)) + [0.9, 0.99, 0.999]

    def get_evaluation_png_path(self, validation_idx):
        return self._get_run_root_path(
            optimizer_params=self.params_dict(alpha=0.05, beta_1=0.2*3 if self.modified else 0.4, beta_2=0.999),
            validation_idx=validation_idx,
            seed=42,
            **self.experiment_settings
        ).joinpath("adv_best_pert/adv_best_pert.png")

    def __init__(self, modified=False):
        self.modified = modified

    @classmethod
    def params_dict(cls, alpha, beta_1, beta_2):
        params = OrderedDict()
        params["alpha"] = alpha
        params["beta_1"] = beta_1
        params["beta_2"] = beta_2
        return params

    @property
    def experiment_settings(self):
        return {
            "optimizer": "M_AB_FGSM" if self.modified else "AB_FGSM",
            "attack_iter": 20,
        }

    def _get_results(self):
        loss_tracks = np.empty((4, len(self.beta_1_range), len(self.beta_2_range),
                                self.experiment_settings["attack_iter"]))
        for beta_1_num, beta_1 in enumerate(self.beta_1_range):
            for beta_2_num, beta_2 in enumerate(self.beta_2_range):
                for val_set_idx in range(4):
                    loss_track = self.get_loss_track(
                        optimizer_params=self.params_dict(beta_1=beta_1, beta_2=beta_2, alpha=0.05),
                        validation_idx=val_set_idx,
                        **self.experiment_settings
                    )
                    loss_tracks[val_set_idx, beta_1_num, beta_2_num, :] = loss_track
        return loss_tracks

    @classmethod
    def _calc_improvements(cls, results):
        return (results.transpose(3, 1, 2, 0) / cls.total_clean_loss_avg - 1).transpose(3, 1, 2, 0)


class apgd_ResultLoader(ResultsLoader):
    _apgd = True
    sparsity_ratio_threshold = [0.85, 0.9, 0.95]

    def __init__(self, optimizer):
        self.momentum = optimizer == "MI_FGSM"

    def get_evaluation_png_path(self, validation_idx):
        return self._get_run_root_path(
            optimizer_params=self.params_dict(alpha=1.0, mu=0.6, sparsity_ratio_threshold=0.85),
            validation_idx=validation_idx,
            seed=42,
            **self.experiment_settings
        ).joinpath("adv_best_pert/adv_best_pert.png")

    @property
    def experiment_settings(self):
        return {
            "optimizer": "I_FGSM" if not self.momentum else "MI_FGSM",
            "attack_iter": 120,
        }

    def params_dict(self, alpha, mu, sparsity_ratio_threshold):
        params = OrderedDict()
        params["alpha"] = alpha
        if self.momentum:
            params["mu"] = mu
        params["sparsity_ratio_threshold"] = sparsity_ratio_threshold
        return params

    def _get_results(self):
        loss_tracks = np.empty((4, len(self.sparsity_ratio_threshold), self.experiment_settings["attack_iter"]))
        for threshold_num, threshold in enumerate(self.sparsity_ratio_threshold):
            for val_idx in range(4):
                loss_track = self.get_loss_track(
                    optimizer_params=self.params_dict(alpha=1.0, mu=0.6, sparsity_ratio_threshold=threshold),
                    validation_idx=val_idx,
                    **self.experiment_settings
                )
                loss_tracks[val_idx, threshold_num] = loss_track
        return loss_tracks

    @classmethod
    def _calc_improvements(self, results):
        return (results.transpose(2, 1, 0) / self.total_clean_loss_avg - 1).transpose(2, 1, 0)
