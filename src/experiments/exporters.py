import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import pandas as pd


class ExperimentResultsExporter:
    results_export_dir = Path("experiments_results")

    def __init__(self, evaluation_improvements):
        self.evaluation_improvements = evaluation_improvements

    def print_average_evaluation_improvements(self):
        print("Improvements on test set by validation set:")
        print("    ", self.evaluation_improvements)
        pd.DataFrame(self.evaluation_improvements, columns=["val_set_improvement"])\
            .to_csv(self.results_export_dir.joinpath("out_of_sample.csv"), )
        print("Average improvements on test set:")
        print("    ", np.average(self.evaluation_improvements))


class I_FGSM_Exporter(ExperimentResultsExporter):
    def __init__(self, evaluation_improvements, improvements):
        super().__init__(evaluation_improvements)
        self.improvements = improvements

    @property
    def results_export_dir(self):
        return super(I_FGSM_Exporter, self).results_export_dir.joinpath("I_FGSM")

    def save_fig(self):
        pd.DataFrame(np.max(self.improvements, axis=1)).to_csv(self.results_export_dir.joinpath("in_sample.csv"))


class MI_FGSM_Exporter(ExperimentResultsExporter):

    def __init__(self, evaluation_improvements, improvements, mu_range):
        super().__init__(evaluation_improvements)
        self.improvements = improvements
        self.mu_range = mu_range

    @property
    def results_export_dir(self):
        return super(MI_FGSM_Exporter, self).results_export_dir.joinpath("MI_FGSM")

    def _plot(self):
        fig = plt.figure(
            dpi=300,
            figsize=(15, 5),
        )
        fig.suptitle("MI-FGSM")

        axes = fig.add_subplot(
            1, 3, 1,
            xlim=(0, 2.0),
            xlabel="mu",
            ylabel="Improvement",
        )

        max_improvements = np.max(self.improvements, axis=2)
        for val_idx, max_improvement in enumerate(max_improvements):
            axes.plot(
                self.mu_range,
                max_improvement,
                label=f"val {val_idx}"
            )
        axes.plot(
            self.mu_range,
            np.average(max_improvements, axis=0),
            linestyle="dotted",
            label="avg"
        )
        axes.legend()

        axes = fig.add_subplot(
            1, 3, (2, 3),
            xlabel="Iteration",
            ylabel="Improvement",
        )
        for mu, track in zip(self.mu_range, self.improvements[1]):
            axes.plot(
                track,
                label=f"mu {round(mu, 1)}",
                linestyle="dotted" if round(mu, 1) == 0.6 else None
            )
        axes.legend()

        bests = np.max(self.improvements, axis=(1, 2))
        print(f"bests: {bests}")
        print(f"avg bests: {np.average(bests)}")

    def save_fig(self):
        self._plot()
        if not self.results_export_dir.is_dir():
            self.results_export_dir.mkdir()
        plt.savefig(Path(self.results_export_dir, "improvement_plot"))
        pd.DataFrame(np.max(self.improvements, axis=2), columns=self.mu_range).to_csv(self.results_export_dir.joinpath("in_sample.csv"))



class AB_FGSM_Exporter(ExperimentResultsExporter):
    def __init__(self, evaluation_improvements, improvements, beta_1_range, beta_2_range, modified=False, **kwargs):
        super().__init__(evaluation_improvements, **kwargs)
        self.beta_2_range = beta_2_range
        self.beta_1_range = beta_1_range
        self.improvements = improvements
        self.modified = modified

    @property
    def results_export_dir(self):
        return super(AB_FGSM_Exporter, self).results_export_dir.joinpath(
            ("M_" if self.modified else "") + "AB_FGSM"
        )

    def _plot(self):
        fig = plt.figure(
            dpi=300,
            figsize=(10, 15)
        )
        fig.suptitle("M-AB-FGSM" if self.modified else "AB-FGSM")
        beta_1_max_improvements = np.max(self.improvements, axis=(2, 3))
        beta_2_max_improvements = np.max(self.improvements, axis=(1, 3))

        axes_beta_1 = fig.add_subplot(
            3, 2, 1,
            xlim=(0, 1),
            ylim=(0, 0.45),
            xlabel="beta_1",
            ylabel="Improvement",
        )
        axes_beta_2 = fig.add_subplot(
            3, 2, 2,
            xlim=(0, 1),
            ylim=(0, 0.45),
            ylabel="Improvement",
            xlabel="beta_2"
        )

        for val_idx, (beta_1_max_improvement, beta_2_max_improvement) \
                in enumerate(zip(beta_1_max_improvements, beta_2_max_improvements)):
            axes_beta_1.plot(
                self.beta_1_range,
                beta_1_max_improvement,
                label=f"val {val_idx}"
            )
            axes_beta_2.plot(
                self.beta_2_range,
                beta_2_max_improvement,
                label=f"val {val_idx}"
            )
        axes_beta_1.plot(
            self.beta_1_range,
            np.average(beta_1_max_improvements, axis=0),
            label=f"avg",
            linestyle="dotted",
        )
        axes_beta_2.plot(
            self.beta_2_range,
            np.average(beta_2_max_improvements, axis=0),
            label=f"avg",
            linestyle="dotted",
        )
        axes_beta_1.legend()
        axes_beta_2.legend()

        axes_track_beta_1 = fig.add_subplot(
            3, 2, (3, 4),
            ylim=(0, 0.45),
            ylabel="Improvement",
            xlabel="Iteration",
        )
        axes_track_beta_2 = fig.add_subplot(
            3, 2, (5, 6),
            ylim=(0, 0.45),
            ylabel="Improvement",
            xlabel="Iteration",
        )

        # validation set idx = 1
        optimal_beta_1_idx = np.argmax(np.max(self.improvements[1], axis=(1, 2)))
        optimal_beta_2_idx = np.argmax(np.max(self.improvements[1], axis=(0, 2)))
        beta_1_tracks = self.improvements[1, optimal_beta_1_idx]
        beta_2_tracks = self.improvements[1, :, optimal_beta_2_idx]
        for beta_2, track in zip(self.beta_2_range, beta_1_tracks):
            axes_track_beta_1.plot(
                track,
                label=f"beta_2 {round(beta_2, 3)}"
            )
        axes_track_beta_1.legend()
        for beta_1, track in zip(self.beta_1_range, beta_2_tracks):
            axes_track_beta_2.plot(
                track,
                label=f"beta_1 {round(beta_1, 3)}"
            )
        axes_track_beta_2.legend()

        print("modified" if self.modified else "original")
        bests = np.max(self.improvements, axis=(1, 2, 3))
        print(f"bests: {bests}")
        print(f"avg best: {np.average(bests)}")
        for i in range(4):
            best_no_iter = np.max(self.improvements[i], axis=(1, 2))
            print(best_no_iter.shape)
            print(best_no_iter.argmax())
            best_no_iter = np.max(self.improvements[i], axis=(0, 2))
            print(best_no_iter.shape)
            print(best_no_iter.argmax())

    def save_fig(self):
        if not self.results_export_dir.is_dir():
            self.results_export_dir.mkdir()
        self._plot()
        plt.savefig(Path(self.results_export_dir, "improvement_plot"))
        pd.DataFrame(np.max(self.improvements[:, (3 if self.modified else 2), 7], axis=1))\
            .to_csv(self.results_export_dir.joinpath("in_sample.csv"))


class apgd_Exporter(ExperimentResultsExporter):
    def __init__(self, evaluation_improvements, improvements, threshold_range, optimizer, **kwargs):
        super().__init__(evaluation_improvements, **kwargs)
        self.improvements = improvements
        self.momentum = optimizer == "MI_FGSM"
        self.threshold_range = threshold_range

    @property
    def results_export_dir(self):
        return super(apgd_Exporter, self).results_export_dir.joinpath(
            "apgd_" + ("M" if self.momentum else "") + "I_FGSM"
        )

    def _plot(self):
        pd.DataFrame(np.max(self.improvements, axis=2), columns=[0.85, 0.9, 0.95]) \
            .to_csv(self.results_export_dir.joinpath("in_sample.csv"))

    def save_fig(self):
        if not self.results_export_dir.is_dir():
            self.results_export_dir.mkdir()
        self._plot()
        # plt.savefig(Path(self.results_export_dir, "improvement_plot"))