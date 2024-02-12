import argparse
from experiments import *


if __name__ == '__main__':
    # options - results to plot
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers", type=str, nargs="*", choices={"MI_FGSM", "AB_FGSM", "M_AB_FGSM", "I_FGSM"})
    parser.add_argument("--attack", type=str, choices={"pgd", "apgd"}, default="pgd")
    args = parser.parse_args()

    if not args.optimizers:
        optimizers = ["MI_FGSM", "AB_FGSM", "M_AB_FGSM"] if args.attack == "pgd" else ["MI_FGSM"]
    else:
        optimizers = args.optimizers
        assert args.attack =="pgd" or {"AB_FGSM", "M_AB_FGSM"}.intersection(set(optimizers)) == set(), optimizers
    if not ExperimentResultsExporter.results_export_dir.is_dir():
        ExperimentResultsExporter.results_export_dir.mkdir()

    for optimizer in optimizers:
        # load data
        results_loader =\
            pgd_results_loaders[optimizer]() if args.attack == "pgd" else apgd_ResultLoader(optimizer)
        # save figures
        results_exporter = \
            pgd_results_exporters[optimizer](results_loader.evaluation_improvements, results_loader.improvements, **pgd_results_exporters_args[optimizer]) \
            if args.attack == "pgd" else apgd_Exporter(results_loader.evaluation_improvements, results_loader.improvements, [.85, .9, .95], optimizer)
        results_exporter.save_fig()
        results_exporter.print_average_evaluation_improvements()
