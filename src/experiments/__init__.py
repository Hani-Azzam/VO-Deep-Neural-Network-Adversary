from .loaders import *
from .exporters import *

pgd_results_loaders = {
    "I_FGSM": I_FGSM_ResultsLoader,
    "MI_FGSM": MI_FGSM_ResultsLoader,
    "AB_FGSM": AB_FGSM_ResultsLoader,
    "M_AB_FGSM": lambda: AB_FGSM_ResultsLoader(modified=True),
}
pgd_results_exporters = {
    "I_FGSM": I_FGSM_Exporter,
    "MI_FGSM": MI_FGSM_Exporter,
    "AB_FGSM": AB_FGSM_Exporter,
    "M_AB_FGSM": AB_FGSM_Exporter,
}
pgd_results_exporters_args = {
    "I_FGSM": {

    },
    "MI_FGSM": {
        "mu_range": MI_FGSM_ResultsLoader.mu_range,
    },
    "AB_FGSM": {
        "beta_1_range": AB_FGSM_ResultsLoader.beta_1_range,
        "beta_2_range": AB_FGSM_ResultsLoader.beta_2_range,
        "modified": False,
    },
    "M_AB_FGSM": {
        "beta_1_range": AB_FGSM_ResultsLoader.beta_1_range,
        "beta_2_range": AB_FGSM_ResultsLoader.beta_2_range,
        "modified": True,
    },
}