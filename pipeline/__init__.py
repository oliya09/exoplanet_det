# export key functions
from .catalog import get_star_params
from .classifier import classify_target_full, CNNClassifier
from .planet import get_planet_data  # Remove generate_planet_params

from .classifier import classify_target_full, CNNClassifier

# export from lightcurve package
from .lightcurve import get_lightcurve_and_bls, plot_lc_and_bls, get_lightcurve

__all__ = [
    "get_star_params", "classify_target_full", "CNNClassifier",
    "get_planet_data",  # remove generate_planet_params
    "get_lightcurve_and_bls", "plot_lc_and_bls", "get_lightcurve"
]