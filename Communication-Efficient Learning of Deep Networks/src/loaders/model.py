import inspect
import logging
from importlib import import_module

logger = logging.getLogger(__name__)


def load_model(args):
    """
    Chargement des différents modèles
    """
    model_class = import_module('..models', package=__package__).__dict__[
        args.model_name]
    required_args = inspect.getfullargspec(model_class)[0]

    model_args = {}
    for argument in required_args:
        if argument == 'self':
            continue
        model_args[argument] = getattr(args, argument)

    logger.info(f"[MODELE] Le modèle chargé est ... {model_class}")

    model = model_class(**model_args)

    return model
