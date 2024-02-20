import logging


from .utils import MetricManager, set_seed, check_args, TqdmToLogger, stratified_split
from .loaders import load_model, load_dataset, split


def set_logger(path):
    """
    Initialisation du logger
    """
    logger = logging.getLogger(__name__)
    logging_format = logging.Formatter(
        fmt='[%(levelname)s] (%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path)
    
    stream_handler.setFormatter(logging_format)
    file_handler.setFormatter(logging_format)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.INFO)
    
    logger.info('[BIENVENNUE] Initialisation...')
    welcome_message = """Apprentissage fédéré. FL."""
    logger.info(welcome_message)
