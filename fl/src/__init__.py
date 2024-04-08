import logging

from .utils import set_seed, Range, TensorBoardRunner, check_args, init_weights, TqdmToLogger, MetricManager, stratified_split, tensorboard_runner
from .loaders import load_dataset, load_model



def set_logger(path, args):
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
    welcome_message = """FEDERATED LEARNING."""
    logger.info(welcome_message)
