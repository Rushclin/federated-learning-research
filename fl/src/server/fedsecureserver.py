
import logging


from .fedavgserver import FedavgServer
logger = logging.getLogger(__name__)


class FedsecureServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedsecureServer, self).__init__(**kwargs)
