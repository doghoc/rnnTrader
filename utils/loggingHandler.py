import logging


class LoggingHandler:

    def __init__(self):
        logger = logging.getLogger('rnnTrader')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('rnnTrader.log', mode='w')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("Init loggingHandler")
