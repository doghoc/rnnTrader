import sys
from utils.loggingHandler import LoggingHandler
from utils.loggingHandler import LoggingHandler


def main():
    # K.tensorflow_backend._get_available_gpus()
    LoggingHandler()
    logger = logging.getLogger("rnnTrader.main")
    logger.setLevel(logging.INFO)
    logger.info("Start rnnTrader")
    training = Training()
    training.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
