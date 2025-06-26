import logging 
import os
# Setup logger
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # Create the log directory if it does not exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger