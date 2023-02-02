__version__ = "0.1.13"

def enable_logging(log_file=None):
    """Turn on internal logging for MPoL

    Parameters
    ----------
    log_file : string, optional
        Output filename to which logging messages are written.
        If not provided, logs will only be printed to the screen
    """
    import logging

    if log_file is not None:
        handlers = [ logging.FileHandler(log_file, mode='w'),
                     logging.StreamHandler()
                     ]
    else:
        handlers = [ logging.StreamHandler() ]

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=handlers
                        )
