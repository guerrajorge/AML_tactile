import logging
import sys
from datetime import datetime
import os


class DispatchingFormatter:

    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)


def logger_initialization(logger_dir, parser):

    # arg object
    args = parser.parse_args()

    # logLevel = ['DEBUG', 'INFO', 'ERROR']
    # no logLevel, default to INFO
    if not args.logLevel:
        logging.getLogger().setLevel(getattr(logging, 'INFO'))
    else:
        logging.getLogger().setLevel(getattr(logging, args.logLevel))

    # not only log to a file but to stdout
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    handler_dictionary = {
        'info': logging.Formatter("%(message)s"),
        'info.line': logging.Formatter("%(message)s\n"),
        'line.info': logging.Formatter("\n%(message)s"),
        'tab.info': logging.Formatter("\t%(message)s"),
        'tab.tab.info': logging.Formatter("\t\t%(message)s"),
        'tab.info.line': logging.Formatter("\t%(message)s\n"),
        'tab.tab.info.line': logging.Formatter("\t\t%(message)s\n"),
        'line.tab.info': logging.Formatter("\n\t%(message)s"),
        'time.info': logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'time.info.line': logging.Formatter("%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.time.regular': logging.Formatter("\n%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.time.info': logging.Formatter("\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.time.info': logging.Formatter("\t\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.time.info.line': logging.Formatter("\t%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.time.info.line': logging.Formatter("\t\t%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.tab.time.info': logging.Formatter("\n\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'time.debug': logging.Formatter("%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'debug.time.line': logging.Formatter("%(asctime)s - %(funcName)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.time.debug': logging.Formatter("\n%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.time.debug': logging.Formatter("\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.time.debug': logging.Formatter("\t\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.time.debug.line': logging.Formatter("\t%(asctime)s - %(funcName)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.time.debug.line': logging.Formatter("\t\t%(asctime)s - %(funcName)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.tab.time.debug': logging.Formatter("\n\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'line.tab.tab.time.debug': logging.Formatter("\n\t\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
    }

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DispatchingFormatter(handler_dictionary, logging.Formatter('%(message)s')))
    logging.getLogger().addHandler(handler)

    # create the logging file handler
    file_name = 'logfile_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
    file_dir = os.path.join(logger_dir, file_name)
    fh = logging.FileHandler(file_dir)
    fh.setFormatter(DispatchingFormatter(handler_dictionary, logging.Formatter('%(message)s')))

    # add handler to logger object
    logging.getLogger().addHandler(fh)
