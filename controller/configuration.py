import configparser
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

static_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
static_config.read('manager_configuration.ini')

log_file_dir = static_config['LOGGING']['logging.directory']
log_file_name = static_config['LOGGING']['logging.file.name']
maxBytes = int(static_config['LOGGING']['logging.maxBytes'])
backupCount = int(static_config['LOGGING']['logging.backupCount'])


if not os.path.exists(log_file_dir):
    os.mkdir(log_file_dir)

logging.basicConfig(level=logging.INFO,
                    handlers=[RotatingFileHandler(log_file_name, maxBytes=maxBytes, backupCount=backupCount),
                              logging.StreamHandler(sys.stdout)],
                    format="%(asctime)s %(levelname)s %(message)s")
