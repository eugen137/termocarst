import configparser
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('./configuration.ini')

log_file_dir = config['LOGGING']['logging.directory']
log_file_name = config['LOGGING']['logging.file.name']
maxBytes = int(config['LOGGING']['logging.maxBytes'])
backupCount = int(config['LOGGING']['logging.backupCount'])

if not os.path.exists(log_file_dir):
    os.mkdir(log_file_dir)

logging.basicConfig(level=logging.INFO,
                    handlers=[RotatingFileHandler(log_file_name, maxBytes=maxBytes, backupCount=backupCount),
                              logging.StreamHandler(sys.stdout)],
                    format="%(asctime)s %(levelname)s %(message)s")
