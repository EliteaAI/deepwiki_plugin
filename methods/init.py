#!/usr/bin/python3
# coding=utf-8

""" Initialization Methods """

import time
from pylon.core.tools import log, web


class Method:
    """ Initialization methods """

    @web.init()
    def init_config(self):
        """ Initialize plugin configuration """
        config = self.runtime_config()
        log.info(f"Wiki Builder configured with base_path: {config['base_path']}")
        
        # Store start time for health checks
        self.start_time = time.time()
        
        # Setup dependencies
        self.setup_dependencies()
