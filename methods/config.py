#!/usr/bin/python3
# coding=utf-8

""" Configuration Management """

import os
import pathlib
from pylon.core.tools import log, web


class Method:
    """ Configuration methods """

    @web.method()
    def runtime_config(self):
        """ Get runtime configuration """
        config = {}
        
        # Base configuration with defaults
        defaults = {
            "base_path": "/tmp/wiki_builder",
            "service_location_url": "http://127.0.0.1:8080"
        }
        
        # Merge with user config
        for key, default in defaults.items():
            config[key] = self.descriptor.config.get(key, default)
        
        # Ensure paths are absolute
        if "base_path" in config:
            config["base_path"] = os.path.abspath(config["base_path"])
        
        return config

    @web.method()
    def setup_directories(self):
        """ Create necessary directories """
        config = self.runtime_config()
        
        if "base_path" in config:
            pathlib.Path(config["base_path"]).mkdir(parents=True, exist_ok=True)
            log.info(f"Created directory: {config['base_path']}")
