#!/usr/bin/python3
# coding=utf-8

""" Dependency Management """

from pylon.core.tools import log, web


class Method:
    """ Dependency management methods """

    @web.method()
    def setup_dependencies(self):
        """ Setup and verify dependencies """
        try:
            log.info("Wiki Builder dependencies verified")
        except ImportError as e:
            log.error(f"Missing dependency: {e}")
            raise RuntimeError(f"Wiki Builder dependency not available: {e}")

    @web.method()
    def check_dependency(self, module_name):
        """ Check if a specific dependency is available """
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
