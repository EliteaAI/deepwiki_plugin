#!/usr/bin/python3
# coding=utf-8

""" Wiki Builder Plugin Module """

from pylon.core.tools import log, module


class Module(module.ModuleModel):
    """ Plugin Module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor

    def init(self):
        """ Initialize the plugin """
        log.info("Initializing Wiki Builder Plugin")
        self.descriptor.init_all(
            url_prefix="/",
            static_url_prefix="/",
        )
        log.info("Wiki Builder Plugin initialized successfully")

    def deinit(self):
        """ Cleanup when plugin is disabled """
        log.info("Deinitializing Wiki Builder Plugin")
