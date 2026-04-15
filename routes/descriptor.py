#!/usr/bin/python3
# coding=utf-8

""" Plugin Descriptor Route """

from pylon.core.tools import web

class Route:
    """ Descriptor route """

    @web.route("/descriptor")
    def descriptor_route(self):
        """ Return plugin descriptor """
        return self.provider_descriptor()
