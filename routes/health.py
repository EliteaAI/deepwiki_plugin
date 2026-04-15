#!/usr/bin/python3
# coding=utf-8

""" Health Check Route """
import datetime
import os
import time
from pylon.core.tools import web


class Route:
    """ Health check route """

    @web.route("/health", methods=["GET"])
    def health_route(self):
        """ Return plugin health status """
        try:
            current_time = time.time()
            uptime = current_time - getattr(self, 'start_time', current_time)

            config = self.runtime_config()

            return {
                "status": "UP",
                "providerVersion": "1.0.0",
                "uptime": int(uptime),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "plugin": "DeepWikiPlugin",
                "configuration": config,
                "extra_info": {
                    "hostname": os.environ.get("HOSTNAME", os.environ.get("POD_NAME", "unknown")),
                    "pod_ip": os.environ.get("POD_IP", "unknown"),
                },
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "plugin": "wiki_builder", 
                "error": str(e)
            }, 500
