#!/usr/bin/python3
# coding=utf-8

""" Tool Invocation Route """
import json
import uuid
import os
import flask
import traceback
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pylon.core.tools import log, web

logger = logging.getLogger(__name__)

default_bucket = 'wiki_artifacts'


class Route:
    """ Invocation route """

    @web.route("/tools/<toolkit_name>/<tool_name>/invoke", methods=["POST"])
    def invoke_route(self, toolkit_name, tool_name):  # pylint: disable=R
        """ Handler """
        #
        # Validate
        #
        try:
            request_data = flask.request.json
        except:  # pylint: disable=W0702
            return {
                "errorCode": "400",
                "message": "Bad Request",
                "details": [],
            }, 400
        #
        # Invoke
        #
        invocation_id = self.invocation_task_node.start_task(
            "perform_invoke_request",
            kwargs={
                "toolkit_name": toolkit_name,
                "tool_name": tool_name,
                "request_data": request_data,
            },
            pool="invocation",
            meta={
                "toolkit_name": toolkit_name,
                "tool_name": tool_name,
            },
        )
        #
        if invocation_id is None:
            return {
                "errorCode": "500",
                "message": "Internal Server Error",
                "details": [],
            }, 500
        #
        # Plugin is async by default (and unconditionally): return the invocation id
        # immediately so other users can start generations without waiting.
        return {
            "invocation_id": invocation_id,
            "status": "Started",
        }