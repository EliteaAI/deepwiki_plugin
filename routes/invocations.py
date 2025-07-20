#!/usr/bin/python3
# coding=utf-8

""" Invocation Status Route """

import flask
from pylon.core.tools import web


class Route:
    """ Invocation status route """

    @web.route("/tools/<toolkit_name>/<tool_name>/invocations/<invocation_id>", methods=["GET", "DELETE"])
    def invocations_route(self, toolkit_name, tool_name, invocation_id):
        """ Handle invocation status requests """
        
        if flask.request.method == "GET":
            # In this simple example, we don't store invocation state
            # For async operations, you would track status here
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "message": "Synchronous operation completed immediately"
            }
        
        elif flask.request.method == "DELETE":
            # Handle cancellation (if supported)
            return {
                "message": "Synchronous operations cannot be cancelled"
            }, 400
