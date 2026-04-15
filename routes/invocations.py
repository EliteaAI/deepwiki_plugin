#!/usr/bin/python3
# coding=utf-8

""" Invocation Status Route """

import flask  # pylint: disable=E0401

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611


class Route:
    """ Invocation status route """
    @web.route("/tools/<toolkit_name>/<tool_name>/invocations/<invocation_id>", methods=["GET", "DELETE"])  # pylint: disable=C0301
    def invocations_route(self, toolkit_name, tool_name, invocation_id):  # pylint: disable=R0911
        """ Handler """
        if flask.request.method == "GET":
            with self.state_lock:
                if toolkit_name not in self.invocation_state:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                if tool_name not in self.invocation_state[toolkit_name]:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                if invocation_id not in self.invocation_state[toolkit_name][tool_name]:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                invocation_state = self.invocation_state[toolkit_name][tool_name][invocation_id]
                invocation_status = invocation_state["status"]
                #
                custom_events = {}
                #
                if "custom_events" in invocation_state and invocation_state["custom_events"]:
                    custom_events["custom_events"] = invocation_state["custom_events"].copy()
                    invocation_state["custom_events"].clear()
                #
                if invocation_status == "pending":
                    return {
                        "invocation_id": invocation_id,
                        "status": "Started",
                        **custom_events,
                    }
                #
                if invocation_status == "running":
                    return {
                        "invocation_id": invocation_id,
                        "status": "InProgress",
                        **custom_events,
                    }
                #
                if "result" in invocation_state:
                    return invocation_state["result"]
        #
        elif flask.request.method == "DELETE":
            with self.state_lock:
                if toolkit_name not in self.invocation_state:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                if tool_name not in self.invocation_state[toolkit_name]:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                if invocation_id not in self.invocation_state[toolkit_name][tool_name]:
                    return {
                        "errorCode": "404",
                        "message": "Resource Not Found",
                        "details": [],
                    }, 404
                #
                invocation_state = self.invocation_state[toolkit_name][tool_name][invocation_id]
                invocation_state["stop_requested"] = True
                #
                if "processes" in invocation_state:
                    for proc in invocation_state["processes"]:
                        if proc.poll() is None:
                            proc.terminate()
            #
            return flask.Response(status=204)
        #
        return {
            "errorCode": "500",
            "message": "Internal Server Error",
            "details": [],
        }, 500
