#!/usr/bin/python3
# coding=utf-8

#   Copyright 2025 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Method """

import time

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611

from arbiter.tasknode.tools import InterruptTaskThread  # pylint: disable=E0611,E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def invocation_task_change(self, event, data):
        """ Task state changed """
        _ = event
        #
        task_id = data.get("task_id", None)
        status = data.get("status", "unknown")  # one of: pending, running, stopped, pruned
        #
        if task_id is None or status == "unknown":
            return
        #
        if status == "pruned":
            # As there are no more meta now - iter all for cleanup
            with self.state_lock:
                for _toolkit_name, toolkit_state in self.invocation_state.items():
                    for _tool_name, tool_state in toolkit_state.items():
                        tool_state.pop(task_id, None)
            #
            return
        #
        task_meta = self.invocation_task_node.get_task_meta(task_id)
        #
        toolkit_name = task_meta.get("toolkit_name", "Toolkit")
        tool_name = task_meta.get("tool_name", "tool")
        #
        with self.state_lock:
            if toolkit_name not in self.invocation_state:
                self.invocation_state[toolkit_name] = {}
            #
            if tool_name not in self.invocation_state[toolkit_name]:
                self.invocation_state[toolkit_name][tool_name] = {}
            #
            if task_id not in self.invocation_state[toolkit_name][tool_name]:
                self.invocation_state[toolkit_name][tool_name][task_id] = {
                    "task_id": task_id,
                    "added_ts": time.time(),
                }
            #
            self.invocation_state[toolkit_name][tool_name][task_id]["status"] = status
            #
            if status == "stopped":
                try:
                    result = self.invocation_task_node.get_task_result(task_id)
                except BaseException as exception:  # pylint: disable=W0718
                    log.exception("Failed to invoke %s:%s", toolkit_name, tool_name)
                    exception_info = str(exception)
                    #
                    result = {
                        "errorCode": "500",
                        "message": "Internal Server Error",
                        "details": [exception_info],
                    }, 500
                #
                self.invocation_state[toolkit_name][tool_name][task_id]["result"] = result

    @web.method()
    def invocation_thinking(self, message):
        """ Store custom event """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            #
            task_id = tasknode_task.id
            task_meta = tasknode_task.meta
        except:  # pylint: disable=W0702
            return
        #
        toolkit_name = task_meta.get("toolkit_name", "Toolkit")
        tool_name = task_meta.get("tool_name", "tool")
        #
        with self.state_lock:
            if toolkit_name not in self.invocation_state:
                return
            #
            if tool_name not in self.invocation_state[toolkit_name]:
                return
            #
            if task_id not in self.invocation_state[toolkit_name][tool_name]:
                return
            #
            if "custom_events" not in self.invocation_state[toolkit_name][tool_name][task_id]:
                self.invocation_state[toolkit_name][tool_name][task_id]["custom_events"] = []
            #
            self.invocation_state[toolkit_name][tool_name][task_id]["custom_events"].append({
                "data": {
                    "message": message,
                },
            })

    @web.method()
    def invocation_stop_checkpoint(self):
        """ Check for stop request """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            #
            task_id = tasknode_task.id
            task_meta = tasknode_task.meta
        except:  # pylint: disable=W0702
            return
        #
        toolkit_name = task_meta.get("toolkit_name", "Toolkit")
        tool_name = task_meta.get("tool_name", "tool")
        #
        with self.state_lock:
            if toolkit_name not in self.invocation_state:
                return
            #
            if tool_name not in self.invocation_state[toolkit_name]:
                return
            #
            if task_id not in self.invocation_state[toolkit_name][tool_name]:
                return
            #
            invocation_state = self.invocation_state[toolkit_name][tool_name][task_id]
            #
            if "stop_requested" not in invocation_state:
                return
            #
            if invocation_state["stop_requested"]:
                if "processes" in invocation_state:
                    for proc in invocation_state["processes"]:
                        if proc.poll() is None:
                            proc.terminate()
                            # Wait briefly for termination without reading streams
                            # to avoid reentrant calls if streams are already being read
                            try:
                                proc.wait(timeout=3)
                            except:  # pylint: disable=W0702
                                proc.kill()
                                try:
                                    proc.wait(timeout=1)
                                except:  # pylint: disable=W0702
                                    pass
                #
                raise InterruptTaskThread()

    @web.method()
    def invocation_process_add(self, proc):
        """ Process: add """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            #
            task_id = tasknode_task.id
            task_meta = tasknode_task.meta
        except:  # pylint: disable=W0702
            return
        #
        toolkit_name = task_meta.get("toolkit_name", "Toolkit")
        tool_name = task_meta.get("tool_name", "tool")
        #
        with self.state_lock:
            if toolkit_name not in self.invocation_state:
                return
            #
            if tool_name not in self.invocation_state[toolkit_name]:
                return
            #
            if task_id not in self.invocation_state[toolkit_name][tool_name]:
                return
            #
            invocation_state = self.invocation_state[toolkit_name][tool_name][task_id]
            #
            if "processes" not in invocation_state:
                invocation_state["processes"] = []
            #
            invocation_state["processes"].append(proc)

    @web.method()
    def invocation_process_remove(self, proc):
        """ Process: remove """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            #
            task_id = tasknode_task.id
            task_meta = tasknode_task.meta
        except:  # pylint: disable=W0702
            return
        #
        toolkit_name = task_meta.get("toolkit_name", "Toolkit")
        tool_name = task_meta.get("tool_name", "tool")
        #
        with self.state_lock:
            if toolkit_name not in self.invocation_state:
                return
            #
            if tool_name not in self.invocation_state[toolkit_name]:
                return
            #
            if task_id not in self.invocation_state[toolkit_name][tool_name]:
                return
            #
            invocation_state = self.invocation_state[toolkit_name][tool_name][task_id]
            #
            if "processes" not in invocation_state:
                return
            #
            if proc in invocation_state["processes"]:
                invocation_state["processes"].remove(proc)
