#!/usr/bin/python3
# coding=utf-8

""" Initialization Methods """

import time
import arbiter
import threading

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

         #
        self.state_lock = threading.Lock()
        #
        self.invocation_state = {}  # toolkit -> tool -> state
        #
        self.invocation_event_node = arbiter.make_event_node(
            config={
                "type": "MockEventNode",
            },
        )
        #
        self.invocation_task_node = arbiter.TaskNode(
            self.invocation_event_node,
            pool="invocation",
            task_limit=None,
            ident_prefix="invocation_",
            multiprocessing_context="threading",
            task_retention_period=3600,
            housekeeping_interval=60,
            thread_scan_interval=0.1,
            start_max_wait=1,
            query_wait=1,
            watcher_max_wait=1,
            stop_node_task_wait=1,
            result_max_wait=1,
            result_transport="memory",
            start_attempts=1,
        )
        #
        self.invocation_task_node.start()
        self.invocation_task_node.subscribe_to_task_statuses(self.invocation_task_change)
        #
        self.invocation_task_node.register_task(
            self.perform_invoke_request, "perform_invoke_request",
        )

    @web.deinit()
    def deinit(self):
        """ De-Init """
        self.invocation_task_node.unregister_task(
            self.perform_invoke_request, "perform_invoke_request",
        )
        #
        self.invocation_task_node.stop()