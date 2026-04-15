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

""" Event """

import time

import requests  # pylint: disable=E0401

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611


class Event:  # pylint: disable=E1101,R0903,W0201
    """
        Event Resource

        self is pointing to current Module instance

        Note: web.event decorator must be the last decorator (at top)
    """

    @web.event("pylon_modules_initialized")
    def handle_pylon_modules_initialized(self, _context, _event, payload):
        """ Handler """
        event_pylon_id = payload
        if self.context.id != event_pylon_id:
            return
        #
        ai_run_platform_url = self.descriptor.config.get("ai_run_platform_url", None)
        ai_run_platform_token = self.descriptor.config.get("ai_run_platform_token", None)
        ai_run_platform_verify = self.descriptor.config.get("ai_run_platform_verify", False)
        ai_run_platform_timeout = self.descriptor.config.get("ai_run_platform_timeout", 120)
        #
        ai_run_platform_delay = self.descriptor.config.get("ai_run_platform_delay", 2)
        #
        if ai_run_platform_url is not None and ai_run_platform_url:
            log.info("Will register AI/Run descriptor in %s seconds", ai_run_platform_delay)
            #
            time.sleep(ai_run_platform_delay)
            #
            log.info("Registering AI/Run descriptor")
            #
            descriptor = self.provider_descriptor()
            headers = None
            #
            if ai_run_platform_token is not None and ai_run_platform_token:
                headers = {
                    "Authorization": f"Bearer {ai_run_platform_token}",
                }
            #
            try:
                register_result = requests.post(
                    ai_run_platform_url,
                    headers=headers,
                    json=descriptor,
                    verify=ai_run_platform_verify,
                    timeout=ai_run_platform_timeout,
                )
                #
                register_result.raise_for_status()
                #
                log.info("Registration result: %s", register_result)
            except:  # pylint: disable=W0702
                log.exception("Failed to register")