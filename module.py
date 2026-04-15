#!/usr/bin/python3
# coding=utf-8

""" Wiki Builder Plugin Module """

import importlib
import os
import sys

from pylon.core.tools import log, module


class Module(module.ModuleModel):
    """ Plugin Module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor

    def init(self):
        """ Initialize the plugin """
        log.info("Initializing Wiki Builder Plugin")
        self._ensure_requirements_on_path()
        self._check_critical_dependencies()
        self.descriptor.init_all(
            url_prefix="/",
            static_url_prefix="/",
        )
        log.info("Wiki Builder Plugin initialized successfully")

    def _ensure_requirements_on_path(self):
        """Ensure the plugin's site-packages directory is on sys.path.

        Pylon's bulk activation should add the requirements path automatically,
        but in some deployments (SaaS, certain Pylon versions) it can fail
        silently.  This method detects the gap and fixes it at runtime.
        """
        # 1) Try the path that Pylon set on the descriptor during activation.
        req_path = getattr(self.descriptor, "requirements_path", None)
        if req_path and req_path in sys.path:
            log.info("Plugin requirements path already active: %s", req_path)
            return

        # 2) Compute the canonical path ourselves as a fallback.
        fallback_path = os.path.join(
            "/data", "requirements", "deepwiki_plugin",
            "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )

        target = req_path or fallback_path

        if not os.path.isdir(target):
            log.warning(
                "Plugin requirements directory does not exist: %s — "
                "dependencies will likely be missing",
                target,
            )
            return

        if target not in sys.path:
            sys.path.insert(0, target)
            importlib.invalidate_caches()
            log.warning(
                "Plugin requirements path was NOT on sys.path. "
                "Injected manually: %s",
                target,
            )
        else:
            log.info("Plugin requirements path already active: %s", target)

    def _check_critical_dependencies(self):
        """Check and log status of critical dependencies at startup."""
        deps = {
            "kubernetes": "kubernetes",
            "langgraph": "langgraph.graph",
            "langchain_core": "langchain_core",
        }
        all_ok = True
        for label, module_path in deps.items():
            try:
                __import__(module_path)
                log.info("Dependency check: %s OK", label)
            except ImportError as exc:
                log.warning("Dependency check: %s MISSING (%s)", label, exc)
                all_ok = False

        if not all_ok:
            log.warning(
                "Some dependencies are missing. sys.path = %s",
                sys.path,
            )

    def deinit(self):
        """ Cleanup when plugin is disabled """
        log.info("Deinitializing Wiki Builder Plugin")
