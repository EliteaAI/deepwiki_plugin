"""Tests for the Pylon class-API surface matcher (Phase A port from wikis).

The Pylon plugin convention has no path decorators on verb methods —
the route is composed at runtime from ``rel_path`` (``api/vN/foo.py``)
and the plugin name (``metadata.json``). See
``_graph_audit/PYLON_ENDPOINT_DERIVATION_NOTE.md`` for the full
context that motivated porting this matcher.

These tests pin the matcher's contract:
- non-class nodes return [] cleanly,
- ``APIBase`` / ``MethodView`` / ``Resource`` inheritance triggers the
  emit path,
- verb methods become METHOD entries (de-duplicated),
- ``url_params`` suffixes parse correctly and ``<int:id>`` style path
  params collapse to ``{var}``,
- ``plugin_name`` produces the deployed ``/api/vN/<plugin>/...`` twin
  alongside the on-disk source path.
"""

from __future__ import annotations

import pytest

from plugin_implementation.code_graph.api_surface_extractor import (
    _match_pylon_api,
    _pylon_route_from_rel_path,
)


# ─── 1. rel_path → route base ─────────────────────────────────────────

class TestRouteFromRelPath:

    def test_versioned_api(self):
        assert _pylon_route_from_rel_path(
            "api/v2/configuration.py"
        ) == "/api/v2/configuration"

    def test_unversioned_api(self):
        assert _pylon_route_from_rel_path(
            "api/foo.py"
        ) == "/api/foo"

    def test_plugin_subdir_stripped_to_first_api_segment(self):
        # The matcher strips everything before ``/api/``.
        assert _pylon_route_from_rel_path(
            "plugins/configurations/api/v2/foo.py"
        ) == "/api/v2/foo"

    def test_non_api_path_returns_empty(self):
        assert _pylon_route_from_rel_path("services/foo.py") == ""

    def test_empty_input(self):
        assert _pylon_route_from_rel_path("") == ""


# ─── 2. _match_pylon_api gating ───────────────────────────────────────

class TestPylonGating:

    def test_non_class_returns_empty(self):
        node = {
            "symbol_type": "function",
            "rel_path": "api/v2/foo.py",
            "source_text": "def get(self): ...",
        }
        assert _match_pylon_api(node) == []

    def test_class_without_apibase_returns_empty(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/foo.py",
            "source_text": (
                "class Foo:\n"
                "    def get(self): pass\n"
            ),
        }
        assert _match_pylon_api(node) == []

    def test_class_with_no_verb_methods_returns_empty(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/foo.py",
            "source_text": (
                "class Foo(APIBase):\n"
                "    def helper(self): pass\n"
            ),
        }
        assert _match_pylon_api(node) == []

    def test_class_with_no_api_path_returns_empty(self):
        node = {
            "symbol_type": "class",
            "rel_path": "services/foo.py",
            "source_text": (
                "class Foo(APIBase):\n"
                "    def get(self): pass\n"
            ),
        }
        assert _match_pylon_api(node) == []


# ─── 3. Verb dispatch ─────────────────────────────────────────────────

class TestPylonVerbDispatch:

    def test_method_view_emits_get_post(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v1/items.py",
            "source_text": (
                "class ItemsAPI(MethodView):\n"
                "    def get(self): ...\n"
                "    def post(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        rests = {s["surface"] for s in surfaces}
        assert "GET /api/v1/items" in rests
        assert "POST /api/v1/items" in rests

    def test_resource_inheritance_recognised(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v1/foo.py",
            "source_text": (
                "class FooAPI(Resource):\n"
                "    def delete(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        assert any(s["surface"] == "DELETE /api/v1/foo" for s in surfaces)

    def test_dedup_when_same_method_appears_twice(self):
        # Same method declared twice at module level (rare but possible
        # in test fixtures) must not emit two surfaces.
        node = {
            "symbol_type": "class",
            "rel_path": "api/v1/x.py",
            "source_text": (
                "class X(APIBase):\n"
                "    def get(self): ...\n"
                "    def Get(self): ...\n"  # case variant
            ),
        }
        surfaces = _match_pylon_api(node)
        gets = [s for s in surfaces if s["surface"].startswith("GET ")]
        # _PYLON_METHOD_DEF is case-insensitive; both produce the same
        # uppercase verb, so dedup keeps only one.
        assert len({s["surface"] for s in gets}) == len(gets)


# ─── 4. url_params suffix parsing ─────────────────────────────────────

class TestUrlParams:

    def test_path_params_collapse_to_var(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/configuration.py",
            "source_text": (
                "class API(APIBase):\n"
                "    url_params = ['<int:project_id>/<string:config_id>']\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        # Parametrised path params should collapse to ``{var}``.
        target = "GET /api/v2/configuration/{var}/{var}"
        assert any(s["surface"] == target for s in surfaces)

    def test_multiple_url_params_emit_each_suffix(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v1/items.py",
            "source_text": (
                "class API(APIBase):\n"
                "    url_params = [\n"
                "        '<int:project_id>',\n"
                "        '<int:project_id>/<string:item_id>',\n"
                "    ]\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        rests = {s["surface"] for s in surfaces}
        assert "GET /api/v1/items/{var}" in rests
        assert "GET /api/v1/items/{var}/{var}" in rests

    def test_no_url_params_emits_bare_route(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v1/health.py",
            "source_text": (
                "class API(APIBase):\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        assert any(s["surface"] == "GET /api/v1/health" for s in surfaces)


# ─── 5. plugin_name twin ──────────────────────────────────────────────

class TestPluginNameTwin:

    def test_plugin_mounted_twin_emitted(self):
        """``plugin_name='configurations'`` produces both the on-disk
        and the deployed-mounted route."""
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/configuration.py",
            "source_text": (
                "class API(APIBase):\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node, plugin_name="configurations")
        rests = {s["surface"] for s in surfaces}
        assert "GET /api/v2/configuration" in rests
        assert "GET /api/v2/configurations/configuration" in rests

    def test_no_twin_when_already_mounted(self):
        """If the on-disk path already starts with the plugin name, the
        twin would be a duplicate; the matcher must not emit it twice."""
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/configurations/foo.py",
            "source_text": (
                "class API(APIBase):\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node, plugin_name="configurations")
        rests = [s["surface"] for s in surfaces]
        # No duplicates.
        assert len(rests) == len(set(rests))

    def test_no_plugin_name_no_twin(self):
        node = {
            "symbol_type": "class",
            "rel_path": "api/v2/configuration.py",
            "source_text": (
                "class API(APIBase):\n"
                "    def get(self): ...\n"
            ),
        }
        surfaces = _match_pylon_api(node)
        rests = {s["surface"] for s in surfaces}
        # Without a plugin_name twin, ``/api/v2/configurations/...`` is
        # absent. The on-disk route is emitted, plus the gateway-prefix
        # stripped variant ``/configuration`` (created by
        # ``_emit_rest_surfaces`` for cross-language pairing) — that
        # one is expected and shared with all REST matchers.
        assert "GET /api/v2/configuration" in rests
        assert not any("configurations" in s for s in rests)


# ─── deployed-URL derivation for documentation ────────────────────────

from plugin_implementation.code_graph.api_surface_extractor import (  # noqa: E402
    derive_pylon_endpoints,
    plugin_name_from_metadata_text,
)


class TestDerivePylonEndpoints:
    """``derive_pylon_endpoints`` renders the human-readable deployed URL
    used to ground the wiki writer (keeps param names, mounts plugin)."""

    CONFIG_SRC = (
        "class API(APIBase):\n"
        "    url_params = ['<int:project_id>']\n"
        "    def get(self, project_id, **kwargs): ...\n"
        "    def post(self, project_id, **kwargs): ...\n"
    )

    def test_monorepo_plugins_path_mounts_and_doubles(self):
        # File named like its plugin doubles the segment at deploy time.
        eps = derive_pylon_endpoints(
            "plugins/configurations/api/v1/configurations.py",
            self.CONFIG_SRC,
            "configurations",
        )
        assert eps == [
            "GET /api/v1/configurations/configurations/{project_id}",
            "POST /api/v1/configurations/configurations/{project_id}",
        ]

    def test_keeps_param_name_not_var(self):
        eps = derive_pylon_endpoints(
            "api/v1/configurations.py", self.CONFIG_SRC, "configurations"
        )
        assert all("{project_id}" in e for e in eps)
        assert not any("{var}" in e for e in eps)

    def test_already_mounted_path_not_doubled(self):
        eps = derive_pylon_endpoints(
            "api/v1/configurations/sub.py", self.CONFIG_SRC, "configurations"
        )
        assert eps[0] == "GET /api/v1/configurations/sub/{project_id}"

    def test_no_plugin_name_bare_base(self):
        eps = derive_pylon_endpoints(
            "api/v1/configurations.py", self.CONFIG_SRC, ""
        )
        assert eps == [
            "GET /api/v1/configurations/{project_id}",
            "POST /api/v1/configurations/{project_id}",
        ]

    def test_non_pylon_source_returns_empty(self):
        assert derive_pylon_endpoints(
            "api/v1/configurations.py", "def helper():\n    return 1\n", "x"
        ) == []

    def test_no_url_params_no_suffix(self):
        src = "class API(APIBase):\n    def get(self): ...\n"
        eps = derive_pylon_endpoints(
            "plugins/configurations/api/v1/configurations.py", src, "configurations"
        )
        assert eps == ["GET /api/v1/configurations/configurations"]


class TestPluginNameFromMetadataText:

    def test_plain_json(self):
        assert plugin_name_from_metadata_text('{"name": "configurations"}') == "configurations"

    def test_tolerates_file_header(self):
        assert plugin_name_from_metadata_text(
            '[File: metadata.json]\n{"name": "configurations"}'
        ) == "configurations"

    def test_rejects_non_identifier(self):
        assert plugin_name_from_metadata_text('{"name": "../etc"}') == ""

    def test_empty(self):
        assert plugin_name_from_metadata_text("") == ""
