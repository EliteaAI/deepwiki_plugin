"""API surface extractor (Phase 6 / Action 6.2).

Extracts canonical API-surface keys from parser output so that the
cross-language linker can pair endpoints across languages
(e.g. a Python ``@app.post("/api/users")`` handler with the TypeScript
client that calls ``fetch("/api/users", {method: "POST"})``).

This module is **side-effect free** — it inspects node attributes and
returns a list of :class:`APISurface` dicts. Persistence and graph
mutation happen in the linker / pipeline integration layer.

The matchers are intentionally lightweight regex/heuristic matchers
keyed by ``language``. The dispatcher returns an empty list when no
matcher applies, so callers can run it on every node without guarding.

Supported surfaces
------------------
* REST (Python: Flask, FastAPI; TS/JS: Express, NestJS; Java: JAX-RS,
  Spring; Go: chi/gin) — canonical key ``"<METHOD> <path>"``.
* gRPC — canonical key ``"grpc:<service>/<method>"`` from the proto
  service / rpc definition or generated stubs.
* GraphQL — canonical key ``"gql:<operation>:<field>"``.
* FFI — canonical key ``"ffi:<symbol>"`` (matches ``extern "C"``,
  ``ctypes``, JNI, P/Invoke, wasm-bindgen).
* BDD — canonical key ``"bdd:<step text>"``.
* CLI — canonical key ``"cli:<command path>"`` for argparse / click /
  cobra subcommand registration.

A node may yield multiple surfaces (e.g. a FastAPI handler decorated
with both ``@router.get("/")`` and ``@router.head("/")``).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict


class APISurface(TypedDict):
    kind: str
    surface: str
    weight_hint: float
    metadata: dict


# ──────────────────────────────────────────────────────────────────────
# REST matchers
# ──────────────────────────────────────────────────────────────────────

# Common HTTP method tokens used across decorators.
_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

# Python: @app.get("/x"), @router.post("/x"), @blueprint.route("/x", methods=["GET"])
_PY_REST_DECORATOR = re.compile(
    r"@\s*(?:\w+\.)?(?P<method>get|post|put|patch|delete|head|options|route)"
    r"\s*\(\s*(?P<args>[^)]+)\)",
    re.IGNORECASE,
)
_PY_ROUTE_METHODS = re.compile(r"methods\s*=\s*\[([^\]]+)\]", re.IGNORECASE)
_QUOTED_PATH = re.compile(r"""['"]([^'"]+)['"]""")

# TypeScript/JavaScript: @Get("/x"), @Post("/x"), app.get("/x", ...)
_TS_NEST_DECORATOR = re.compile(
    r"@\s*(?P<method>Get|Post|Put|Patch|Delete|Head|Options)"
    r"\s*\(\s*(?P<args>[^)]*)\)",
)
_TS_EXPRESS_CALL = re.compile(
    r"\b(?:app|router)\s*\.\s*(?P<method>get|post|put|patch|delete|head|options)"
    r"\s*\(\s*(?P<args>[^,]+),",
    re.IGNORECASE,
)
# OpenAPI-generated / generic clients: ``axios.get("/x")``, ``client.post("/x", ...)``,
# ``api.put("/x")``, ``http.delete("/x")``, ``request.patch("/x")``.
_TS_HTTP_CLIENT_CALL = re.compile(
    r"\b(?:axios|client|api|http|request)\s*\.\s*"
    r"(?P<method>get|post|put|patch|delete|head|options)"
    r"\s*\(\s*['\"`](?P<path>[^'\"`]+)['\"`]",
    re.IGNORECASE,
)
# Bare ``fetch("/x", {method: "POST"})`` calls. Default verb is GET when no
# ``method`` option is supplied (matches the WHATWG fetch spec).
_TS_FETCH_CALL = re.compile(
    r"\bfetch\s*\(\s*['\"`](?P<path>/[^'\"`]+)['\"`](?P<rest>[^)]*)\)",
)
_TS_FETCH_METHOD_OPT = re.compile(
    r"method\s*:\s*['\"`](?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)['\"`]",
    re.IGNORECASE,
)
# OpenAPI-generated request-options style (``@hey-api/openapi-ts``,
# ``openapi-typescript-codegen``, ``openapi-fetch``, custom ``__request``
# wrappers): the call passes a single options object containing both
# ``method`` and ``url`` fields. Match the small window between the
# opening ``{`` and the next ``}``/closing ``)``.
_TS_REQUEST_OPTIONS = re.compile(
    r"\{[^{}]{0,400}?method\s*:\s*['\"`](?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)['\"`][^{}]{0,400}?url\s*:\s*['\"`](?P<path>/[^'\"`]+)['\"`]",
    re.IGNORECASE | re.DOTALL,
)
_TS_REQUEST_OPTIONS_REVERSED = re.compile(
    r"\{[^{}]{0,400}?url\s*:\s*['\"`](?P<path>/[^'\"`]+)['\"`][^{}]{0,400}?method\s*:\s*['\"`](?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)['\"`]",
    re.IGNORECASE | re.DOTALL,
)
# Python: detect router prefix declarations so we can prepend them to
# every route surface in the same file. FastAPI/Starlette routers and
# Flask blueprints share the same general shape (``prefix="/items"``
# or ``url_prefix="/items"`` keyword).
_PY_ROUTER_PREFIX = re.compile(
    r"\b(?:APIRouter|Blueprint|Router)\s*\([^)]*?(?:url_)?prefix\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE | re.DOTALL,
)
# Python: ``<var> = ctypes.CDLL("libfoo")`` — captures the binding
# name so we can later detect ``<var>.foo(...)`` calls inside function
# bodies and emit ``ffi:foo`` surfaces. Without this, a Python wrapper
# like ``def compute_hash(d): return lib.compute_hash(d)`` never pairs
# with the matching native ``extern "C" fn compute_hash`` because the
# wrapper's symbol slice doesn't contain the ``ctypes.CDLL`` call.
_PY_CTYPES_LIB = re.compile(
    r"\b(?P<var>\w+)\s*=\s*ctypes\.(?:CDLL|WinDLL|cdll|windll)\b",
)
# Common API gateway prefixes worth stripping when emitting suffix
# alternates (``/api/v1/items`` ↔ ``/items``). Matches a leading
# ``/api`` or ``/rest`` or ``/graphql`` optionally followed by a
# version segment (``/v1``, ``/v2beta``, ...).
_API_PREFIX_RE = re.compile(
    r"^/(?:api|rest|graphql)(?:/v\d+(?:beta\d*|alpha\d*|rc\d*)?)?(?=/|$)",
    re.IGNORECASE,
)

# Java: @GET / @POST + @Path("/x"); Spring: @GetMapping("/x")
_JAVA_PATH = re.compile(r"""@\s*Path\s*\(\s*['"]([^'"]+)['"]""")
_JAVA_METHOD = re.compile(r"@\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")
_JAVA_SPRING = re.compile(
    r"@\s*(?P<method>Get|Post|Put|Patch|Delete|Head|Options)Mapping"
    r"\s*\(\s*['\"]?(?P<path>[^'\")\s,]*)",
)
_JAVA_REQUEST = re.compile(
    r"@\s*RequestMapping\s*\(\s*['\"]?(?P<path>[^'\")\s,]*)",
)

# Go: chi/gin r.GET("/x", ...)
_GO_REST = re.compile(
    r"\b\w+\s*\.\s*(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)"
    r"\s*\(\s*['\"](?P<path>[^'\"]+)",
)


def _normalize_path(path: str) -> str:
    p = path.strip().strip("\"'")
    if not p:
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    # Drop trailing slash except for the root path.
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def _surface_rest(method: str, path: str, weight_hint: float = 0.7) -> APISurface:
    return APISurface(
        kind="rest",
        surface=f"{method.upper()} {_normalize_path(path)}",
        weight_hint=weight_hint,
        metadata={"method": method.upper(), "path": _normalize_path(path)},
    )


def _strip_common_api_prefix(path: str) -> str:
    """Return *path* with a leading ``/api[/vX]`` stripped, or ``""`` if no
    such prefix exists. Used to emit a suffix-alternate surface so that
    a TS client calling ``POST /api/v1/items`` can pair against a Python
    route declared as ``POST /items`` (router prefix), even though the
    full app-level prefix (``/api/v1``) is only known dynamically.
    """
    p = _normalize_path(path)
    new = _API_PREFIX_RE.sub("", p, count=1)
    if not new or new == p:
        return ""
    if not new.startswith("/"):
        new = "/" + new
    return _normalize_path(new)


def _emit_rest_surfaces(
    method: str, path: str, weight_hint: float = 0.7, *, router_prefix: str = ""
) -> List[APISurface]:
    """Yield 1-2 surfaces for a single decorator/call site.

    Always emits the canonical ``METHOD /path`` (with the optional
    same-file ``router_prefix`` prepended). When the resulting path
    starts with a common API gateway prefix (``/api`` or ``/api/v1``),
    additionally emits a ``prefix_stripped`` alternate so that
    cross-language pairing works against routes declared without the
    gateway prefix.
    """
    full_path = path
    if router_prefix:
        prefix = _normalize_path(router_prefix)
        rest = _normalize_path(path)
        # Avoid double-prefixing if the route already starts with it.
        if not rest.startswith(prefix + "/") and rest != prefix:
            full_path = (prefix.rstrip("/") + ("" if rest == "/" else rest)) or "/"
    surfaces: List[APISurface] = [_surface_rest(method, full_path, weight_hint)]
    stripped = _strip_common_api_prefix(full_path)
    if stripped and stripped != _normalize_path(full_path):
        surfaces.append(APISurface(
            kind="rest",
            surface=f"{method.upper()} {stripped}",
            weight_hint=max(0.4, weight_hint - 0.1),
            metadata={
                "method": method.upper(),
                "path": stripped,
                "prefix_stripped": True,
            },
        ))
    return surfaces


def _match_rest_python(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    # Same-file router prefix (FastAPI ``APIRouter(prefix="/x")``,
    # Flask ``Blueprint(... url_prefix="/x")``). When extract is called
    # on a per-symbol slice the prefix declaration is usually NOT in
    # ``text`` — the orchestrator (`extract_api_surfaces_for_graph`)
    # is responsible for surfacing it via the ``router_prefix`` hint.
    prefix_match = _PY_ROUTER_PREFIX.search(text)
    router_prefix = prefix_match.group(1) if prefix_match else ""
    for m in _PY_REST_DECORATOR.finditer(text):
        method = m.group("method").lower()
        args = m.group("args")
        path_match = _QUOTED_PATH.search(args)
        if not path_match:
            continue
        path = path_match.group(1)
        if method == "route":
            methods_match = _PY_ROUTE_METHODS.search(args)
            methods = []
            if methods_match:
                methods = [
                    s.strip().strip("\"'").upper()
                    for s in methods_match.group(1).split(",")
                    if s.strip()
                ]
            if not methods:
                methods = ["GET"]
            for met in methods:
                if met in _HTTP_METHODS:
                    out.extend(_emit_rest_surfaces(met, path, router_prefix=router_prefix))
        else:
            out.extend(_emit_rest_surfaces(method, path, router_prefix=router_prefix))
    return out


def _match_rest_typescript(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _TS_NEST_DECORATOR.finditer(text):
        method = m.group("method")
        args = m.group("args") or ""
        path_match = _QUOTED_PATH.search(args)
        path = path_match.group(1) if path_match else "/"
        out.extend(_emit_rest_surfaces(method, path))
    for m in _TS_EXPRESS_CALL.finditer(text):
        method = m.group("method")
        args = m.group("args") or ""
        path_match = _QUOTED_PATH.search(args)
        if path_match:
            out.extend(_emit_rest_surfaces(method, path_match.group(1)))
    # OpenAPI-generated / generic HTTP clients (axios.get, client.post, ...)
    for m in _TS_HTTP_CLIENT_CALL.finditer(text):
        out.extend(_emit_rest_surfaces(m.group("method"), m.group("path")))
    # Bare fetch("/path", {method: "POST"}) — default GET when no method opt.
    for m in _TS_FETCH_CALL.finditer(text):
        path = m.group("path")
        rest = m.group("rest") or ""
        method_match = _TS_FETCH_METHOD_OPT.search(rest)
        method = method_match.group("method") if method_match else "GET"
        out.extend(_emit_rest_surfaces(method, path))
    # OpenAPI-generated request-options style:
    #   __request(OpenAPI, { method: 'POST', url: '/api/v1/items/' })
    # Both field orders observed in the wild (hey-api emits method-first,
    # openapi-typescript-codegen sometimes emits url-first).
    for m in _TS_REQUEST_OPTIONS.finditer(text):
        out.extend(_emit_rest_surfaces(m.group("method"), m.group("path")))
    for m in _TS_REQUEST_OPTIONS_REVERSED.finditer(text):
        out.extend(_emit_rest_surfaces(m.group("method"), m.group("path")))
    return out


def _match_rest_java(text: str) -> List[APISurface]:
    out: List[APISurface] = []

    # JAX-RS: @Path + @GET/POST/...
    paths = [m.group(1) for m in _JAVA_PATH.finditer(text)]
    methods = [m.group(1) for m in _JAVA_METHOD.finditer(text)]
    if paths and methods:
        for p in paths:
            for met in methods:
                out.extend(_emit_rest_surfaces(met, p))

    # Spring: @GetMapping("/x"), @PostMapping("/x"), @RequestMapping("/x")
    for m in _JAVA_SPRING.finditer(text):
        out.extend(_emit_rest_surfaces(m.group("method"), m.group("path") or "/"))
    for m in _JAVA_REQUEST.finditer(text):
        # @RequestMapping defaults to GET when no method= specified.
        out.extend(_emit_rest_surfaces("GET", m.group("path") or "/"))

    return out


def _match_rest_go(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _GO_REST.finditer(text):
        out.extend(_emit_rest_surfaces(m.group("method"), m.group("path")))
    return out


# ──────────────────────────────────────────────────────────────────────
# gRPC
# ──────────────────────────────────────────────────────────────────────

# proto: ``rpc Foo(BarRequest) returns (BarResponse);`` inside ``service Svc { ... }``
_PROTO_SERVICE = re.compile(r"\bservice\s+(\w+)\s*{", re.MULTILINE)
_PROTO_RPC = re.compile(r"\brpc\s+(\w+)\s*\(", re.MULTILINE)


def _match_grpc(text: str, language: str) -> List[APISurface]:
    """Detect gRPC services across .proto definitions and Python/Java stubs."""
    out: List[APISurface] = []

    if language == "proto" or "service " in text and "rpc " in text:
        services = _PROTO_SERVICE.findall(text)
        rpcs = _PROTO_RPC.findall(text)
        for svc in services or [""]:
            for rpc in rpcs:
                out.append(APISurface(
                    kind="grpc",
                    surface=f"grpc:{svc}/{rpc}" if svc else f"grpc:{rpc}",
                    weight_hint=0.8,
                    metadata={"service": svc, "method": rpc},
                ))

    # Python servicer convention: ``class <Svc>Servicer:`` with each
    # ``def <RpcName>(self, request, context)`` mapping to one gRPC
    # method. Pairs against ``.proto`` ``service <Svc> { rpc <RpcName>
    # ... }`` declarations.
    if language == "python" and "Servicer" in text:
        for sm in re.finditer(r"\bclass\s+(?P<svc>\w+)Servicer\b", text):
            svc = sm.group("svc")
            for dm in re.finditer(
                r"\bdef\s+(?P<rpc>[A-Z]\w*)\s*\(\s*self\b", text
            ):
                rpc = dm.group("rpc")
                out.append(APISurface(
                    kind="grpc",
                    surface=f"grpc:{svc}/{rpc}",
                    weight_hint=0.7,
                    metadata={"service": svc, "method": rpc},
                ))

    return out


# ──────────────────────────────────────────────────────────────────────
# GraphQL
# ──────────────────────────────────────────────────────────────────────

_GQL_FIELD = re.compile(
    r"\b(?P<op>type|extend\s+type)\s+(?P<root>Query|Mutation|Subscription)\s*{",
    re.IGNORECASE,
)
_GQL_RESOLVER_DEC = re.compile(
    r"@\s*(?P<op>Query|Mutation|Subscription|Resolver|FieldResolver)\b",
)


def _match_graphql(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    if "type Query" in text or "type Mutation" in text or "type Subscription" in text:
        # Coarse SDL detection — emit a single root surface so the linker
        # can still pair files. Field-level surfaces would need a real
        # GraphQL parser.
        for m in _GQL_FIELD.finditer(text):
            out.append(APISurface(
                kind="graphql",
                surface=f"gql:{m.group('root').lower()}",
                weight_hint=0.6,
                metadata={"root": m.group("root").lower()},
            ))
    for m in _GQL_RESOLVER_DEC.finditer(text):
        out.append(APISurface(
            kind="graphql",
            surface=f"gql:{m.group('op').lower()}",
            weight_hint=0.5,
            metadata={"resolver": m.group("op").lower()},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# FFI
# ──────────────────────────────────────────────────────────────────────

_FFI_EXTERN_C = re.compile(r"""extern\s+["']C["']""")
_FFI_CTYPES = re.compile(r"\bctypes\.(?:CDLL|WinDLL|cdll|windll)\b")
_FFI_JNI = re.compile(r"\bnative\s+\w+\s+\w+\s*\(")
_FFI_PINVOKE = re.compile(r"\[\s*DllImport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_FFI_WASM = re.compile(r"#\s*\[\s*wasm_bindgen\s*\]")


def _match_ffi(text: str, symbol_name: str) -> List[APISurface]:
    out: List[APISurface] = []
    triggers = (
        bool(_FFI_EXTERN_C.search(text))
        or bool(_FFI_CTYPES.search(text))
        or bool(_FFI_JNI.search(text))
        or bool(_FFI_WASM.search(text))
    )
    if triggers and symbol_name:
        out.append(APISurface(
            kind="ffi",
            surface=f"ffi:{symbol_name}",
            weight_hint=0.6,
            metadata={"symbol": symbol_name},
        ))
    for m in _FFI_PINVOKE.finditer(text):
        out.append(APISurface(
            kind="ffi",
            surface=f"ffi:{m.group(1)}",
            weight_hint=0.7,
            metadata={"library": m.group(1)},
        ))
        # Also emit a symbol-keyed surface so the C# extern method (or
        # any DllImport-decorated function) pairs with native externs
        # that share the same function name (e.g. Rust ``extern "C" fn
        # compute_hash`` ↔ C# ``[DllImport("libnative")] extern
        # compute_hash``).
        if symbol_name and symbol_name != m.group(1):
            out.append(APISurface(
                kind="ffi",
                surface=f"ffi:{symbol_name}",
                weight_hint=0.6,
                metadata={"symbol": symbol_name, "library": m.group(1)},
            ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Object / data shape (cross-language DTO pairing)
# ──────────────────────────────────────────────────────────────────────
#
# Surface key: ``obj:<typename_lower>#<sorted_lower_field_csv>``
#
# Pairs DTOs that travel across the wire (Python dataclass /
# pydantic.BaseModel / TypedDict ↔ TypeScript ``interface`` /
# ``type`` ↔ Go ``struct`` ↔ Java ``class``/``record`` ↔ Rust
# ``struct`` ↔ C# ``class``/``record``). Field names are case-folded
# and sorted so casing/order differences across languages don't break
# the join. The L1 specificity factor (``1/log(1+N_matches)``) keeps
# trivial shapes (``User#id,name``) at low weight while rare shapes
# (``BillingRefundRequest#amount,currency,reason,reference``) get
# strong cross-language edges.
#
# Tag-aware: Go ``json:"<x>"`` and serde ``rename = "<x>"`` override
# the source-code field name when present.

# Python: class header.
_OBJ_PY_CLASS = re.compile(r"^\s*class\s+(?P<name>\w+)\s*[:\(]", re.MULTILINE)
# Python: ``field_name: TypeAnnotation`` (PEP 526 style — dataclass /
# pydantic / TypedDict / attrs all share this surface form).
_OBJ_PY_FIELD = re.compile(
    r"^[ \t]+(?P<name>[A-Za-z_]\w*)\s*:\s*[^=#\n]+(?:=\s*[^#\n]+)?\s*(?:#.*)?$",
    re.MULTILINE,
)
_OBJ_PY_DUNDER = re.compile(r"^__\w+__$")

# TypeScript / JavaScript: interface or type alias.
_OBJ_TS_INTERFACE = re.compile(
    r"\b(?:export\s+)?interface\s+(?P<name>\w+)(?:\s+extends\s+[^\{]+)?\s*\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)
_OBJ_TS_TYPE = re.compile(
    r"\b(?:export\s+)?type\s+(?P<name>\w+)\s*=\s*\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)
# Field detected anywhere after a separator (line start, ``;`` or ``,``).
_OBJ_TS_FIELD = re.compile(
    r"(?:^|[;,\n])\s*(?:readonly\s+)?(?P<name>[A-Za-z_]\w*)\s*\??\s*:",
)

# Go struct.
_OBJ_GO_STRUCT = re.compile(
    r"\btype\s+(?P<name>\w+)\s+struct\s*\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)
_OBJ_GO_FIELD = re.compile(
    r"^\s*(?P<name>[A-Z]\w*)\s+[^`\n]+(?:`(?P<tag>[^`]+)`)?",
    re.MULTILINE,
)
_OBJ_GO_JSON_TAG = re.compile(r"json:\"([^,\"]+)")

# Java class / record. Body extraction allows nested braces so method
# bodies inside the class don't truncate the match.
_OBJ_JAVA_CLASS_HEADER = re.compile(
    r"\b(?:public\s+|private\s+|protected\s+|static\s+|final\s+|abstract\s+)*"
    r"class\s+(?P<name>\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*\{",
)
_OBJ_JAVA_RECORD = re.compile(
    r"\b(?:public\s+)?record\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
)
# Field: visibility + type + name + (= | ;) — methods are excluded
# because their declarations end in ``)`` or ``{``, not ``;`` or ``=``.
_OBJ_JAVA_FIELD = re.compile(
    r"\b(?:public|private|protected)\s+(?:static\s+|final\s+)*"
    r"[\w<>\[\],\s\.]+?\s+(?P<name>[a-zA-Z_]\w*)\s*[=;]",
    re.MULTILINE,
)

# Rust struct (with optional serde rename tag on field).
_OBJ_RUST_STRUCT = re.compile(
    r"\b(?:pub\s+)?struct\s+(?P<name>\w+)\s*\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)
_OBJ_RUST_FIELD = re.compile(
    r"(?:#\[serde\([^)]*?rename\s*=\s*\"(?P<rename>[^\"]+)\"[^)]*\)\]\s*)?"
    r"(?:pub\s+)?(?P<name>[a-zA-Z_]\w*)\s*:\s*[^,\n]+,?",
)

# C# class / record. Body uses nested-brace-aware matching: scan whole
# text after the class header rather than relying on a brace-balanced
# regex (auto-properties contain ``{ get; set; }`` which break that).
_OBJ_CSHARP_CLASS_HEADER = re.compile(
    r"\b(?:public\s+|internal\s+|private\s+|sealed\s+|abstract\s+)*"
    r"class\s+(?P<name>\w+)(?:\s*:\s*[^{]+)?\s*\{",
)
_OBJ_CSHARP_RECORD = re.compile(
    r"\b(?:public\s+)?record\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
)
# Property / field — accept ``{`` (auto-property), ``=`` (initializer) or ``;``.
_OBJ_CSHARP_FIELD = re.compile(
    r"\b(?:public|private|protected|internal)\s+(?:static\s+|readonly\s+|virtual\s+)*"
    r"[\w<>\[\],\.\?]+\s+(?P<name>[A-Za-z_]\w*)\s*(?:\{|=|;)",
    re.MULTILINE,
)


_OBJ_SNAKE_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_OBJ_SNAKE_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _to_snake(name: str) -> str:
    """Convert PascalCase / camelCase identifiers to snake_case.

    ``OrderId`` → ``order_id``, ``computeHash`` → ``compute_hash``,
    ``HTTPRequest`` → ``http_request``, ``order_id`` → ``order_id``.
    Used to normalise object-shape field names so that DTOs sharing
    the same logical schema across languages with different casing
    conventions (C# ``Id`` ↔ Python ``id``, Java ``orderId`` ↔ Python
    ``order_id``) collapse onto the same ``obj:`` surface key.
    """
    s = _OBJ_SNAKE_RE_1.sub(r"\1_\2", name)
    s = _OBJ_SNAKE_RE_2.sub(r"\1_\2", s)
    return s.lower()


def _obj_surface(name: str, fields: Iterable[str]) -> Optional["APISurface"]:
    """Build an ``obj:`` surface from a type name and field iterable.

    Returns ``None`` if the field set is empty or the type is unnamed.
    Field names are snake-cased + sorted + de-duplicated to make the
    surface stable across language casing conventions
    (Go ``ID`` ↔ TS ``id`` ↔ Python ``id``;
    C# ``OrderId`` ↔ Python ``order_id``).
    """
    name = (name or "").strip()
    if not name:
        return None
    norm: list[str] = []
    seen: set[str] = set()
    for f in fields:
        if not f:
            continue
        f_snake = _to_snake(f.strip())
        if not f_snake or f_snake in seen:
            continue
        if _OBJ_PY_DUNDER.match(f_snake):
            continue
        seen.add(f_snake)
        norm.append(f_snake)
    if not norm:
        return None
    norm.sort()
    return APISurface(
        kind="obj",
        surface=f"obj:{_to_snake(name)}#{','.join(norm)}",
        weight_hint=0.65,
        metadata={"type": name, "fields": norm},
    )


def _split_param_list(params: str) -> List[str]:
    """Extract field names from a ``(type x, type y)`` style param list."""
    out: List[str] = []
    depth = 0
    current = ""
    parts: List[str] = []
    for ch in params:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    for p in parts:
        toks = p.strip().split()
        if toks:
            # last token is the param name (handles `final int x`, `String name`).
            out.append(toks[-1].rstrip(",;"))
    return out


def _iter_obj_typescript(text: str) -> Iterable[Tuple[str, List[str]]]:
    for m in _OBJ_TS_INTERFACE.finditer(text):
        body = m.group("body") or ""
        fields = [f.group("name") for f in _OBJ_TS_FIELD.finditer(body)]
        yield m.group("name"), fields
    for m in _OBJ_TS_TYPE.finditer(text):
        body = m.group("body") or ""
        fields = [f.group("name") for f in _OBJ_TS_FIELD.finditer(body)]
        yield m.group("name"), fields


def _iter_obj_go(text: str) -> Iterable[Tuple[str, List[str]]]:
    for m in _OBJ_GO_STRUCT.finditer(text):
        body = m.group("body") or ""
        fields: List[str] = []
        for f in _OBJ_GO_FIELD.finditer(body):
            tag = f.group("tag") or ""
            tag_match = _OBJ_GO_JSON_TAG.search(tag)
            if tag_match:
                fields.append(tag_match.group(1))
            else:
                fields.append(f.group("name"))
        yield m.group("name"), fields


def _iter_obj_java(text: str) -> Iterable[Tuple[str, List[str]]]:
    for m in _OBJ_JAVA_CLASS_HEADER.finditer(text):
        # Scan from the class header to end-of-text; methods have ``)``
        # or ``{`` terminators, not ``;``/``=``, so the field regex
        # naturally skips them.
        body = text[m.end():]
        fields = [f.group("name") for f in _OBJ_JAVA_FIELD.finditer(body)]
        yield m.group("name"), fields
    for m in _OBJ_JAVA_RECORD.finditer(text):
        yield m.group("name"), _split_param_list(m.group("params") or "")


def _iter_obj_rust(text: str) -> Iterable[Tuple[str, List[str]]]:
    for m in _OBJ_RUST_STRUCT.finditer(text):
        body = m.group("body") or ""
        fields: List[str] = []
        for f in _OBJ_RUST_FIELD.finditer(body):
            rename = f.group("rename")
            fields.append(rename or f.group("name"))
        yield m.group("name"), fields


def _iter_obj_csharp(text: str) -> Iterable[Tuple[str, List[str]]]:
    for m in _OBJ_CSHARP_CLASS_HEADER.finditer(text):
        body = text[m.end():]
        fields = [f.group("name") for f in _OBJ_CSHARP_FIELD.finditer(body)]
        yield m.group("name"), fields
    for m in _OBJ_CSHARP_RECORD.finditer(text):
        yield m.group("name"), _split_param_list(m.group("params") or "")


def _iter_obj_python(text: str) -> Iterable[Tuple[str, List[str]]]:
    classes = list(_OBJ_PY_CLASS.finditer(text))
    for i, m in enumerate(classes):
        end = classes[i + 1].start() if i + 1 < len(classes) else len(text)
        body = text[m.end():end]
        fields: List[str] = []
        for f in _OBJ_PY_FIELD.finditer(body):
            fields.append(f.group("name"))
        yield m.group("name"), fields


def _match_objects(text: str, language: str) -> List[APISurface]:
    """Detect cross-language data-shape surfaces."""
    out: List[APISurface] = []
    iters = {
        "python": _iter_obj_python,
        "typescript": _iter_obj_typescript,
        "javascript": _iter_obj_typescript,
        "go": _iter_obj_go,
        "java": _iter_obj_java,
        "kotlin": _iter_obj_java,
        "rust": _iter_obj_rust,
        "csharp": _iter_obj_csharp,
        "c#": _iter_obj_csharp,
        "cs": _iter_obj_csharp,
    }
    fn = iters.get(language)
    if not fn:
        return out
    for name, fields in fn(text):
        s = _obj_surface(name, fields)
        if s is not None:
            out.append(s)
    return out


# ──────────────────────────────────────────────────────────────────────
# BDD (Gherkin step → step definition)
# ──────────────────────────────────────────────────────────────────────

_BDD_DECORATOR = re.compile(
    r"@\s*(?P<kind>given|when|then|step)\s*\(\s*['\"](?P<text>[^'\"]+)['\"]\s*\)",
    re.IGNORECASE,
)
_BDD_GHERKIN = re.compile(
    r"^\s*(?P<kind>Given|When|Then|And|But)\s+(?P<text>.+)$",
    re.MULTILINE,
)


def _match_bdd(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _BDD_DECORATOR.finditer(text):
        out.append(APISurface(
            kind="bdd",
            surface=f"bdd:{m.group('text').strip().lower()}",
            weight_hint=0.7,
            metadata={"kind": m.group("kind").lower()},
        ))
    for m in _BDD_GHERKIN.finditer(text):
        out.append(APISurface(
            kind="bdd",
            surface=f"bdd:{m.group('text').strip().lower()}",
            weight_hint=0.6,
            metadata={"kind": m.group("kind").lower()},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# CLI (argparse / click / cobra)
# ──────────────────────────────────────────────────────────────────────

_CLI_CLICK = re.compile(
    r"@\s*(?:\w+\.)?(?:command|group)\s*\(\s*(?:name\s*=\s*)?['\"]([^'\"]+)['\"]",
)
_CLI_ARGPARSE = re.compile(
    r"add_subparsers\s*\(.*?\)\.add_parser\s*\(\s*['\"]([^'\"]+)['\"]",
    re.DOTALL,
)
_CLI_COBRA = re.compile(r"&cobra\.Command\s*{[^}]*?Use:\s*['\"]([^'\"]+)['\"]", re.DOTALL)


def _match_cli(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _CLI_CLICK.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1)}",
            weight_hint=0.6,
            metadata={"framework": "click"},
        ))
    for m in _CLI_ARGPARSE.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1)}",
            weight_hint=0.6,
            metadata={"framework": "argparse"},
        ))
    for m in _CLI_COBRA.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1).split()[0]}",
            weight_hint=0.6,
            metadata={"framework": "cobra"},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────

_REST_BY_LANGUAGE: Dict[str, Callable[[str], List[APISurface]]] = {
    "python": _match_rest_python,
    "typescript": _match_rest_typescript,
    "javascript": _match_rest_typescript,
    "java": _match_rest_java,
    "kotlin": _match_rest_java,
    "go": _match_rest_go,
}


def extract_api_surfaces(
    node_data: dict,
    parser_metadata: Optional[dict] = None,
) -> List[APISurface]:
    """Return all API surfaces visible in *node_data*.

    The matchers operate on ``source_text`` plus a few normalised
    attributes (``language``, ``symbol_name``). ``parser_metadata`` is
    accepted but currently unused — reserved for matchers that need
    decorator AST detail beyond what survives in ``source_text``.
    """
    existing = node_data.get("api_surface") or []
    if existing:
        return [
            APISurface(**dict(surface))
            for surface in existing
            if isinstance(surface, dict)
        ]

    text = (node_data.get("source_text") or "")
    if not text:
        symbol = node_data.get("symbol")
        text = getattr(symbol, "source_text", "") if symbol is not None else ""
    if not text:
        return []
    language = (node_data.get("language") or "").lower()
    symbol_name = node_data.get("symbol_name") or ""

    surfaces: List[APISurface] = []

    rest_fn = _REST_BY_LANGUAGE.get(language)
    if rest_fn:
        surfaces.extend(rest_fn(text))

    surfaces.extend(_match_grpc(text, language))
    surfaces.extend(_match_graphql(text))
    surfaces.extend(_match_ffi(text, symbol_name))
    surfaces.extend(_match_objects(text, language))
    surfaces.extend(_match_bdd(text))
    surfaces.extend(_match_cli(text))

    # De-duplicate while preserving order; tag the first occurrence wins.
    seen = set()
    unique: List[APISurface] = []
    for s in surfaces:
        key = (s["kind"], s["surface"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


# ──────────────────────────────────────────────────────────────────────
# Phase 1c orchestrator
# ──────────────────────────────────────────────────────────────────────


def extract_api_surfaces_for_graph(
    g: "Any",  # nx.MultiDiGraph — annotation lazy to avoid import cost
    *,
    parser_metadata_by_node: Optional[Dict[str, dict]] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, List[APISurface]]:
    """Walk *g* and attach API-surface metadata to every node.

    Side effects: each node whose ``source_text`` exposes any surface
    gets its ``api_surface`` attribute set to the list of ``APISurface``
    dicts. Nodes without surfaces are left untouched (no key written).

    Returns a mapping ``{node_id: [APISurface, ...]}`` containing
    only the nodes for which at least one surface was detected. The
    return value is what :func:`run_cross_language_linker` expects as
    ``surfaces_by_node`` for its L1 pass.

    When ``repo_root`` is supplied the orchestrator pre-scans every
    Python file once for an ``APIRouter(prefix="/x")`` /
    ``Blueprint(... url_prefix="/x")`` declaration and propagates the
    found prefix into each node's ``source_text`` so the per-symbol
    matcher can prepend it to all routes. Without this, FastAPI/Flask
    routes declared on a prefixed router (e.g. ``@router.get("/me")``
    on ``APIRouter(prefix="/users")``) would emit ``GET /me`` and fail
    to pair against TS clients calling ``GET /api/v1/users/me``.

    Pure with respect to edges and to nodes that produce no surfaces.
    """
    parser_metadata_by_node = parser_metadata_by_node or {}

    # Per-file router-prefix cache (rel_path -> prefix string or "").
    file_prefix_cache: Dict[str, str] = {}
    # Per-file ctypes-loaded library variable cache (rel_path -> set of
    # variable names bound to ``ctypes.CDLL(...)`` calls).
    file_ctypes_libs: Dict[str, set] = {}
    # Per-file source-line cache for the empty-source_text fallback below
    # (rel_path -> list[str] of file lines, or [] if read failed).
    file_lines_cache: Dict[str, List[str]] = {}

    def _file_slice(rel_path: str, start_line: int, end_line: int) -> str:
        """Return source ``[start_line..end_line]`` (1-based, inclusive)
        from disk. Used when the parser left ``source_text`` empty for
        thin symbols. Examples: openapi-ts ``types.gen.ts`` ``type_alias``
        nodes, Go ``const`` group members. Without this, the obj/rest/
        grpc/etc. matchers have nothing to scan and these symbols never
        produce surfaces — even though their full body is on disk.
        """
        if not rel_path or not repo_root or start_line <= 0 or end_line < start_line:
            return ""
        lines = file_lines_cache.get(rel_path)
        if lines is None:
            try:
                from pathlib import Path as _Path
                full = _Path(repo_root) / rel_path
                if not full.is_file():
                    file_lines_cache[rel_path] = []
                    return ""
                lines = full.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                lines = []
            file_lines_cache[rel_path] = lines
        if not lines:
            return ""
        s = max(0, start_line - 1)
        e = min(len(lines), end_line)
        if e <= s:
            return ""
        return "\n".join(lines[s:e])

    def _python_router_prefix(rel_path: str) -> str:
        if not rel_path or not repo_root:
            return ""
        if rel_path in file_prefix_cache:
            return file_prefix_cache[rel_path]
        try:
            from pathlib import Path as _Path
            full = _Path(repo_root) / rel_path
            if not full.is_file():
                file_prefix_cache[rel_path] = ""
                return ""
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            file_prefix_cache[rel_path] = ""
            return ""
        m = _PY_ROUTER_PREFIX.search(content)
        prefix = m.group(1) if m else ""
        file_prefix_cache[rel_path] = prefix
        return prefix

    def _python_ctypes_lib_vars(rel_path: str) -> set:
        if not rel_path or not repo_root:
            return set()
        if rel_path in file_ctypes_libs:
            return file_ctypes_libs[rel_path]
        try:
            from pathlib import Path as _Path
            full = _Path(repo_root) / rel_path
            if not full.is_file():
                file_ctypes_libs[rel_path] = set()
                return set()
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            file_ctypes_libs[rel_path] = set()
            return set()
        names = {m.group("var") for m in _PY_CTYPES_LIB.finditer(content)}
        file_ctypes_libs[rel_path] = names
        return names

    out: Dict[str, List[APISurface]] = {}
    for node_id, data in g.nodes(data=True):
        # For Python nodes, splice the file-level router-prefix line into
        # the symbol's source_text so _match_rest_python's regex can find
        # it. Cheap one-line synthetic prepend; original text preserved.
        scratch_text: Optional[str] = None
        slice_filled = False
        original_source_text = data.get("source_text")
        language = (data.get("language") or "").lower()

        # Step 0: source-text slice fallback. Some parsers (notably the
        # TypeScript parser for ``type_alias`` symbols and certain Go
        # const groups) persist nodes with ``source_text`` set to ``None``
        # or empty even though the symbol body is fully recoverable from
        # the on-disk slice ``[start_line..end_line]``. Without this,
        # downstream matchers (``_match_objects`` etc.) get nothing and
        # those symbols never produce surfaces — silently dropping
        # cross-language pairs (e.g. openapi-ts ``UserPublic`` ↔
        # FastAPI ``UserPublic`` Pydantic model).
        if not (original_source_text or "") and data.get("rel_path"):
            try:
                sl = int(data.get("start_line") or 0)
                el = int(data.get("end_line") or 0)
            except (TypeError, ValueError):
                sl = el = 0
            sliced = _file_slice(data.get("rel_path") or "", sl, el)
            if sliced:
                data["source_text"] = sliced
                slice_filled = True

        if language == "python" and not data.get("api_surface"):
            prefix = _python_router_prefix(data.get("rel_path") or "")
            if prefix:
                base = data.get("source_text") or ""
                scratch_text = f'router = APIRouter(prefix="{prefix}")\n' + base
                data["source_text"] = scratch_text
        try:
            surfaces = extract_api_surfaces(
                data, parser_metadata=parser_metadata_by_node.get(str(node_id))
            )
            # Python ctypes wrapper detection (file-level lib var ↔
            # ``<libvar>.<func>(`` call sites in the symbol body).
            if language == "python":
                lib_vars = _python_ctypes_lib_vars(data.get("rel_path") or "")
                body = data.get("source_text") or ""
                if lib_vars and body:
                    seen_ffi = {s["surface"] for s in surfaces if s["kind"] == "ffi"}
                    for lv in lib_vars:
                        for m in re.finditer(
                            rf"\b{re.escape(lv)}\.(?P<fn>[A-Za-z_]\w*)\s*\(", body
                        ):
                            key = f"ffi:{m.group('fn')}"
                            if key in seen_ffi:
                                continue
                            seen_ffi.add(key)
                            surfaces.append(APISurface(
                                kind="ffi",
                                surface=key,
                                weight_hint=0.6,
                                metadata={
                                    "symbol": m.group("fn"),
                                    "via": f"{lv} = ctypes.CDLL(...)",
                                },
                            ))
        except Exception:  # pragma: no cover — defensive; matchers are regex-only
            surfaces = []
        finally:
            # Restore the original source_text if we mutated it so we
            # don't leak the synthetic prefix line into downstream
            # consumers (chunkers, retrievers, vector store).
            if scratch_text is not None:
                data["source_text"] = scratch_text[scratch_text.find("\n") + 1:]
            # If we filled the slice purely for this matcher pass, drop
            # it back to whatever the parser originally left (typically
            # ``None``) so we don't materialise large bodies into nodes
            # that the parser intentionally kept thin.
            if slice_filled:
                data["source_text"] = original_source_text
        if not surfaces:
            continue
        # Mutate the in-memory node attrs so the SQLite/Postgres
        # serialisers persist the surfaces via the ``api_surface``
        # column.
        data["api_surface"] = surfaces
        out[str(node_id)] = surfaces
    return out
