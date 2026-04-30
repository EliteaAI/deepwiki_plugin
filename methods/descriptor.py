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

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def provider_descriptor(self):
        """ Descriptor """
        service_location_url = self.descriptor.config.get(
            "service_location_url", "http://127.0.0.1:8080"
        )
        #
        return {
            "name": "deepwiki",
            "service_location_url": service_location_url,
            "configuration": {
                "provided_ui": [
                    {
                        "name": "ui",
                        "path": "/ui",
                        "headers": {
                            "X-User-Id": {"type": "user_id"},
                            "X-Project-Id": {"type": "project_id"}
                        }
                    },
                    {
                        "name": "api",
                        "path": "/",
                        "headers": {
                            "X-User-Id": {"type": "user_id"},
                            "X-Project-Id": {"type": "project_id"}
                        }
                    }
                ]
            },
            "provided_toolkits": [
                {
                    "name": "Deepwiki",
                    "description": "Comprehensive wiki generation from GitHub repository analysis",
                    "toolkit_config": {
                        "type": "Wiki Builder Configuration",
                        "description": "Configuration for Wiki Builder.",
                        "fields_order": [
                            "code_toolkit",
                            "llm_model",
                            "max_tokens",
                            "embedding_model"
                        ],
                        "parameters": {
                            "code_toolkit": {
                                "type": "Integer",
                                "required": True,
                                "description": "Code toolkit ID for accessing repository data",
                                "json_schema_extra": {
                                    "toolkit_types": ["github", "gitlab", "bitbucket", "ado_repos"]
                                }
                            },
                            "llm_model": {
                                "type": "String",
                                "required": True,
                                "description": "LLM Model to use for text generation",
                                "json_schema_extra": {
                                    "configuration_model": "llm",
                                }
                            },
                            "max_tokens": {
                                "type": "Integer",
                                "required": True,
                                "description": "Maximum tokens for LLM responses",
                                "default": 64000
                            },
                            "embedding_model": {
                                "type": "JSON",
                                "required": True,
                                "description": "Embedding model to use with embeddings",
                                "json_schema_extra": {
                                    "configuration_model": "embedding",
                                }
                            }
                        }
                    },
                    "provided_tools": [
                        {
                            "name": "generate_wiki",
                            "args_schema": {
                                "query": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Query or description of what wiki content to generate"
                                },
                                "planner_type": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Structure planner to use: 'cluster' (default, deterministic graph-clustering planner) or 'deepagents' (agentic loop). Aliases 'agent'/'agentic' are accepted."
                                },
                                "exclude_tests": {
                                    "type": "Boolean",
                                    "required": False,
                                    "description": "When using the cluster planner, skip test files so they don't form their own wiki pages. Test nodes are still indexed and available to ask/deep_research."
                                }
                            },
                            "description": "Generate comprehensive wiki from repository analysis",
                            "tool_metadata": {
                                "result_composition": "list_of_objects",
                                "result_objects": [
                                    {
                                        "object_type": "message",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    },
                                    {
                                        "object_type": "wiki_structure",
                                        "result_target": "artifact",
                                        "result_extension": "json",
                                        "result_encoding": "plain",
                                        "result_bucket": "wiki"
                                    },
                                    {
                                        "object_type": "wiki_manifest",
                                        "result_target": "artifact",
                                        "result_extension": "json",
                                        "result_encoding": "plain",
                                        "result_bucket": "wiki"
                                    },
                                    {
                                        "object_type": "wiki_page",
                                        "result_target": "artifact",
                                        "result_extension": "md",
                                        "result_encoding": "plain",
                                        "result_bucket": "wiki"
                                    },
                                    {
                                        "object_type": "repository_context",
                                        "result_target": "artifact",
                                        "result_extension": "txt",
                                        "result_encoding": "plain",
                                        "result_bucket": "wiki"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "ask",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Question about the repository code or functionality"
                                },
                                "repo_identifier_override": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional canonical repo identifier override (e.g., owner/repo:branch:commit8) to pin caches"
                                },
                                "analysis_key_override": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional analysis key override (e.g., owner/repo:branch:commit8@wiki_version_id) to pin repository analysis to a wiki version"
                                }
                            },
                            "description": "Ask questions about the repository - requires wiki to be generated first",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "answer",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "deep_research",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Research question for deep analysis of the repository"
                                },
                                "repo_identifier_override": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional canonical repo identifier override (e.g., owner/repo:branch:commit8) to pin caches"
                                },
                                "analysis_key_override": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional analysis key override (e.g., owner/repo:branch:commit8@wiki_version_id) to pin repository analysis to a wiki version"
                                }
                            },
                            "description": "Deep research on a repository topic - uses multi-step analysis with planning, delegation, and comprehensive investigation. Requires wiki to be generated first.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "answer",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        }
                    ],
                    "toolkit_metadata": {
                        "application": True,
                        "interface": {
                            "type": "iframe",
                            "create_url": None,
                            "app_url": "/app/ui_host/deepwiki/ui/{project_id}/{toolkit_id}?theme={theme}"
                        }
                    }
                },
                # ============================================================
                # DEEPWIKI_QUERY TOOLKIT - Read-only access to existing wiki
                # ============================================================
                # This toolkit references an existing deepwiki application and
                # exposes only read-only query tools (ask, deep_research) for use by other agents.
                # Unlike the main "Deepwiki" toolkit (which is an application),
                # this provides direct tool access without needing to configure repository access.
                {
                    "name": "deepwiki_query",
                    "description": "Read-only access to an existing DeepWiki knowledge base. Ask questions and perform deep research on a repository that has already been analyzed. Use this when you want to query an existing wiki without setting up your own.",
                    "toolkit_config": {
                        "type": "DeepWiki Query Tools",
                        "description": "Connect to an existing DeepWiki to ask questions and perform research. Select a DeepWiki toolkit that has already been configured and has a wiki generated.",
                        "fields_order": [
                            "deepwiki_toolkit",
                            "llm_model",
                            "embedding_model"
                        ],
                        "parameters": {
                            "deepwiki_toolkit": {
                                "type": "Integer",
                                "required": True,
                                "description": "ID of an existing DeepWiki toolkit to connect to",
                                "json_schema_extra": {
                                    "toolkit_types": ["deepwiki_Deepwiki"],
                                    "application": True,
                                    "label": "DeepWiki Toolkit"
                                }
                            },
                            "llm_model": {
                                "type": "String",
                                "required": True,
                                "description": "LLM Model to use for answering questions",
                                "json_schema_extra": {
                                    "configuration_model": "llm"
                                }
                            },
                            "embedding_model": {
                                "type": "JSON",
                                "required": True,
                                "description": "Embedding model for vector search",
                                "json_schema_extra": {
                                    "configuration_model": "embedding"
                                }
                            }
                        }
                    },
                    "provided_tools": [
                        {
                            "name": "ask",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Question about the repository code or functionality"
                                }
                            },
                            "description": "Ask questions about the repository using an existing DeepWiki knowledge base. The referenced DeepWiki toolkit must have a wiki already generated.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "answer",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "deep_research",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Research question for deep analysis of the repository"
                                }
                            },
                            "description": "Perform deep research on a repository topic using an existing DeepWiki knowledge base. Uses multi-step analysis with planning, delegation, and comprehensive investigation. The referenced DeepWiki toolkit must have a wiki already generated.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "answer",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        }
                    ],
                    "toolkit_metadata": {
                        "type_override": "deepwiki_query",
                        "application": False,
                        "required_context": ["project_id"]
                    }
                },
                # ============================================================
                # WIKI_QUERY TOOLKIT - Context7-style multi-wiki query
                # ============================================================
                # This toolkit provides discovery and query access to ALL wikis
                # in the registry without needing to know specific repo details.
                # LLM-based resolution: "how does elitea-sdk handle auth?" ->
                # finds the relevant wiki and answers the question.
                {
                    "name": "wiki_query",
                    "description": "Context7-style access to multiple DeepWiki knowledge bases. Discover available wikis, resolve repository queries automatically, and ask questions across different repositories. Perfect for AI agents that need to query multiple codebases without prior knowledge of which repository to use.",
                    "toolkit_config": {
                        "type": "Multi-Wiki Query Tools",
                        "description": "Connect to the wiki registry to discover and query multiple wikis. Configure LLM and embedding models for intelligent repository resolution and question answering.",
                        "fields_order": [
                            "llm_model",
                            "embedding_model"
                        ],
                        "parameters": {
                            "llm_model": {
                                "type": "String",
                                "required": True,
                                "description": "LLM Model to use for repository resolution and answering questions",
                                "json_schema_extra": {
                                    "configuration_model": "llm"
                                }
                            },
                            "embedding_model": {
                                "type": "JSON",
                                "required": True,
                                "description": "Embedding model for vector search",
                                "json_schema_extra": {
                                    "configuration_model": "embedding"
                                }
                            }
                        }
                    },
                    "provided_tools": [
                        {
                            "name": "list_wikis",
                            "args_schema": {
                                "include_metadata": {
                                    "type": "Boolean",
                                    "required": False,
                                    "default": False,
                                    "description": "Include full metadata for each wiki (repository URL, creation date, etc.)"
                                }
                            },
                            "description": "List all available wikis in the registry. Returns wiki IDs, repository names, and descriptions for discovery. Use this to understand what repositories have been indexed before asking questions.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "wiki_list",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "resolve_and_ask",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Question that may reference a repository. The system will automatically determine which wiki to query. Examples: 'How does elitea-sdk handle authentication?', 'What is the structure of the Pylon framework?'"
                                },
                                "wiki_id_hint": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional wiki ID hint if you already know which wiki to query. Format: owner--repo--branch"
                                }
                            },
                            "description": "Intelligently resolve which repository to query and answer the question. Uses LLM to match questions to available wikis based on repository names and descriptions. If wiki_id_hint is provided, skips resolution.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "answer",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "resolve_and_deep_research",
                            "args_schema": {
                                "question": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Complex question requiring deep analysis. The system will automatically determine which wiki to query."
                                },
                                "wiki_id_hint": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Optional wiki ID hint if you already know which wiki to query. Format: owner--repo--branch"
                                },
                                "research_type": {
                                    "type": "String",
                                    "required": False,
                                    "default": "general",
                                    "description": "Type of research: 'general', 'architecture', 'security', 'performance'"
                                }
                            },
                            "description": "Resolve which repository to query and perform deep multi-step research. Use this for complex questions requiring analysis of code patterns, architecture, or cross-cutting concerns.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "report",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        },
                        {
                            "name": "delete_wiki",
                            "args_schema": {
                                "wiki_id": {
                                    "type": "String",
                                    "required": True,
                                    "description": "Wiki ID to delete. Format: owner--repo--branch. Use list_wikis to see available wiki IDs."
                                }
                            },
                            "description": "Delete a wiki and all its associated artifacts. This removes all wiki pages, manifests, and cache files for the specified wiki, and removes it from the registry. This action cannot be undone.",
                            "tool_metadata": {
                                "result_composition": "single_object",
                                "result_objects": [
                                    {
                                        "object_type": "message",
                                        "result_target": "response",
                                        "result_encoding": "plain"
                                    }
                                ]
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": True
                        }
                    ],
                    "toolkit_metadata": {
                        "type_override": "wiki_query",
                        "application": False,
                        "required_context": ["project_id"]
                    }
                }
            ],

        }