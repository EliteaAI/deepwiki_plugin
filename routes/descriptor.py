#!/usr/bin/python3
# coding=utf-8

""" Plugin Descriptor Route """

from pylon.core.tools import web


class Route:
    """ Descriptor route """

    @web.route("/descriptor")
    def descriptor_route(self):
        """ Return plugin descriptor """
        config = self.runtime_config()
        
        descriptor = {
            "name": "WikiBuilderServiceProvider",
            "service_location_url": config["service_location_url"],
            "configuration": {},
            "provided_toolkits": [
                {
                    "name": "WikiBuilderToolkit",
                    "description": "Comprehensive wiki generation from GitHub repository analysis",
                    "toolkit_config": {
                        "type": "Wiki Builder Configuration",
                        "description": "Configuration for Wiki Builder.",
                        "parameters": {
                            "github_repository": {
                                "type": "String",
                                "description": "GitHub repository in format 'owner/repo'"
                            },
                            "github_access_token": {
                                "type": "String",
                                "description": "GitHub personal access token"
                            },
                            "github_base_branch": {
                                "type": "String",
                                "description": "Base branch to analyze"
                            },
                            "active_branch": {
                                "type": "String",
                                "description": "Active branch to analyze"
                            },
                            "github_base_url": {
                                "type": "String",
                                "description": "GitHub API base URL (for GitHub Enterprise)"
                            },
                            "max_files": {
                                "type": "Integer",
                                "description": "Maximum number of files to analyze"
                            },
                            "max_depth": {
                                "type": "Integer",
                                "description": "Maximum directory depth to traverse"
                            },
                            "rate_limit_delay": {
                                "type": "Number",
                                "description": "Delay between GitHub API calls"
                            },
                            "enable_semantic_chunking": {
                                "type": "Boolean",
                                "description": "Enable semantic chunking for better analysis"
                            },
                            "parallel_processing": {
                                "type": "Boolean",
                                "description": "Enable parallel processing for faster analysis"
                            },
                            "bucket": {
                                "type": "String",
                                "description": "Bucket for storing wiki artifacts"
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
                                "wiki_title": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Title for the generated wiki"
                                },
                                "include_research": {
                                    "type": "Boolean",
                                    "required": False,
                                    "description": "Include web research in wiki generation"
                                },
                                "include_diagrams": {
                                    "type": "Boolean",
                                    "required": False,
                                    "description": "Generate diagrams for documentation"
                                },
                                "output_format": {
                                    "type": "String",
                                    "required": False,
                                    "description": "Output format: json, markdown"
                                }
                            },
                            "description": "Generate comprehensive wiki from repository analysis",
                            "tool_metadata": {
                                "result_target": "artifact",
                                "result_extension": "json",
                                "result_encoding": "utf-8"
                            },
                            "tool_result_type": "String",
                            "sync_invocation_supported": True,
                            "async_invocation_supported": False
                        }
                    ],
                    "toolkit_metadata": {
                        "elitea_llm_required": True,
                        "elitea_toolkits_required": [
                            "artifact"
                        ]
                    }
                }
            ]
        }
        
        return descriptor
