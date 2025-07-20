#!/usr/bin/python3
# coding=utf-8

""" Tool Invocation Route """

import uuid
import os
import flask
import traceback
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pylon.core.tools import log, web

# Import the actual working wiki toolkit components
from ..plugin_implementation.wiki_toolkit_wrapper import OptimizedWikiToolkitWrapper

logger = logging.getLogger(__name__)

default_bucket = 'wiki_artifacts'


class Route:
    """ Invocation route """

    @web.route("/tools/<toolkit_name>/<tool_name>/invoke", methods=["POST"])
    def invoke_route(self, toolkit_name, tool_name):
        """ Handle tool invocation """
        
        # Validate toolkit
        if toolkit_name != "WikiBuilderToolkit":
            return {
                "errorCode": "404",
                "message": "Toolkit not found",
                "details": [f"Unknown toolkit: {toolkit_name}"]
            }, 404
        
        try:
            # Get request data
            request_data = flask.request.json
            if not request_data or "parameters" not in request_data:
                return {
                    "errorCode": "400",
                    "message": "Missing parameters",
                    "details": ["Request must include 'parameters' field"]
                }, 400
            
            parameters = request_data["parameters"]
            
            # Route to appropriate tool
            if tool_name == "generate_wiki":
                result = self._handle_generate_wiki(parameters)
            else:
                return {
                    "errorCode": "404",
                    "message": "Tool not found",
                    "details": [f"Unknown tool: {tool_name}"]
                }, 404
            
            # Generate invocation ID
            invocation_id = str(uuid.uuid4())
            
            # Return success response
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "result": result,
                "result_type": "String"
            }
            
        except Exception as e:
            log.exception(f"Tool invocation failed: {toolkit_name}:{tool_name}")
            return {
                "errorCode": "500",
                "message": "Internal server error",
                "details": [str(e)]
            }, 500

    def _handle_generate_wiki(self, parameters):
        """ Handle wiki generation tool """
        # Validate required parameters
        required_params = ['query']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        try:

            # Get runtime config for cache directories
            runtime_config = self.runtime_config()
            
            # Set environment variables for Hugging Face cache directories
            model_cache_dir = runtime_config.get('sentence_transformers_cache', f"{runtime_config['base_path']}/huggingface_cache")
            os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
            os.environ['HF_HOME'] = model_cache_dir
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
            
            # Create the wiki toolkit wrapper using config values from self.config
            # Pass the LLM and artifact toolkit like browseruse plugin does
            wrapper = OptimizedWikiToolkitWrapper(
                github_repository=self.config.github_repository,
                github_access_token=self.config.github_access_token,
                github_base_branch=getattr(self.config, 'github_base_branch', 'main'),
                active_branch=getattr(self.config, 'active_branch', 'main'),
                github_base_url=getattr(self.config, 'github_base_url', 'https://api.github.com'),
                max_files=getattr(self.config, 'max_files', None),
                max_depth=getattr(self.config, 'max_depth', None),
                rate_limit_delay=getattr(self.config, 'rate_limit_delay', 0.1),
                max_file_size=getattr(self.config, 'max_file_size', 1000000),
                parallel_processing=getattr(self.config, 'parallel_processing', True),
                enable_semantic_chunking=getattr(self.config, 'enable_semantic_chunking', True),
                cache_dir=runtime_config.get('cache_dir', f"{runtime_config['base_path']}/cache"),
                model_cache_dir=runtime_config.get('sentence_transformers_cache', f"{runtime_config['base_path']}/huggingface_cache"),
                force_rebuild_index=parameters.get('force_rebuild_index', False),
                llm=self.llm,  # Pass the LLM from plugin framework
                alita=None,    # Don't use AlitaClient in plugin, use artifact callback instead
                artifact_callback=self.artifact  # Pass the plugin artifact toolkit
            )
            
            # Extract generation parameters
            query = parameters['query']
            wiki_title = parameters.get('wiki_title', f"Wiki: {query}")
            include_research = parameters.get('include_research', True)
            include_diagrams = parameters.get('include_diagrams', True)
            output_format = parameters.get('output_format', 'json')
            
            # Generate the wiki
            log.info(f"Starting wiki generation for query: {query}")
            result = wrapper.generate_wiki(
                query=query,
                wiki_title=wiki_title,
                include_research=include_research,
                include_diagrams=include_diagrams,
                output_format=output_format
            )
            
            # Save the result as an artifact
            artifact_filename = f"wiki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            artifact_content = result if isinstance(result, str) else str(result)
            
            # Store the artifact like in browseruse
            files = self._save_wiki_artifact(artifact_content, artifact_filename)
            
            log.info(f"Wiki generation completed successfully. Generated {len(files)} files.")
            
            return {
                "wiki_content": artifact_content,
                "files": files,
                "metadata": {
                    "query": query,
                    "title": wiki_title,
                    "format": output_format,
                    "timestamp": datetime.now().isoformat(),
                    "repository": self.config.github_repository,
                    "branch": getattr(self.config, 'active_branch', 'main')
                }
            }
            
        except Exception as e:
            log.error(f"Wiki generation failed: {str(e)}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_wiki_artifact(self, content: str, filename: str) -> List[str]:
        """Save wiki content as an artifact"""
        try:
            # Store the main content
            self.artifact.create(filename, content)
            files = [filename]
            
            # Also create a metadata file
            metadata_filename = f"{filename}.meta"
            metadata = {
                "created_at": datetime.now().isoformat(),
                "content_length": len(content),
                "filename": filename
            }
            self.artifact.create(metadata_filename, str(metadata))
            files.append(metadata_filename)
            
            return files
        except Exception as e:
            log.error(f"Failed to save wiki artifact: {str(e)}")
            return []
