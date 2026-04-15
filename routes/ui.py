#!/usr/bin/python3
# coding=utf-8

""" Plugin UI Route """

import os
import json
import flask
from pylon.core.tools import web

class Route:
    """ UI route to serve static files """
    # Routes for ui_host proxy (project_id comes from X-Project-Id header)
    @web.route("/ui/<int:toolkit_id>", endpoint="ui_route_with_toolkit")
    @web.route("/ui/<int:toolkit_id>/assets/<path:asset_path>", endpoint="ui_route_assets")
    # Routes for direct access (project_id in path)
    @web.route("/ui/<int:project_id>/<int:toolkit_id>", endpoint="ui_route_direct")
    @web.route("/ui/<int:project_id>/<int:toolkit_id>/assets/<path:asset_path>", endpoint="ui_route_assets_direct")
    def ui_route(self, project_id=None, toolkit_id=None, asset_path=None):
        """ Serve static UI files """
        # Get the plugin directory
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(plugin_dir, "static", "ui", "dist")
        
        # If asset_path is provided, serve the asset file directly
        if asset_path:
            response = flask.send_from_directory(static_dir, f"assets/{asset_path}")
            # Vite emits hashed asset filenames by default, safe to cache aggressively.
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
            return response
        
        # Otherwise serve index.html with runtime config injection
        idx_path = os.path.join(static_dir, "index.html")
        
        if not os.path.exists(idx_path):
            return "UI not built. Please run: cd static/ui && ./build.sh", 404
        
        # Read index.html
        with open(idx_path, "r", encoding="utf-8") as idx_file:
            idx_data = idx_file.read()
        
        # Determine base_uri from request headers
        # When accessed via ui_host proxy, custom headers are injected:
        # X-Project-Id, X-User-Id, etc.
        # The full client path is: /app/ui_host/deepwiki/ui/{project_id}/...
        # But the plugin only sees: /ui/...
        
        # Check if we're being accessed through ui_host by looking for injected headers
        header_project_id = flask.request.headers.get('X-Project-Id')
        effective_project_id = header_project_id or project_id
        
        if header_project_id:
            # We're being accessed via ui_host proxy
            # Reconstruct the base_uri: /app/ui_host/deepwiki/ui/{project_id}/{toolkit_id}
            base_uri = f"/app/ui_host/deepwiki/ui/{effective_project_id}/{toolkit_id}"
        elif project_id:
            # Direct access with project_id in path
            base_uri = f"/ui/{project_id}/{toolkit_id}"
        else:
            # Direct access to the plugin (legacy)
            base_uri = f"/ui/{toolkit_id}" if toolkit_id else "/ui"
        
        # Create runtime config
        deepwiki_ui_config = {
            "base_uri": base_uri,
        }
        
        # Inject config script
        config_script = f'<script>window.deepwiki_ui_config = {json.dumps(deepwiki_ui_config)};</script>'
        idx_data = idx_data.replace(
            '<!-- deepwiki_ui_config -->',
            config_script
        )
        
        # Rewrite asset paths to use base_uri
        # Vite builds with base: './' so assets are referenced as ./assets/...
        # We need to rewrite these to {base_uri}/assets/...
        idx_data = idx_data.replace(
            'src="./assets', f'src="{base_uri}/assets'
        )
        idx_data = idx_data.replace(
            'href="./assets', f'href="{base_uri}/assets'
        )
        
        response = flask.make_response(idx_data, 200)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        # Ensure clients always revalidate index.html so new hashed bundles get picked up.
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
