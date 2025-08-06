#!/usr/bin/env python3
"""
Simple web server for chess game reviewer.

This module provides a basic HTTP server for the chess game reviewer UI.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler


class ChessHandler(BaseHTTPRequestHandler):
    """HTTP request handler for chess game reviewer."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Chess Game Reviewer UI - TODO")
        else:
            self.send_response(404)
            self.end_headers()


def main():
    """Start the web server."""
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, ChessHandler)
    print("Chess server starting on http://localhost:8080")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

# TODO: Add endpoints for loading PGN, displaying board, etc.
# Integrate with frontend JS for chessboard.