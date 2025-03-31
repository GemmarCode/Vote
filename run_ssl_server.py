import os
import sys
from django.core.management import execute_from_command_line
from django.core.servers import basehttp
import ssl
import socket
import logging

class SecureHTTPServer(basehttp.HTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket = ssl.wrap_socket(
            self.socket,
            keyfile="key.pem",
            certfile="cert.pem",
            server_side=True
        )

class SecureWSGIServer(basehttp.WSGIServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket = ssl.wrap_socket(
            self.socket,
            keyfile="key.pem",
            certfile="cert.pem",
            server_side=True
        )

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartvote.settings')
    
    # Check if SSL certificates exist
    if not (os.path.exists('cert.pem') and os.path.exists('key.pem')):
        print("SSL certificates not found. Generating new ones...")
        from generate_cert import generate_self_signed_cert
        generate_self_signed_cert()
    
    # Check if port 8443 is available
    port = 8443
    if not is_port_available(port):
        print(f"Port {port} is already in use. Please free up the port or use a different one.")
        sys.exit(1)
    
    try:
        # Override the default server class
        basehttp.WSGIServer = SecureWSGIServer
        basehttp.HTTPServer = SecureHTTPServer
        
        # Run the server
        execute_from_command_line(['manage.py', 'runserver', f'0.0.0.0:{port}'])
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 