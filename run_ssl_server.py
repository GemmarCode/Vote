import os
import sys
import ssl
import socket

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartvote.settings')

# Import Django and set up the application
import django
django.setup()

from django.core.management import call_command
from django.core.servers.basehttp import WSGIServer
from django.core.wsgi import get_wsgi_application

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def run_ssl_server(host='0.0.0.0', port=8000):
    # Get the WSGI application
    application = get_wsgi_application()
    
    # Import the SSL-enabled server
    from werkzeug.serving import run_simple
    
    # Run the server with SSL
    run_simple(
        host, 
        port, 
        application,
        ssl_context=(
            'cert.pem',  # Path to certificate file
            'key.pem'    # Path to key file
        ),
        use_reloader=True
    )

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"Starting SSL server at https://{local_ip}:8000/")
    print(f"You can also access it at https://localhost:8000/")
    print("Using certificate: cert.pem")
    print("Using key: key.pem")
    print("Quit the server with CTRL-C")
    
    try:
        run_ssl_server()
    except KeyboardInterrupt:
        print("\nServer stopped.") 