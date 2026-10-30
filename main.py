import os
import sys

# Ensure flask_app is in the Python search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flask_app"))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

