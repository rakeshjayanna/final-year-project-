from flask import Flask
from flask_cors import CORS
from routes.detect import detect_bp
import os

app = Flask(__name__)

allowed_origins = os.getenv('CORS_ORIGINS', '*')
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

app.register_blueprint(detect_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
