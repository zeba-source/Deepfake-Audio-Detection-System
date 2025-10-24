from flask import Flask
import sys

print("Python version:", sys.version)
print("Starting minimal Flask test...")

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask!"

if __name__ == '__main__':
    print("About to call app.run()...")
    app.run(host='0.0.0.0', port=5001, debug=False)
    print("app.run() returned - this should not happen!")
