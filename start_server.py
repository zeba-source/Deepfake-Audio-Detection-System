"""
Simple server starter that keeps Flask alive
"""
import subprocess
import sys
import time

print("="*70)
print("  🚀 STARTING DEEPFAKE DETECTION SERVER")
print("="*70)
print("\n📌 Server will run on: http://localhost:5001")
print("⚠️  Keep this terminal window open!")
print("🛑 Press Ctrl+C to stop the server\n")
print("="*70 + "\n")

try:
    # Run the Flask app
    python_exe = r"E:\Downloadsss\Deepfake\.venv\Scripts\python.exe"
    process = subprocess.Popen(
        [python_exe, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')
    
    process.wait()
    
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user (Ctrl+C)")
    if process:
        process.terminate()
        process.wait()
except Exception as e:
    print(f"\n\n❌ Error: {e}")
finally:
    print("\n✅ Server shutdown complete\n")
