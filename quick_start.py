import os
import sys
import subprocess

def check_file(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def main():
    print("="*60)
    print("🚀 TicTacToe Flask App - Quick Start")
    print("="*60)
    print()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    print()
    
    # Check all required files
    print("📋 Checking required files...")
    checks = [
        check_file("app.py", "Flask application"),
        check_file("templates/index.html", "HTML template"),
        check_file("q_agent.pkl", "Trained model"),
        check_file("requirements.txt", "Requirements file")
    ]
    print()
    
    if not all(checks):
        print("❌ Some required files are missing!")
        print("\n💡 To fix:")
        print("   1. Make sure you're in the correct folder")
        print("   2. Train the model using the notebook (creates q_agent.pkl)")
        print("   3. Ensure templates folder has index.html")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("✅ All files found!")
    print()
    
    # Check Python packages
    print("📦 Checking Python packages...")
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError:
        print("✗ Flask not installed")
        print("\nInstalling Flask...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "numpy"])
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        print("\nInstalling NumPy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    print()
    print("="*60)
    print("🎮 Starting Flask server...")
    print("="*60)
    print()
    print("📱 Open browser: http://127.0.0.1:5000")
    print("🛑 Press CTRL+C to stop")
    print()
    
    # Start the Flask app
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
