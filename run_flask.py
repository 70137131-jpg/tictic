"""
Simple launcher for the Flask TicTacToe app.
Run this file to start the server.
"""
import os
import sys

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("="*60)
print("Starting Flask App...")
print(f"Directory: {os.getcwd()}")
print("="*60)

# Import and run the app
try:
    from app import app, load_agent
    
    # Pre-check
    if not os.path.exists('q_agent.pkl'):
        print("\n⚠️  WARNING: No trained model found (q_agent.pkl)")
        print("Please train the model first using the notebook.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\nLoading agent...")
    agent = load_agent()
    print("✅ Agent loaded!")
    
    print("\n" + "="*60)
    print("🚀 Server starting on http://127.0.0.1:5000")
    print("🛑 Press CTRL+C to stop")
    print("="*60 + "\n")
    
    # Start server
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\n💡 Try installing requirements:")
    print("   pip install flask numpy")
    input("\nPress Enter to exit...")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)
