import os
import sys

print("="*60)
print("🔍 Flask App Diagnostic Check")
print("="*60)

# Check current directory
print(f"\n📁 Current directory: {os.getcwd()}")

# Check if app.py exists
print(f"\n✓ app.py exists: {os.path.exists('app.py')}")

# Check if templates folder exists
print(f"✓ templates/ folder exists: {os.path.exists('templates')}")

# Check if index.html exists
print(f"✓ templates/index.html exists: {os.path.exists('templates/index.html')}")

# Check if trained model exists
print(f"✓ q_agent.pkl exists: {os.path.exists('q_agent.pkl')}")

# Check requirements
print(f"\n📦 Checking Python packages:")
try:
    import flask
    print(f"✓ Flask version: {flask.__version__}")
except ImportError:
    print("❌ Flask not installed. Run: pip install flask")

try:
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
except ImportError:
    print("❌ NumPy not installed. Run: pip install numpy")

# Try to load the agent
if os.path.exists('q_agent.pkl'):
    try:
        import pickle
        with open('q_agent.pkl', 'rb') as f:
            agent = pickle.load(f)
        print(f"\n✓ Agent loaded successfully")
        print(f"  - Q-values: {sum(len(agent.Q[a]) for a in agent.actions)}")
    except Exception as e:
        print(f"\n❌ Error loading agent: {e}")
else:
    print(f"\n⚠️  No trained agent found. Run training cells in notebook first.")

print("\n" + "="*60)
print("💡 If everything shows ✓, run: python app.py")
print("="*60)
