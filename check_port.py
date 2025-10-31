import socket
import sys

def check_port(port):
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return True
    except OSError:
        return False

print("="*60)
print("🔍 Port Checker")
print("="*60)

port = 5000
if check_port(port):
    print(f"\n✅ Port {port} is AVAILABLE")
    print(f"   You can run the Flask app on this port.")
else:
    print(f"\n❌ Port {port} is BUSY")
    print(f"   Another program is using this port.")
    print(f"\n💡 Solutions:")
    print(f"   1. Close programs that might use port 5000")
    print(f"   2. Try a different port (5001, 5002, etc.)")
    print(f"   3. Run: netstat -ano | findstr :5000")
    print(f"      To see what's using the port")

# Check alternative ports
print("\n📊 Checking alternative ports...")
for alt_port in [5001, 5002, 8000, 8080]:
    status = "✅ Available" if check_port(alt_port) else "❌ Busy"
    print(f"   Port {alt_port}: {status}")

print("\n" + "="*60)
input("\nPress Enter to exit...")
