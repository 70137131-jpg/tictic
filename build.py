"""
Build script to prepare files for Vercel deployment
"""
import os
import shutil

# Create public directory
os.makedirs('public/static', exist_ok=True)

# Copy static files
if os.path.exists('static'):
    for file in os.listdir('static'):
        src = os.path.join('static', file)
        dst = os.path.join('public/static', file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")

print("Build complete!")
