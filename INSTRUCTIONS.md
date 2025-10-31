# Flask App Setup Instructions

## 🚨 If you see "ERR_CONNECTION_REFUSED"

This means the Flask server is NOT running. Follow these steps:

### Step 1: Check Python Installation
```cmd
python --version
```
Should show Python 3.7 or higher.

### Step 2: Navigate to Project Directory
```cmd
cd "c:\Users\Ali Haider\Desktop\reinforecemtn"
```

### Step 3: Install Requirements
```cmd
pip install flask numpy
```

### Step 4: Check Port Availability
```cmd
python check_port.py
```
- If port 5000 is busy, change the port in app.py

### Step 5: Verify Model Exists
```cmd
dir q_agent.pkl
```
- If not found, train the model first using the notebook

### Step 6: Run the Server

**Option A - Use the launcher (Recommended):**
```cmd
python run_flask.py
```

**Option B - Run app directly:**
```cmd
python app.py
```

**Option C - Use Flask command:**
```cmd
set FLASK_APP=app.py
flask run
```

### Step 7: Check Server Output

You should see:
```
✅ Agent loaded successfully!
🌐 Starting server at http://127.0.0.1:5000
 * Running on http://127.0.0.1:5000
```

### Step 8: Open Browser

Go to: http://127.0.0.1:5000

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```cmd
pip install flask
```

### Issue: "FileNotFoundError: q_agent.pkl"
**Solution:**
1. Open `Untitled2.ipynb` in Jupyter/VS Code
2. Run the training cells (especially cell 4 or 8)
3. Verify `q_agent.pkl` is created

### Issue: "Port 5000 is already in use"
**Solution:**
Edit `app.py` line with `app.run()` and change:
```python
app.run(host="127.0.0.1", port=5001, ...)  # Changed to 5001
```

### Issue: Terminal closes immediately
**Solution:**
Open CMD, navigate to folder, then run:
```cmd
python run_flask.py
```
Don't double-click the Python file.

### Issue: "templates/index.html not found"
**Solution:**
1. Create `templates` folder
2. Put `index.html` inside it
3. Folder structure should be:
```
reinforecemtn/
├── app.py
├── templates/
│   └── index.html
└── q_agent.pkl
```

---

## ✅ What Success Looks Like

**In Terminal:**
```
✅ Trained agent loaded successfully!
🌐 Starting server at http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

**In Browser:**
- White background with a 3x3 grid
- Two buttons: "Ask Agent for Move" and "Reset Game"
- You can click cells to place X and O

---

## 📞 Still Having Issues?

Run diagnostics:
```cmd
python test_setup.py
```

This will check all files and dependencies.
