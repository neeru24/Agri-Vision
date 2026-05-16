# Security Fix - Issue #33

## What was the problem?

This Flask app had two security issues:

### 1. SECRET_KEY was not safe
- It was using a default fallback value: `"default-secret-key"`
- This is unsafe because it can allow session attacks

### Fix:
- Now the app will only work if `SECRET_KEY` is set in environment
- No default key is used

```python
secret_key = os.getenv("SECRET_KEY")

if not secret_key:
    raise ValueError("SECRET_KEY is required. Please set it in .env file.")

app.secret_key = secret_key

2. Unsafe debug mode

Debug mode was previously always enabled or hardcoded, which is not safe for production.

Fix:

Debug mode is now controlled using environment variable FLASK_DEBUG and is disabled by default.

is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

app.run(host="0.0.0.0", debug=is_debug)

###. How to run
Create .env file:

SECRET_KEY=your_secret_key
FLASK_DEBUG=False

Run the app:
python app.py

###. Security Improvements
No hardcoded secret keys
Debug mode disabled by default
Safer production configuration