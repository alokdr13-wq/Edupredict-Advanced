import hashlib, hmac, os, re
from database.db import get_user_by_username, get_user_by_id, insert_user, update_password

ITERATIONS = 310_000

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"

def verify_password(password, encoded):
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256": return False
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)).hex()
        return hmac.compare_digest(test, digest_hex)
    except (ValueError, TypeError):
        return False

def ensure_default_admin():
    if get_user_by_username("admin") is None:
        insert_user("EduPredict Administrator","admin@localhost",None,"admin",hash_password("Admin@12345"),"admin")

def authenticate(username, password):
    if not username or not password: return None
    user = get_user_by_username(username)
    if user and verify_password(password, user["password_hash"]):
        user.pop("password_hash", None)
        return user
    return None

def create_account(name,email,roll,username,password,confirm):
    name,email,roll,username = name.strip(),email.strip().lower(),roll.strip(),username.strip()
    if not all([name,email,roll,username,password,confirm]): return False,"All fields are required."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email): return False,"Enter a valid email address."
    if len(username) < 3 or not re.fullmatch(r"[A-Za-z0-9_.-]+",username): return False,"Username must be at least 3 characters and use letters, numbers, dot, underscore or hyphen."
    if len(password) < 10: return False,"Use a password with at least 10 characters."
    if password != confirm: return False,"Passwords do not match."
    try:
        insert_user(name,email,roll,username,hash_password(password),"student")
        return True,"Account created. You can now sign in."
    except Exception as e:
        if "UNIQUE" in str(e): return False,"Email, roll number, or username is already registered."
        return False,"Could not create account. Please check the details."

def change_password(user_id, old, new, confirm):
    user = get_user_by_id(user_id)
    if not user or not verify_password(old,user["password_hash"]): return False,"Current password is incorrect."
    if len(new) < 10: return False,"New password must contain at least 10 characters."
    if new != confirm: return False,"New passwords do not match."
    update_password(user_id,hash_password(new))
    return True,"Password updated successfully."

def logout():
    for key in ("user","last_prediction","nav_target"):
        if key in __import__("streamlit").session_state:
            del __import__("streamlit").session_state[key]
