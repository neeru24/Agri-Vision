"""
Script to create an admin user for Agri-Vision
Run this script to create the first admin user
"""
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import User

def create_admin_user():
    """Create an admin user"""
    with app.app_context():
        # Check if admin already exists
        existing_admin = User.query.filter_by(role='admin').first()
        if existing_admin:
            print(f"Admin user already exists: {existing_admin.email}")
            print("If you want to create another admin, please delete the existing one first.")
            return
        
        # Get admin details
        print("=" * 60)
        print("Create Admin User")
        print("=" * 60)
        
        email = input("Enter admin email: ").strip()
        if not email:
            print("Email is required!")
            return
        
        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(f"User with email {email} already exists!")
            return
        
        full_name = input("Enter admin full name: ").strip()
        if not full_name:
            print("Full name is required!")
            return
        
        password = input("Enter admin password (min 8 characters): ").strip()
        if len(password) < 8:
            print("Password must be at least 8 characters!")
            return
        
        confirm_password = input("Confirm password: ").strip()
        if password != confirm_password:
            print("Passwords do not match!")
            return
        
        # Create admin user
        admin = User(
            email=email,
            full_name=full_name,
            role='admin'
        )
        admin.set_password(password)
        
        db.session.add(admin)
        db.session.commit()
        
        print("\n" + "=" * 60)
        print("Admin user created successfully!")
        print("=" * 60)
        print(f"Email: {email}")
        print(f"Full Name: {full_name}")
        print(f"Role: admin")
        print("\nYou can now login at http://localhost:5000/login")

if __name__ == "__main__":
    create_admin_user()
