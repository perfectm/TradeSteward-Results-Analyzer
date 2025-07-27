#!/usr/bin/env python3
"""
Test script for forgot password functionality
"""
import requests
import json

def test_forgot_password():
    """Test the forgot password flow for unauthenticated users"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Forgot Password Functionality")
    print("=" * 50)
    
    # Test data
    test_user = {
        "username": "testuser",
        "email": "test@example.com", 
        "password": "testpass123"
    }
    
    new_password = "newpass456"
    
    try:
        # 1. Register a test user (if not already exists)
        print("1. Registering test user...")
        register_data = {
            "username": test_user["username"],
            "email": test_user["email"],
            "password": test_user["password"]
        }
        
        response = requests.post(f"{base_url}/register", data=register_data)
        if response.status_code in [200, 400]:  # 400 if user already exists
            print("✅ Test user ready")
        else:
            print(f"❌ Failed to register user: {response.text}")
            return
        
        # 2. Test forgot password with existing username
        print("2. Testing forgot password with existing username...")
        forgot_data = {
            "username": test_user["username"]
        }
        
        response = requests.post(f"{base_url}/forgot-password", data=forgot_data)
        if response.status_code == 200:
            result = response.json()
            if result.get("success") and result.get("allow_reset"):
                print("✅ Forgot password initiated successfully")
            else:
                print(f"✅ Forgot password response: {result.get('message', 'Unknown')}")
        else:
            print(f"❌ Failed to initiate forgot password: {response.text}")
            return
        
        # 3. Test forgot password with non-existing username
        print("3. Testing forgot password with non-existing username...")
        forgot_data = {
            "username": "nonexistentuser"
        }
        
        response = requests.post(f"{base_url}/forgot-password", data=forgot_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Forgot password with non-existing user handled correctly: {result.get('message', 'Unknown')}")
        else:
            print(f"❌ Failed to handle non-existing user: {response.text}")
        
        # 4. Test updating forgotten password with correct username
        print("4. Testing password update with correct username...")
        update_data = {
            "username": test_user["username"],
            "new_password": new_password
        }
        
        response = requests.post(f"{base_url}/update-forgotten-password", data=update_data)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Password updated successfully")
            else:
                print(f"❌ Password update failed: {result}")
        else:
            print(f"❌ Password update request failed: {response.text}")
            return
        
        # 5. Test login with old password (should fail)
        print("5. Testing login with old password...")
        login_data = {
            "username": test_user["username"],
            "password": test_user["password"]
        }
        
        response = requests.post(f"{base_url}/login", data=login_data)
        if response.status_code == 401:
            print("✅ Correctly rejected old password")
        else:
            print(f"❌ Should have rejected old password: {response.text}")
        
        # 6. Test login with new password (should succeed)
        print("6. Testing login with new password...")
        login_data = {
            "username": test_user["username"],
            "password": new_password
        }
        
        response = requests.post(f"{base_url}/login", data=login_data)
        if response.status_code == 200:
            print("✅ Successfully logged in with new password")
        else:
            print(f"❌ Failed to login with new password: {response.text}")
        
        # 7. Test password update with too short password
        print("7. Testing password update with too short password...")
        update_data = {
            "username": test_user["username"],
            "new_password": "123"  # Too short
        }
        
        response = requests.post(f"{base_url}/update-forgotten-password", data=update_data)
        if response.status_code == 400:
            print("✅ Correctly rejected too short password")
        else:
            print(f"❌ Should have rejected short password: {response.text}")
        
        # 8. Test password update with non-existing username
        print("8. Testing password update with non-existing username...")
        update_data = {
            "username": "nonexistentuser",
            "new_password": "validpassword123"
        }
        
        response = requests.post(f"{base_url}/update-forgotten-password", data=update_data)
        if response.status_code == 404:
            print("✅ Correctly rejected non-existing username")
        else:
            print(f"❌ Should have rejected non-existing username: {response.text}")
        
        print("\n🎉 Forgot password functionality test completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_forgot_password()