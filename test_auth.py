#!/usr/bin/env python3
"""
Test script for authentication endpoints
"""

import requests
import json

def test_register():
    """Test user registration"""
    url = "http://localhost:8000/register"
    data = {
        'username': 'testuser',
        'email': 'test@example.com', 
        'password': 'testpass123'
    }
    
    try:
        print("Testing registration endpoint...")
        response = requests.post(url, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registration successful!")
            print(f"User ID: {result.get('user_id')}")
            print(f"Username: {result.get('username')}")
            print(f"Token: {result.get('access_token')[:50]}...")
            return result.get('access_token')
        else:
            print(f"❌ Registration failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during registration: {e}")
        return None

def test_login():
    """Test user login"""
    url = "http://localhost:8000/login"
    data = {
        'username': 'testuser',
        'password': 'testpass123'
    }
    
    try:
        print("\nTesting login endpoint...")
        response = requests.post(url, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Login successful!")
            print(f"User ID: {result.get('user_id')}")
            print(f"Username: {result.get('username')}")
            return result.get('access_token')
        else:
            print(f"❌ Login failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during login: {e}")
        return None

def test_me_endpoint(token):
    """Test the /me endpoint"""
    if not token:
        print("❌ No token available for /me test")
        return
        
    url = "http://localhost:8000/me"
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        print("\nTesting /me endpoint...")
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ /me endpoint successful!")
            print(f"User info: {result}")
        else:
            print(f"❌ /me endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during /me test: {e}")

if __name__ == "__main__":
    print("🧪 Testing TradeSteward Authentication System")
    print("=" * 50)
    
    # Test registration
    token = test_register()
    
    # Test login 
    if not token:
        token = test_login()
    
    # Test /me endpoint
    test_me_endpoint(token)
    
    print("\n" + "=" * 50)
    print("🏁 Authentication tests completed")