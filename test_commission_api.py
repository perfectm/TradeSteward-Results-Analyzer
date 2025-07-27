#!/usr/bin/env python3
"""
Test commission analysis API endpoint
"""

import requests
import json

def test_database_analysis():
    """Test the database analysis endpoint with authentication"""
    
    # First login to get token
    login_url = "http://localhost:8000/login"
    login_data = {
        'username': 'cotton',
        'password': 'your_password_here'  # You'll need to replace this
    }
    
    print("🔐 Testing login...")
    try:
        login_response = requests.post(login_url, data=login_data)
        print(f"Login Status: {login_response.status_code}")
        
        if login_response.status_code == 200:
            login_result = login_response.json()
            token = login_result.get('access_token')
            print(f"✅ Login successful! Token: {token[:50]}...")
            
            # Test database analysis endpoint
            analysis_url = "http://localhost:8000/database-analysis"
            headers = {'Authorization': f'Bearer {token}'}
            
            print("\n📊 Testing database analysis...")
            analysis_response = requests.get(analysis_url, headers=headers)
            print(f"Analysis Status: {analysis_response.status_code}")
            
            if analysis_response.status_code == 200:
                result = analysis_response.json()
                print(f"✅ Analysis successful!")
                
                # Check commission metrics
                metrics = result.get('metrics', {})
                print(f"\n📈 Commission Metrics:")
                commission_keys = ['total_commissions', 'avg_commission_per_trade', 'commission_percentage']
                
                for key in commission_keys:
                    if key in metrics:
                        print(f"  ✅ {key}: {metrics[key]}")
                    else:
                        print(f"  ❌ {key}: MISSING")
                
                # Check if plots include commission analysis
                plots = result.get('plots', [])
                print(f"\n🎨 Generated plots: {len(plots)}")
                for i, plot in enumerate(plots):
                    print(f"  {i+1}. {plot}")
                
                return True
            else:
                print(f"❌ Analysis failed: {analysis_response.text}")
                return False
        else:
            print(f"❌ Login failed: {login_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Commission Analysis API")
    print("=" * 50)
    
    # You may need to update the password
    print("⚠️  Note: You may need to update the password in this script")
    print("    Or create a test user first")
    print()
    
    success = test_database_analysis()
    
    if not success:
        print("\n🔧 Alternative: Test without authentication")
        print("Try accessing: http://localhost:8000/")
        print("And register a new user or check browser console for errors")