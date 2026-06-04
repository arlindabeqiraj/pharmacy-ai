"""
Test Databricks connection for Pharmacy AI
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

host  = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")

print("Testing Databricks connection...")
print(f"Host: {host}")

headers = {"Authorization": f"Bearer {token}"}

# Test me SCIM API
r = requests.get(f"{host}/api/2.0/clusters/list", headers=headers)

print(f"Status code: {r.status_code}")

if r.status_code == 200:
    print("✅ Databricks connected successfully!")
    data = r.json()
    clusters = data.get("clusters", [])
    print(f"   Clusters: {len(clusters)}")
elif r.status_code == 403:
    print("❌ Token invalid — gjenero token të ri")
elif r.status_code == 401:
    print("❌ Unauthorized — kontrollo DATABRICKS_HOST dhe TOKEN")
else:
    print(f"❌ Error: {r.text[:200]}")