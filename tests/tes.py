import requests
import uuid
from datetime import datetime

url = "https://api.eu.crosschexcloud.com/"  
request_id = str(uuid.uuid4())
timestamp = datetime.utcnow().isoformat() + "+00:00"


headers = {
    "Content-Type": "application/json"
}
data = {
    "header": {
        "nameSpace": "attendance.record",
        "nameAction": "getrecord",
        "version": "1.0",
        "requestId": request_id,
        "timestamp": timestamp
    },
    "authorize": {
        "type": "token",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb21wYW55X2NvZGUiOiIzMTAwMDUyMTkiLCJob29rc19vbiI6MCwidXJsIjoiIiwic2VjcmV0IjoiIiwiY3JlYXRlX3RpbWUiOiIyMDI1LTA2LTI5IDE2OjA4OjEyIiwidXBkYXRlX3RpbWUiOiIyMDI1LTA2LTI5IDE2OjExOjQ3IiwiYXBpX2tleSI6IjBjYzAwNWI4MDg3NjVlYzBjZjU4ZjhhN2UxNzVhZGE5IiwiYXBpX3NlY3JldCI6IjIxODQ4NmFjOWQ0MzA1NmRkMTc5ZmE3NTY3NjhjMWZjIiwiZXhwIjoxNzUxMjIxMjUxfQ.vu0SJ5ebm3xdSbFX0psZkvulqgAbuCuuqP00CfJtuIw"  # 上一步拿到的 token
    },
    "payload": {
        "begin_time": "2024-06-01T00:00:00+00:00",
        "end_time": "2025-06-30T00:00:00+00:00",
        "page": 1,
        "per_page": 100
    }
}

res = requests.post(url, json=data, headers=headers)
response_json = res.json()
print("✅ Response:", response_json)

# Access records safely
if "payload" in response_json and "list" in response_json["payload"]:
    records = response_json["payload"]["list"]
    print("✅ Records:", records)
else:
    print("❌ No records found in response structure.")
    print("Available keys in payload:", response_json.get("payload", {}).keys())

