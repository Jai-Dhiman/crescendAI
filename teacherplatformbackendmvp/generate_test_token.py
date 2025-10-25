#!/usr/bin/env python3
import jwt
import base64
from datetime import datetime, timedelta

# JWT secret from .env (base64 encoded)
jwt_secret_b64 = "jK2leh7MkEAliiwAwKrZ7v5Mqe1FWMxSCg7jamMH2iAv/oTltHF7T2cWfJAZOHIP/QJ5WwjMEKT8VtiTR47CgA=="
jwt_secret = base64.b64decode(jwt_secret_b64)

# Create claims for test user
now = datetime.utcnow()
claims = {
    "sub": "11111111-1111-1111-1111-111111111111",
    "email": "teacher@test.com",
    "role": "teacher",
    "iat": int(now.timestamp()),
    "exp": int((now + timedelta(hours=1)).timestamp()),
    "iss": "supabase",
    "aud": "authenticated"
}

# Generate token
token = jwt.encode(claims, jwt_secret, algorithm="HS256")
print(f"JWT Token:\n{token}")
