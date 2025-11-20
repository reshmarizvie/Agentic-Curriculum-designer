import os, time, json
from jose import jwt

secret = os.getenv("JWT_SECRET","change_me_super_secret")
alg = os.getenv("JWT_ALG","HS256")
now = int(time.time())
claims = {
  "sub": "local-user",
  "role": "dev",
  "iat": now,
  "exp": now + 24*3600
}
print(jwt.encode(claims, secret, algorithm=alg))
