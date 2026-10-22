import httpx
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

DITTO_USER = os.getenv("USER_DITTO")
DITTO_PASS = os.getenv("PASS_DITTO")
TOKEN_URL = os.getenv("TOKEN_URL")
CLIENT_ID = os.getenv("CLIENT_ID")

class AuthenticationError(Exception):
    """Exception for authentication error"""
    pass


class AuthenticationService:
    def __init__(self):
        self._client = httpx.Client()
        self._auth_data = {
            "grant_type": "password",
            "username": DITTO_USER,
            "password": DITTO_PASS,
            "client_id": CLIENT_ID,
        }

        self._token = None
        self._expires_in = datetime.now()

    def seconds_until_expiration(self) -> float:
        """Returns how many seconds are left until the token actually expires."""
        remaining = self._expires_in - datetime.now()
        return max(0.0, remaining.total_seconds())

    def get_token(self) -> str:
        if self._token is not None and self._expires_in > datetime.now():
            return self._token

        return self.refresh()

    def refresh(self) -> str:
        res = self._client.post(
            TOKEN_URL, data=self._auth_data
        )
        if not res.is_success:
            raise AuthenticationError(
                f"Error refreshing authentication token [{res.status_code}]: {res.content}"
            )
        body = res.json()

        self._token = body["access_token"]
        expires_in = body.get("expires_in", 60)
        self._expires_in = datetime.now() + timedelta(seconds=expires_in // 2)

        return self._token