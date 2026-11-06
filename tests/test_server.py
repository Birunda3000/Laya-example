"""
Unit and integration tests for server.py (FastAPI application).
"""

import os
import unittest
from starlette.testclient import TestClient

from server import app


class TestServerAPI(unittest.TestCase):

    def test_health_endpoint(self):
        """Verifies GET /health returns 200 OK and valid status."""
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "laya-anonymization-decider")
            self.assertIn("laya_version", data)
            self.assertIn("vegaml_version", data)
            self.assertIn("calibration_loaded", data)

    def test_api_key_auth(self):
        """Verifies 401 Unauthorized when LAYA_API_KEY is configured and missing."""
        original_key = os.environ.get("LAYA_API_KEY")
        try:
            os.environ["LAYA_API_KEY"] = "test-secret-key-xyz"
            with TestClient(app) as client:
                # Without key
                res_unauth = client.post(
                    "/v1/anonymization/decide-column",
                    json={"dataset": {}, "column": {}},
                )
                self.assertEqual(res_unauth.status_code, 401)

                # With incorrect key
                res_wrong = client.post(
                    "/v1/anonymization/decide-column",
                    headers={"X-API-Key": "wrong-key"},
                    json={"dataset": {}, "column": {}},
                )
                self.assertEqual(res_wrong.status_code, 401)
        finally:
            if original_key is not None:
                os.environ["LAYA_API_KEY"] = original_key
            else:
                os.environ.pop("LAYA_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
