"""Prints the API's OpenAPI schema, which ``web/`` generates its types from.

    python3 -m app.api.openapi > web/openapi.json   # or: cd web && npm run gen:api
"""

import json

from app.api.main import app

if __name__ == "__main__":
    print(json.dumps(app.openapi(), indent=2))
