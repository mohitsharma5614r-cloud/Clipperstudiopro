"""Root FastAPI entrypoint for Clipper Studio.

Allows standard deployment platforms (Railway, Render, Koyeb, Heroku)
and standard FastAPI commands (`fastapi run`, `uvicorn main:app`)
to automatically discover and serve the application.
"""
import os
import uvicorn

from clipper.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)

