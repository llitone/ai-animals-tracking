import asyncio
import os

from flask import Flask
from database.db_session import global_init


app = Flask(__name__)

if not os.path.isdir("./.database/"):
    os.mkdir("./.database/")
asyncio.run(global_init("./.database/animals.db"))

if __name__ == "__main__":
    app.run()