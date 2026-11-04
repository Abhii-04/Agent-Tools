"""Sandbox for agent which it can use to write commands and work with bash instead of working on local machine. """
from daytona import Daytona, DaytonaConfig
from langchain_daytona import DaytonaSandbox

import os
from dotenv import load_dotenv

load_dotenv()

config = DaytonaConfig(api_key = os.getenv("DAYTONA_API_KEY"))
daytona = Daytona(config)

def sandbox(query):
    sb=None
    try:
        sb= Daytona().create()
        sandbox = DaytonaSandbox(sandbox=sb)
        result = sandbox.execute(query)
        return result
    except Exception as e:
        return f"Sandbox execution failed: {e}"
    finally:
        if sb is not None:
            daytona.delete(sb)