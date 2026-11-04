import os 
from dotenv import load_dotenv
from agentphone import AgentPhone


load_dotenv()


client = AgentPhone(api_key=os.getenv("AGENTPHONE_API_KEY"))

agent = ""