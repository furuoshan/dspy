from time import perf_counter
import dspy
import agent
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
lm = dspy.LM(model=MODEL, api_key=API_KEY)
dspy.configure(lm=lm)

agent1 = agent.AgentRead()
PROMPT_READ = "尽可能最大化飞机的升阻比，同时保证机翼各项参数正常（比如高度不可能为负，角度不可能为负等)"
print(agent1(PROMPT_READ).fomula)

