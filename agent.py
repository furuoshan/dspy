"""
基于design提供的设计规范实现智能体和智能体系统
"""
import dspy
from design import Read

class AgentRead(dspy.Module):
    def __init__(self):
        super().__init__()
        self.read = dspy.Predict(Read)

    def forward(self, natural_decs: str):
        return self.read(requirements=natural_decs)
