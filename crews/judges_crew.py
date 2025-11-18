import os
from pathlib import Path
from typing import List

from crewai import LLM, Agent, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("DEEPSEEK_API_KEY")

deepseek_llm = LLM(
    model="openai/deepseek-ai/DeepSeek-R1",
    base_url="https://api.siliconflow.cn/v1",
    api_key=openai_api_key,
    temperature=0.7,
    max_tokens=8192
)

@CrewBase
class JudgesCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        super().__init__()
        # 加载案情材料
        case_path = Path(__file__).parent / "config" / "case_summary.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"文件不存在: {case_path}")
        self.case_text = case_path.read_text(encoding="utf-8")

    @agent
    def truepenny(self) -> Agent:
        return Agent(config=self.agents_config["Truepenny C.J."])

    @agent
    def foster(self) -> Agent:
        return Agent(config=self.agents_config["Foster J."])

    @agent
    def tatting(self) -> Agent:
        return Agent(config=self.agents_config["Tatting J."])

    @agent
    def keen(self) -> Agent:
        return Agent(config=self.agents_config["Keen J."])

    @agent
    def handy(self) -> Agent:
        return Agent(config=self.agents_config["Handy J."])

    @agent
    def clerk(self) -> Agent:
        return Agent(config=self.agents_config["Clerk"])

    @agent
    def manager(self) -> Agent: