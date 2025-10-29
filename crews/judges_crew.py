import os

import yaml

from pathlib import Path
from crewai import Agent, LLM, Task, Process, Crew
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


current_dir = Path(__file__).parent

def _load_yaml(filepath):
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)

class JudgesCrew:
    """洞穴奇案法官 Crew"""

    def __init__(self):
        config_dir = current_dir / "config"
        self.agents_config = config_dir / "agents.yaml"
        self.tasks_config = config_dir / "tasks.yaml"
        case_path = config_dir / "case_summary.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"文件不存在: {case_path}")

        self.case_text = case_path.read_text(encoding="utf-8")

    def create_crew(self):
        agents_yaml = _load_yaml(self.agents_config)
        tasks_yaml = _load_yaml(self.tasks_config)

        # 创建所有法官 Agent
        agents = []
        for a in agents_yaml["agents"]:
            agent = Agent(
                role=a["role"],
                goal=a["goal"],
                backstory=a["backstory"],
                llm=deepseek_llm,
                verbose=True,
                allow_delegation=False
            )
            agents.append(agent)

        # 创建讨论任务
        tasks = []
        task_1_config = tasks_yaml["tasks"][0]

        # 法官陈述观点
        judge_tasks = []

        for i, agent in enumerate(agents[:-1]):
            # 排除最后的 Clerk

            agent_name = agents_yaml["agents"][i]["name"]
            agent_role = agent.role
            full_description = f"""
                ## 案情材料

                {self.case_text}

                ## 任务要求

                {task_1_config['description']}

                请以 {agent_role}: {agent_name} 的身份发表法律意见，用中文。
            """

            task = Task(
                description=full_description,
                expected_output=task_1_config["expected_output"],
                agent=agent
            )
            judge_tasks.append(task)
            tasks.append(task)


        # 书记官总结任务（依赖所有法官的讨论）
        clerk_agent = agents[-1]
        agent_config = tasks_yaml["tasks"][2]
        clerk_task_1  = Task(
            description=agent_config['description'],
            expected_output=agent_config['expected_output'],
            agent=clerk_agent,
            context=judge_tasks
        )
        tasks.append(clerk_task_1)

        # 第二轮法官互相辩论
        debate_tasks = []

        task_2_config = tasks_yaml["tasks"][1]

        for i, agent in enumerate(agents[:-1]):
            # 排除最后的 Clerk

            agent_name = agents_yaml["agents"][i]["name"]
            agent_role = agent.role
            full_description = f"""
                ## 任务要求
                
                {task_2_config['description']}

                请以 {agent_role}: {agent_name} 的身份发表法律意见，用中文。
            """

            task = Task(
                description=full_description,
                expected_output=task_2_config["expected_output"],
                agent=agent,
                context=judge_tasks
            )
            debate_tasks.append(task)
            tasks.append(task)

        clerk_task_2 = Task(
            description=agent_config['description'],
            expected_output=agent_config['expected_output'],
            agent=clerk_agent,
            context=debate_tasks
        )
        tasks.append(clerk_task_2)

        # 创建 Crew
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
        )
