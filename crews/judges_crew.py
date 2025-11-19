import os
from pathlib import Path
from typing import List

import yaml
from crewai import LLM, Agent, Task, Crew, Process

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv("DEEPSEEK_API_KEY")

# 大语言模型
deepseek_llm = LLM(
    model="openai/deepseek-ai/DeepSeek-R1",
    base_url="https://api.siliconflow.cn/v1",
    api_key=openai_api_key,
    temperature=0.7,
    max_tokens=8192
)

current_dir = Path(__file__).parent

# 加载 yaml
def _load_yaml(filepath):
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


class JudgesCrew:
    def __init__(self):
        # config 文件所在目录
        config_dir = current_dir / "config"
        # 加载文件
        self.agents_config = config_dir / "agents.yaml"
        self.tasks_config = config_dir / "tasks.yaml"
        # 加载案情描述
        case_path = config_dir / "case_summary.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"文件不存在: {case_path}")
        self.case_text = case_path.read_text(encoding="utf-8")

    def create_crew(self):
        agents_yaml = _load_yaml(self.agents_config)
        tasks_yaml = _load_yaml(self.tasks_config)

        # 创建所有法官 Agent（不包括 Clerk）
        judge_agents = []
        for key in ["Truepenny_CJ", "Foster_J", "Tatting_J", "Keen_J", "Handy_J"]:
            agent = Agent(
                config=agents_yaml[key],
                name=key,
                llm=deepseek_llm,
                verbose=True,
                allow_delegation=False
            )
            judge_agents.append(agent)

        # 创建书记官 Agent
        clerk_agent = Agent(
            config=agents_yaml["clerk"],
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=False
        )

        # 创建 Manager Agent
        manager_agent = Agent(
            config=agents_yaml["manager"],
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=True
        )

        tasks = []

        # ==================== 第一阶段：法官陈述初步观点 ====================
        initial_opinion_tasks = []
        task_1_config = tasks_yaml["collect_initial_opinions"]

        for judge_agent in judge_agents:
            full_description = f"""
## 案情材料

{self.case_text}

## 任务要求

{task_1_config['description']}

请以 {judge_agent.role}: {judge_agent.name} 的身份发表你对本案的初步法律意见。
"""
            task = Task(
                description=full_description,
                expected_output=task_1_config["expected_output"],
                agent=judge_agent
            )
            initial_opinion_tasks.append(task)
            tasks.append(task)


        # ==================== 第二阶段：辩论（Manager 组织，多轮） ====================
        debate_tasks = []
        
        # 第一轮辩论：识别分歧点并启动第一轮讨论
        debate_round_1 = Task(
            description=f"""
## 案情材料

{self.case_text}

## 任务要求

{tasks_yaml['debate_round_1']['description']}

基于法官的初步意见，请开始辩论。
""",
            expected_output=tasks_yaml['debate_round_1']['expected_output'],
            agent=manager_agent,
            context=initial_opinion_tasks
        )
        debate_tasks.append(debate_round_1)


        # ==================== 创建 Crew ====================
        # 第一阶段：sequential（法官们按顺序发表观点）
        phase_1_crew = Crew(
            agents=judge_agents,
            tasks=initial_opinion_tasks,
            process=Process.sequential,
            verbose=True
        )

        # 第二阶段：sequential + hierarchical（manager 多轮协调辩论）
        phase_2_crew = Crew(
            agents=[*judge_agents,clerk_agent],
            tasks=debate_tasks,
            process=Process.sequential,
            verbose=True
        )

        # 返回一个包含两个 crew 的对象，以便在 Flow 中按顺序执行
        return {
            "phase_1": phase_1_crew,
            "phase_2": phase_2_crew,
            "initial_opinion_tasks": initial_opinion_tasks,
            "debate_tasks": debate_tasks
        }

