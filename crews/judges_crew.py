import os
import yaml

from pathlib import Path
from typing import List

from crewai import LLM, Agent, Task, Crew, Process

from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.DEBUG)


# 加载环境变量
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
bailian_api_key = os.getenv("BAILIAN_API_KEY")

# 创建 deepseek_llm，用于法官和书记官
deepseek_llm = LLM(
    model="openai/deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key=deepseek_api_key,
    temperature=0.7,
    max_tokens=1024,
    max_retries=5,
    timeout=120,
)

# 创建 bailian_llm，用于辩论主持人
bailian_llm = LLM(
    model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=bailian_api_key,
    temperature=0.3,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.2,
    max_tokens=2048,
    max_retries=5,
    timeout=120,
)

# 当前目录，用于加载 yaml
current_dir = Path(__file__).parent

def _load_yaml(filepath):
    """加载 yaml 工具函数，输入路径"""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


class JudgesCrew:
    def __init__(self):
        """初始化，加载配置文件，创建 agents 和 tasks"""
        config_dir = current_dir / "config"  # config 文件所在目录
        # 获取 config 文件路径
        self.agents_config = config_dir / "agents.yaml"
        self.tasks_config = config_dir / "tasks.yaml"
        # 加载 config 文件
        self.agents_yaml = _load_yaml(self.agents_config)
        self.tasks_yaml = _load_yaml(self.tasks_config)
        # 加载案情描述
        case_path = config_dir / "case_summary.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"文件不存在: {case_path}")
        self.case_text = case_path.read_text(encoding="utf-8")
        # 创建 agents 和 tasks
        self._init_agents()
        self._init_tasks()

    def _init_agents(self):
        """创建所有的 agents"""
        # 全局法官风格
        judge_style_prompt = """
        ## 角色定位
        你是参与合议庭辩论的法官之一，不是主持人或审判长。你的任务是表达自己的法律观点。

        ## 发言风格（强制要求）
        1. **口语化表达**：像在法庭上当面交流，不要写成书面条款或判决书格式
           - ✓ 正确："我认为这个案子的关键在于，威特莫尔最后是否真的同意了抽签..."
           - ✗ 错误："一、案件焦点；二、法律分析；三、结论"

        2. **逻辑表达方式**：
           - 可以用"首先...其次...最后..."来组织论述
           - 禁止使用数字列表（1. 2. 3.）或项目符号（- * •）

        3. **字数限制**：请控制发言长度，每次发言控制在 **400 字** 以内

        4. **禁止行为**：
           - 不要添加动作描述（如"敲击法槌""站起身"）
           - 不要宣布"休庭""开庭""现在进入XX环节"

        5. **语言要求**：
           - 称呼自己和其他法官时使用英文名（如"我认同 Foster 法官的观点""我是法官 Keen.J"）
           - 所有发言内容使用中文
        """

        # 创建法官
        def build_judge(name: str):
            """根据法官名字创建法官"""
            cfg = self.agents_yaml[name]
            full_backstory = f"{cfg['backstory']}\n\n{judge_style_prompt}"
            judge = Agent(
                config=cfg,
                backstory=full_backstory,
                llm=deepseek_llm,
                verbose=True,
                allow_delegation=False,
                max_rpm=5,
                max_iter=7,
            )
            return judge

        # 获取所有法官的姓名
        self.judges_info = list(self.agents_yaml.keys())[:-2]
        # 存储法官列表
        self.judges=[]
        # 创建法官列表
        for judge_name in self.judges_info:
            agent = build_judge(judge_name)
            self.judges.append(agent)

        # 创建书记官
        self.clerk = Agent(
            config=self.agents_yaml["clerk"],
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=False
        )

        # 创建辩论主持人
        manager_config = self.agents_yaml["manager"]

        self.manager = Agent(
            role=manager_config["role"],
            goal=manager_config["goal"],
            backstory=manager_config["backstory"],
            allow_delegation=True,
            llm=bailian_llm,
            verbose=True,
            max_rpm=3,
            max_iter=6,
            respect_context_window=True,
        )

    def _init_tasks(self):
        """创建所有的 task"""
        # ==================== 第一阶段：法官陈述初步观点 ====================
        self.initial_opinion_tasks = []
        opinion_task_config = self.tasks_yaml["collect_initial_opinions"]

        for agent in self.judges:
            full_description = f"""
## 案情材料

{self.case_text}

## 任务要求

{opinion_task_config['description']}

请以法官 {agent.role} 的身份发表你对本案的初步法律意见。

            """
            task = Task(
                config=opinion_task_config,
                description=full_description,
                agent=agent
            )
            self.initial_opinion_tasks.append(task)

        # ==================== 第二阶段：书记官记录法官辩论 ====================
        record_task_config = self.tasks_yaml["record_all_opinions"]
        self.record_opinions_task = Task(
            config=record_task_config,
            agent=self.clerk,
            context = self.initial_opinion_tasks
        )

        # ==================== 第三阶段：主持人组织辩论 ====================
        judge_roles_list = " ".join([f"- {agent.role.lower()}" for agent in self.judges])

        debate_task_config = self.tasks_yaml["organize_debate"]
        debate_desc = debate_task_config["description"].format(
            case_text=self.case_text,
            judge_roles_list=judge_roles_list
        )

        self.organize_debate_task = Task(
            config=debate_task_config,
            description=debate_desc,
            context=[self.record_opinions_task],
        )

        # ================ 注意！！！！！！===================
        # 如果作为 hierarchical task 的 manager 的任务，不能指定任务的 agent 为 manager，否则 manager 无法 delegate 任务

        # ==================== 第四阶段：书记官总结 ====================
        self.clerk_summary_task = Task(
            config=self.tasks_yaml["clerk_summary"],
            context=[*self.initial_opinion_tasks,self.organize_debate_task],
            agent=self.clerk
        )

    def create_crew(self):
        """创建 Crew"""
        # 第一轮：发表观点
        opinion_crew = Crew(
            agents=self.judges,
            tasks=self.initial_opinion_tasks,
            process=Process.sequential,
            verbose=True
        )

        # 第二轮：记录法官发言
        record_crew = Crew(
            agents=[self.clerk],
            tasks=[self.record_opinions_task],
            process=Process.sequential,
            verbose=True
        )

        # 第三轮：辩论
        debate_crew = Crew(
            agents=self.judges,
            tasks=[self.organize_debate_task],
            process=Process.hierarchical,
            manager_agent=self.manager,
            verbose=True,
            memory=False,
        )

        # 第四轮：书记官总结
        summary_crew = Crew(
            agents=[self.clerk],
            tasks=[self.clerk_summary_task],
            process=Process.sequential,
            verbose=True
        )

        # 返回所有的 crew
        return {
            "opinion_crew": opinion_crew,
            "record_crew": record_crew,
            "debate_crew": debate_crew,
            "summary_crew": summary_crew
        }