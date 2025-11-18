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


class HierarchicalJudgesCrew:
    """使用层级流程的法官辩论 Crew"""

    def __init__(self):
        config_dir = current_dir / "config"
        self.agents_config = config_dir / "agents.yaml"
        self.tasks_config = config_dir / "tasks.yaml"
        case_path = config_dir / "case_summary.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"文件不存在: {case_path}")

        self.case_text = case_path.read_text(encoding="utf-8")

    def create_manager_agent(self, judge_roles):
        """创建主持人/调度者 Agent"""
        roles_list = "\n".join([f"{role}" for role in judge_roles])

        manager = Agent(
            role="辩论主持人",
            goal="组织法官辩论,根据讨论内容动态决定发言顺序,确保充分讨论",
            backstory=f"""
            我负责主持最高法院的辩论会议。

            可以委派任务的法官有(使用他们的 role):
            {roles_list}

            案情材料:
            {self.case_text}

            我的职责:
            1. 先让每位法官基于案情陈述初步意见
            2. 识别关键分歧点
            3. 根据辩论内容动态安排法官发言(不按固定顺序)
            4. 当某个争议点被充分讨论后,转向下一个
            5. 确保每位法官都有机会充分表达
            6. 最后让书记官总结

            重要: 委派时 coworker 参数必须使用上面列出的完整 role 名称
            """,
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=True
        )
        return manager

    def create_crew(self):
        agents_yaml = _load_yaml(self.agents_config)
        tasks_yaml = _load_yaml(self.tasks_config)

        # 创建所有法官 Agent
        judges = []
        judge_roles = []
        for a in agents_yaml["agents"][:-1]:  # 排除书记官
            # 将 name 和 role 组合，作为可识别的 role
            agent = Agent(
                role=a['role'],  # 使用组合后的 role
                goal=a['goal'],
                backstory=a['backstory'],
                llm=deepseek_llm,
                verbose=True,
                allow_delegation=False
            )
            judges.append(agent)

        # 创建书记官
        clerk_config = agents_yaml["agents"][-1]
        clerk = Agent(
            role=clerk_config['role'],
            goal=clerk_config['goal'],
            backstory=clerk_config['backstory'],
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=False
        )

        # 创建 Manager (传入所有可用的 roles)
        manager = self.create_manager_agent(judge_roles)

        # 定义任务
        tasks = []

        # 任务1: 收集初始意见
        initial_opinions_task = Task(
            description=f"""
## 案情材料

{self.case_text}

## 任务目标

组织以下法官依次发表他们对案件的初步法律意见:

- 首席法官
- 自然法法官
- 内心冲突法官
- 法律实证主义法官
- 实用主义法官

每位法官应该:
- 说明认为被告有罪、无罪或弃权
- 给出100-200字的理由
- 基于各自的法律哲学立场

**重要**: 委派任务时:
1. coworker 参数必须使用上面列出的完整 role 名称
2. 在 context 中包含完整案情材料
3. 明确说明该法官应该做什么
            """,
            expected_output="""
一份包含所有法官初步意见的汇总报告,格式为:
- 法官姓名
- 判决倾向(有罪/无罪/弃权)
- 核心理由(摘要)
            """,
            agent=manager
        )
        tasks.append(initial_opinions_task)

        # 任务2: 组织动态辩论
        debate_task = Task(
            description=f"""
## 案情材料

{self.case_text}

## 任务目标

基于法官们的初步意见,识别主要分歧点,组织针对性辩论。

可委派的法官:
{chr(10).join([f"- {role}" for role in judge_roles[:-1]])}

你需要:
1. 分析哪些观点存在直接冲突
2. **动态决定**让哪些法官针对特定争议点发言(不要按固定顺序)
3. 确保每位法官至少有2-3次深入辩论的机会
4. 当某个论点被充分讨论后,转向下一个争议点
5. 进行2-3轮辩论

注意:
- 委派任务时 coworker 参数必须使用上面列出的完整 role 名称
- 根据讨论内容决定谁应该发言,而非固定顺序
- 鼓励针锋相对的交锋
- 每次委派时都要在 context 中提供案情和之前的讨论内容
            """,
            expected_output="""
一份完整的辩论记录,包括:
- 每轮辩论的主题
- 各法官的发言内容
- 关键论点的交锋过程
            """,
            agent=manager,
            context=[initial_opinions_task]
        )
        tasks.append(debate_task)

        # 任务3: 书记官总结
        summary_task = Task(
            description=f"""
## 案情材料

{self.case_text}

## 任务目标

委派给 Clerk (书记官) 撰写正式的会议纪要。

书记官需要汇总整个辩论过程,包括:
1. 每位法官的最终立场
2. 主要法律争议点
3. 各方论据的核心要点
4. 投票结果统计(有罪/无罪/弃权)
5. 最终裁定结论

**重要**: coworker 参数必须填写 "Clerk"
            """,
            expected_output="""
一份完整的"联邦最高法院会议纪要"文本,
格式正式,逻辑清晰,准确反映辩论全过程。
            """,
            agent=manager,
            context=[initial_opinions_task, debate_task]
        )
        tasks.append(summary_task)

        # 创建 Hierarchical Crew
        # 注意: manager_agent 不应该包含在 agents 列表中
        return Crew(
            agents=[*judges, clerk],  # 只包含工作 agents,不包含 manager
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True
        )


def main():
    """运行层级化辩论"""
    crew_manager = HierarchicalJudgesCrew()
    crew = crew_manager.create_crew()

    print("\n" + "=" * 60)
    print("开始层级化法官辩论")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("辩论结束")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()