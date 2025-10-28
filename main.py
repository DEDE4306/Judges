import os

from crewai import Crew, Agent, Task, LLM
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
    deepseek_llm = LLM(
        model="openai/deepseek-ai/DeepSeek-R1",
        base_url="https://api.siliconflow.cn/v1",
        api_key=openai_api_key,
        temperature=0.7,
        max_tokens=8192
    )

    print("=== 加载配置文件 ===")
    with open("crews/config/agents.yaml", encoding="utf-8") as f:
        agents_yaml = yaml.safe_load(f)

    with open("crews/config/tasks.yaml", encoding="utf-8") as f:
        tasks_yaml = yaml.safe_load(f)

    # 读取案情文本
    case_text = Path("crews/config/case_summary.txt").read_text(encoding="utf-8")
    print(f"案情文本长度: {len(case_text)} 字符\n")

    # 创建 Agents
    print("=== 创建 Agents ===")
    agents = {}
    for a in agents_yaml["agents"]:
        agent_name = a["name"]
        agents[agent_name] = Agent(
            role=a["role"],
            goal=a["goal"],
            backstory=a["backstory"],
            llm=deepseek_llm,
            verbose=True,
            allow_delegation=False
        )
        print(f"✅ {agent_name}")

    # 创建 Tasks
    print(f"\n=== 创建 Tasks ===")
    tasks = []
    task_config = tasks_yaml["tasks"][0]

    for agent_name in task_config["agents"]:
        if agent_name not in agents:
            print(f"❌ Agent '{agent_name}' 不存在")
            continue

        # 组合案情和任务描述
        full_description = f"""
## 案情材料

{case_text}

## 任务要求

{task_config['description']}

请以 {agent_name} 的身份发表法律意见。
"""

        # 创建任务
        task = Task(
            description=full_description,
            expected_output=task_config["expected_output"],
            agent=agents[agent_name]
        )
        tasks.append(task)
        print(f"✅ 为 {agent_name} 创建任务")

    # 创建 Crew
    print(f"\n=== 创建 Crew (共 {len(agents)} 个 Agent, {len(tasks)} 个 Task) ===\n")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=True,
    )

    # 执行
    print("=== 开始执行 ===\n")
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("执行完成")
    print("=" * 60 + "\n")
    print(result)

    return result


if __name__ == "__main__":
    main()

