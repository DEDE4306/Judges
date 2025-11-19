import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("DEEPSEEK_API_KEY")

llm = LLM(
    model="openai/deepseek-ai/DeepSeek-R1",
    base_url="https://api.siliconflow.cn/v1",
    api_key=openai_api_key,
    temperature=0.7,
    max_tokens=8192
)

# 定义你的 Agent
researcher = Agent(
    role="研究员",
    goal="对人工智能及其智能代理进行深入研究与分析",
    backstory="你是一名资深研究人员，专长于科技、软件工程、人工智能和初创领域。目前你是一名自由职业者，正在为一位新客户进行研究。",
    allow_delegation=False,
    llm=llm,
)

writer = Agent(
    role="高级写作者",
    goal="撰写关于人工智能及智能代理的高质量内容",
    backstory="你是一名技术类高级写作者，擅长软件工程、人工智能以及初创企业方向。目前你作为自由职业者为一位新客户撰写内容。",
    allow_delegation=False,
    llm=llm,
)

# 定义任务
task = Task(
    description="生成 5 个关于文章写作的有趣主题，并为每个主题撰写一段吸引人的短文，用以展示该主题发展为完整文章的潜力。请返回包含主题及段落的列表，并附上你的备注。",
    expected_output="5 个要点，每个要点包含对应段落及备注。",
)

# 定义管理者 Agent
manager = Agent(
    role="项目经理",
    goal="高效管理整个团队，确保任务高质量完成",
    backstory="你是一名经验丰富的项目经理，擅长协调复杂项目并引导团队达成目标。你负责统筹团队成员的工作，确保任务按时且高质量完成。",
    allow_delegation=True,
    llm=llm,
)

# 初始化 Crew，使用自定义的 manager
crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
    verbose=True,
)

# 启动团队工作
result = crew.kickoff()
