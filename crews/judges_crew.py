from crewai import Agent, Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, task, crew
import yaml



@CrewBase
class JudgeCrew:
    "洞穴奇案法官"

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _get_config(self, section, name):
        with open(self.agents_config if section == "agents" else self.tasks_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)[section]
        for item in cfg:
            if item["name"] == name:
                return item
        raise ValueError(f"{name} not found in YAML")

    @agent
    def judges(self):
        agents = []
        agent = Agent(
            config = self._get_config("Tatting J."),
            llm = self.llm,
            verbose=True
        )
        agents.append(agent)
        return agents

    @task
    def discuss_case(self):
        task_cfg = self._get_config("tasks", "法官发表意见")
        return [
            Task(
                config=task_cfg,
                agent=a,
            ) for a in self.judges() if a.name != "Clerk"]

    @task
    def summerize_discussion(self):
        clerk_agent = next(a for a in self.judges() if a.name == "Clerk")
        task_cfg = self._get_config("tasks","书记官总结讨论")
        return Task(
            config=task_cfg,
            agent=clerk_agent
        )

    @crew
    def crew(self):
        """创建一个 Crew"""
        return Crew(
            agents = self.judges(),
            tasks = [self.discuss_case(), self.summerize_discussion()],
            process = Process.sequential,
            verbose = True
        )

