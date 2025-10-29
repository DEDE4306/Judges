from typing import List

from crewai.flow.flow import Flow, start
from pydantic import BaseModel
from crews.judges_crew import JudgesCrew

class JudgeOpinion(BaseModel):
    name: str
    opinion: str

class CaseState(BaseModel):
    id: str = "1"
    case: str = "洞穴探险者案件"
    first_round: List[JudgeOpinion] = []
    second_round: List[JudgeOpinion] = []
    final_summary: str = ""

class CaveCaseFlow(Flow[CaseState]):
    initial_state = CaseState

    @start()
    def begin_discussion(self):

        crew = JudgesCrew().create_crew()
        result = crew.kickoff()

        self.state.first_round = result.tasks_output[0].raw_output  # 第一个任务输出
        self.state.second_round = result.tasks_output[1].raw_output
        self.state.final_summary = result.tasks_output[-1].raw_output

        return result

if __name__ == "__main__":
    flow = CaveCaseFlow()
    flow.kickoff()
    print("第一轮法官意见：", flow.state.first_round)
    print("第二轮辩论意见：", flow.state.second_round)
    print("最终汇总：", flow.state.final_summary)