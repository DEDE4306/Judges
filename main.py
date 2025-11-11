from typing import List

from crewai.flow.flow import Flow, start
from pydantic import BaseModel
from crews.judges_crew import JudgesCrew

class JudgeOpinion(BaseModel):
    name: str
    opinion: str

class CaseState(BaseModel):
    id: str = "1"
    case: str = "洞穴奇案"
    first_round: List[JudgeOpinion] = []
    second_round: List[JudgeOpinion] = []
    final_summary: str = ""

class CaveCaseFlow(Flow[CaseState]):
    initial_state = CaseState

    @start()
    def begin_discussion(self):

        crew = JudgesCrew().create_crew()
        result = crew.kickoff()

        return result

if __name__ == "__main__":
    flow = CaveCaseFlow()
    result = flow.kickoff()
