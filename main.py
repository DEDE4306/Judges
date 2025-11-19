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

        judges_crew = JudgesCrew().create_crew()
        
        # Phase 1: Sequential - judges express initial opinions
        print("\n" + "="*80)
        print("【第一阶段】法官陈述初步观点（按顺序）")
        print("="*80 + "\n")
        phase_1_result = judges_crew["phase_1"].kickoff()
        
        # Phase 2: Hierarchical - manager orchestrates debate
        print("\n" + "="*80)
        print("【第二阶段】经理组织法官辩论")
        print("="*80 + "\n")
        phase_2_result = judges_crew["phase_2"].kickoff()

        return {
            "phase_1_result": phase_1_result,
            "phase_2_result": phase_2_result
        }

if __name__ == "__main__":
    flow = CaveCaseFlow()
    result = flow.kickoff()
