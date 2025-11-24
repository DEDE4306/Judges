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

class CaveCaseFlow(Flow[CaseState]):
    """创建辩论流程 flow"""
    initial_state = CaseState # 创建 State

    @start()
    def begin_discussion(self):
        # 创建 crews
        crews = JudgesCrew().create_crew()
        
        # 第一轮：法官陈述初步观点
        opinion_result = crews["opinion_crew"].kickoff()

        # 第二轮：书记官记录法官观点
        record_result = crews["record_crew"].kickoff()
        
        # 第三轮：主持人组织辩论
        debate_result = crews["debate_crew"].kickoff()

        # 第四轮：书记官总结会议
        summary_result = crews["summary_crew"].kickoff()

        return {
            "initial_opinions": opinion_result,
            "record": record_result,
            "debate": debate_result,
            "summary": summary_result,
        }


if __name__ == "__main__":
    flow = CaveCaseFlow()
    result = flow.kickoff()
