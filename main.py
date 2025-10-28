from crewai.flow.flow import Flow, start
from pydantic import BaseModel
from crews.judges_crew import JudgesCrew

class CaseState(BaseModel):
    id: str = "1"
    case: str = "洞穴探险者案件"
    discussion_results: str = ""

class CaveCaseFlow(Flow[CaseState]):
    initial_state = CaseState

    @start()
    def begin_discussion(self):

        crew = JudgesCrew().create_crew()
        result = crew.kickoff()

        self.state.discussion_results = str(result)

        return result

if __name__ == "__main__":
    flow = CaveCaseFlow()
    flow.kickoff()