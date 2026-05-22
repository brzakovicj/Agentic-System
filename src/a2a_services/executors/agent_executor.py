from src.a2a_services.executors.base_agent_executor import BaseAgentExecutor
from src.agenda_agent.graph import AgendaAgent
from src.documents_agent.graph import DocumentsAgent
from src.scholar_agent.graph import ScholarAgent
from src.study_plan_agent.graph import StudyPlanAgent

class AgendaAgentExecutor(BaseAgentExecutor[AgendaAgent]):
    async def _create_agent(self) -> AgendaAgent:
        agent = AgendaAgent()
        await agent.build_graph()
        return agent
    
class ScholarAgentExecutor(BaseAgentExecutor[ScholarAgent]):
    async def _create_agent(self) -> ScholarAgent:
        agent = ScholarAgent()
        await agent.build_graph()
        return agent

class DocumentsAgentExecutor(BaseAgentExecutor[DocumentsAgent]):
    async def _create_agent(self) -> DocumentsAgent:
        agent = DocumentsAgent()
        await agent.build_graph()
        return agent
    
class StudyPlanAgentExecutor(BaseAgentExecutor[StudyPlanAgent]):
    async def _create_agent(self) -> StudyPlanAgent:
        agent = StudyPlanAgent()
        await agent.build_graph()
        return agent