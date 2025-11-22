import pacai.core.agentinfo
import pacai.util.alias

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """
    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_DEFENSIVE)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_OFFENSIVE)

    return [agent1_info, agent2_info]
