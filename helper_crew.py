from crewai import Agent, Task
from crewai.project import agent, task
import os
import yaml

@agent
class KeywordsExtractorAgent(Agent):
    def __init__(self):
        agents_config_file_path = os.path.join(os.path.dirname(__file__), 'config/keyword_extractor_agent.yaml')
        agents_config = yaml.safe_load(open(agents_config_file_path, 'r'))
        super().__init__(config=agents_config["keyword_extractor_agent"])


@agent
class QuestionSetterAgent(Agent):
    def __init__(self):
        agents_config_file_path = os.path.join(os.path.dirname(__file__), 'config/question_setter_agent.yaml')
        agents_config = yaml.safe_load(open(agents_config_file_path, 'r'))
        super().__init__(config=agents_config["question_setter_agent"])

@task
class KeywordExtractTask(Task):
    def __init__(self, agent: KeywordsExtractorAgent):
        agents_config_file_path = os.path.join(os.path.dirname(__file__), 'config/keyword_extractor_task.yaml')
        agents_config = yaml.safe_load(open(agents_config_file_path, 'r'))
        super().__init__(agent=agent, config=agents_config["keyword_extractor_task"])


@task
class QuestionSetterTask(Task):
    def __init__(self, agent: QuestionSetterAgent, keyword_extractor_task: KeywordExtractTask):
        super().__init__(
            agent=agent,
            description="""Analyze the input of technical keywords given and provide interview questions on each of the keyword present in {{keyword_extractor_task.output}}.
                        Provide atleast 7 interview questions.If you need additional information to answer correctly, use available tools to research the topic.""",
            expected_output="A list of interview questions based on each topic in markdown(.md) format.",
            input_tasks=[keyword_extractor_task]
        )
