from crewai import Crew
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from helper_crew import KeywordsExtractorAgent, QuestionSetterAgent, KeywordExtractTask, QuestionSetterTask

load_dotenv()

app = FastAPI()

class JDInput(BaseModel):
    description: str

@app.post("/questions")
def generate_blog(input_data: JDInput):
    description = input_data.description

    researcher = KeywordsExtractorAgent()
    writer = QuestionSetterAgent()

    research_task = KeywordExtractTask(agent=researcher)
    write_task = QuestionSetterTask(agent=writer, keyword_extractor_task=research_task)

    crew = Crew(
        tasks=[research_task, write_task],
        agents=[researcher, writer],
        verbose=True
    )

    result = crew.kickoff(inputs={'description': description})

    return {"Questions List": result}
