from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph.models import load_model

class PromptGraderChain:
    def __init__(self, model_name):
        self.llm = load_model(model_name)
        self.prompt = PromptTemplate.from_template(
            """
            Role: You are an AI language model specialized in evaluating and grading the quality of prompts. 
            Your task is to grade the provided prompt based on the following criteria: role specification, objective, target audience, variables, instructions, constraints, and requirements. Additionally, evaluate the clarity and conciseness of the prompt.

            Evaluation Criteria:
            1. Role Specification: Does the prompt clearly define the role of the AI or the user?
            2. Objective: Is the objective of the prompt clearly stated and logically correct?
            3. Target Audience: Is the target audience specified and appropriate?
            4. Variables: Are all necessary variables included, defined, and logically correct?
            5. Instructions: Are the instructions clear, easy to follow, and logically correct?
            6. Constraints: Are any constraints or limitations clearly outlined and logically correct?
            7. Requirements: Are the requirements for completing the task clearly specified and logically correct?
            8. Clarity: Is the prompt clear and easy to understand?
            9. Conciseness: Is the prompt concise without unnecessary information?

            Here is the prompt to be graded: {initial_prompt}

            Return a binary response (Pass/Fail) for each criterion along with a brief evaluation. Provide an overall response (Pass/Fail) based on the individual criteria evaluations.
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, input_data):
        return self.chain.invoke(input_data)
