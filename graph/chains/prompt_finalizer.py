from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph.models import load_model

class PromptFinalizerChain:
    def __init__(self, model_name):
        self.llm = load_model(model_name)
        self.prompt = PromptTemplate.from_template(
            """
            Role: You are an AI language model specialized in refining and finalizing prompts. 
            Your task is to review the graded prompt and make necessary revisions to ensure it meets all criteria. After making the revisions, generate the final version of the prompt.

            Target Audience:
            - Large language models
            - AI developers

            Instructions: Follow these step-by-step instructions to finalize the prompt:
            1. Review the evaluations from the prompt grader. Note any criteria that did not pass.
            2. Make necessary revisions to address any issues identified in the grading process.
            3. Ensure all parts of the prompt are logically correct, clear, and concise.
            4. Ensure the prompt includes:
               - Role Specification
               - Objective
               - Target Audience
               - Variables
               - Instructions
               - Constraints
               - Output Requirements
            5. Generate the final version of the prompt.

            Here is the graded prompt: {initial_prompt}
            Here are the evaluations: {evaluations}

            Return the finalized prompt.
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, input_data):
        return self.chain.invoke(input_data)
