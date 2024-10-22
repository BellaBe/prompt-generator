from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph.models import load_model

class PromptGeneratorChain:
    def __init__(self, model_name):
        self.llm = load_model(model_name)
        self.prompt = PromptTemplate.from_template(
            """
            You are a helpful AI assistant specializing in generating prompt templates for AI language models. 
            Your task is to convert user request into a meaningful prompt. 
            
            Target Audience:
            - Large language models
            - AI developers

            Instructions: focus on the following aspects:

            1. Role Specification: Determine the specific role the AI should assume to optimize the prompt's performance. For instance, should the AI act as a tutor, assistant, writer, etc.? Clearly specify the role.

            2. Objective: Understand the main goal or purpose of the prompt. What is the user aiming to achieve with this prompt?

            3. Target Audience: Identify the intended audience for the prompt. Who will be interacting with the output generated by the AI?

            4. Variables: Identify any specific variables that will be incorporated into the prompt template. What inputs should the user provide? Include variable names into curly_brackets.

            5. Instructions: Provide any specific instructions or guidelines for the prompt. What steps should the AI follow to generate the desired output?

            6. Constraints: Clarify any constraints or limitations for the output. What should the output explicitly avoid?

            7. Output requirements: Define any specific requirements or criteria that the output must meet. What are the essential elements?

            Always return a well-structured and clear response. Ensure clarity and consistency.

            Here is the user request: {user_input}
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, input_data):
        return self.chain.invoke(input_data)
