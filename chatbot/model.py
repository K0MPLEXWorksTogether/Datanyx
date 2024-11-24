import os
import google.generativeai as genai
import logging

from dotenv import load_dotenv
from chatbot.data_from_api import returnFromApi

class Gemini:

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        try:
            load_dotenv()
            genai.configure(api_key=os.environ["GEMINI"])

            self.model = genai.GenerativeModel(model_name=model)
        except Exception as ConstructorError:
            pass

    def respond(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as ReponseError:
            pass

def main():
    myAi = Gemini()
    data = returnFromApi("2024-10-11", "2024-10-21")
    answer = myAi.respond(
        f"""
        {data[0]}
        This is the predicted profit.
        {data[1]}
        This is the total aggregated revenue for each flower.
        {data[2]}
        This is the total profit for each flower.
        {data[3]}
        This is the  top profit generated for each flower.

        This data pertains to a shop owner in rural India. This person will ask
        questions to you with regards to this data. Continue the conversation.
        """
    )
    print(answer)

if __name__ == "__main__":
    main()