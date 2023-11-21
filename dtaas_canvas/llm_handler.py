from langchain.chat_models import GigaChat
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate


class Giga:
    def __init__(self, prompt, sys_message):
        self._llm = GigaChat(temperature=1e-15, verify_ssl_certs=False)
        self.prompt = prompt
        self.sys_message = sys_message

    def call(self, message):
        messages = [
            SystemMessage(
                content=self.prompt
            ),
            HumanMessage(
                content=message
            ),
        ]
        response = self._llm(messages).content
        return response


    def get_response(self, message, relevant_docs_k=3):
        messages = [
                SystemMessage(
                    content=(
                        "Заполни Lean Canvas по остервальдеру для компании"
                    )
                ),
                
                HumanMessage(content=message)
            ]
        
        response = self._llm(messages).content
        return response
