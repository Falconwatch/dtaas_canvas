from langchain.chat_models import GigaChat
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate


class Giga:
    def __init__(self, prompt, sys_message):
        self._llm = GigaChat(temperature=1e-15, verify_ssl_certs=False, scope='GIGACHAT_API_CORP')
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
        chat_history = messages
        chat_history.append(AIMessage(content=response))

        prompt = """Вот ваша история общения с пользователем:
        {chat_history}
        Ответь навопрос, продолжая диалог:
        """
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompt),
                HumanMessage(content="Теперь опиши основные тренды для компаний аналогичных описанной выше. Обозначь описание трендов под пунктом \"10\"")
            ]
        )
        response2 = self._llm(chat_template.format_messages(chat_history=chat_history)).content
        chat_history.append(chat_template)
        chat_history = [item for sublist in chat_history for item in sublist]

        prompt = """Вот ваша история общения с пользователем:
                {chat_history}
                Ответь навопрос, продолжая диалог:
                """
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompt),
                HumanMessage(
                    content="Теперь предложи стратегию цифровой трансформации для компании аналогичной описанной выше. Начни описание с обозначения параграфа \"11\". Каждый элемент списка элементов стратегии должен начинаться с « - ».")
            ]
        )
        response3 = self._llm(chat_template.format_messages(chat_history=chat_history)).content
        return [response, response2, response3]

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
