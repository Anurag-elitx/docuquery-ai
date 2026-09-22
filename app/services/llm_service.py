from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.config import settings
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question based ONLY on the following context. 
If the answer is not in the context, say 'I don't know'.
Context: {context}
Question: {question}
Answer:"""
        )
        
        # Primary: OpenAI GPT-4o
        self.primary_llm = ChatOpenAI(
            model_name="gpt-4o", 
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
        
        # Fallback: Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.fallback_llm = genai.GenerativeModel('gemini-1.5-pro')

    def generate_answer(self, context: str, question: str) -> str:
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            response = self.primary_llm.predict(prompt)
            return response
        except Exception as e:
            logger.warning(f"OpenAI API failed, falling back to Gemini. Error: {e}")
            try:
                response = self.fallback_llm.generate_content(prompt)
                return response.text
            except Exception as ex:
                logger.error(f"Gemini API also failed. Error: {ex}")
                return "I don't know (API Error or dummy key used without mock)"
