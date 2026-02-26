from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from vectorstore import retriever


llm = ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0.2)

template = """
You are a expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

while True:
    user_input = input("\nEnter your question: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    reviews = retriever.invoke(user_input)

    response = chain.invoke({"reviews": reviews, "question": user_input})
    print("\n✅✅Answer:", response.content)
