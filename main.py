from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# from dotenv import load_dotenv
from langchain.prompts import (
  MessagesPlaceholder,
  HumanMessagePromptTemplate,
  ChatMessagePromptTemplate,
  ChatPromptTemplate,
)

# load_dotenv()

chat = ChatOpenAI()

memory = ConversationBufferMemory(
  memory_key="messages",
  return_messages=True,
)

prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}"),
  ],
)

chain = LLMChain(
  llm=chat,
  prompt=prompt,
  memory=memory,
)

while True:
  content = input(">> ")
  result = chain({"content": content})
  print(result["text"])


# def main():
#   print("Hello from tchat!")


# if __name__ == "__main__":
#   main()
