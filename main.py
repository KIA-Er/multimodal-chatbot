from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from config import get_settings

setting = get_settings()

#提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手：小秘。京你所能回答所有问题。提供的聊天里是包含与你对话用户的相关信息。'),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ('human', '{input}')
])

#实例化大模型对象
llm = ChatOpenAI(
    model = setting.openai_model,
    streaming=True,
)

chain = prompt | llm

#存储聊天记录：（内存，关系型数据库或者redis数据库、Postgresql）

store = {} #用來保留所有所有历史消息，key: 会话ID session_id


def get_session_history(session_id: str):
    """从内存中的历史消息列表中，回复当前会话的所有历史消息"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

#实现Postgresql链接以持久化存储
DB_URI = "postgresql://{}:{}@{}:{}/{}?sslmode=disable".format(
setting.postgres_myusername,
setting.postgres_mypassword,
setting.postgres_host,
setting.postgres_port,
setting.postgres_mydatabase,
)
def get_session_history_from_postgres(session_id: str):
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URI
    )

#langchain中所有消息类型：SystemMessage, HumanMessage, AIMessage, ToolMessage

#创建带历史记录的处理链

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history= get_session_history_from_postgres,
    input_messages_key="input",
    history_messages_key="chat_history",
)

#配置文件，使大模型识别会话id
session_id = "KIAEr_1"
config = {"configurable": {"session_id": session_id}}

result1 = chain_with_message_history.invoke({"input":"你好，我的名字叫张文凯"}, config=config)
print(result1)

result2 = chain_with_message_history.invoke({"input":"那你的呢？"}, config=config)
print(result2)