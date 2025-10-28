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
#√已完成:切分聊天上下文，形成摘要记忆以节省token
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history= get_session_history_from_postgres,
    input_messages_key="input",
    history_messages_key="chat_history",
)

#剪辑和摘要上下文历史记录：最近前k条数据，把之前的消息形成摘要
def summarize_messages(current_input, k: int =2):
    """剪辑和摘要上下文，历史记录"""
    session_id = current_input['config']['configurable']['session_id']
    if not session_id:
        raise ValueError("必须通过config参数提供session_id")
    
    #获取当前会话id的历史聊天记录
    chat_history = get_session_history_from_postgres(session_id)#返回的类型是：PostgresChatMessageHistory对象
    stored_messages = chat_history.messages#通过history对象调取messages
    if len(stored_messages)<=k:#保留最近k条历史记录
        return False
    
    #剪辑消息列表
    last_k_messages = stored_messages[-k:]#保留的k条消息
    messages_to_summarize = stored_messages[:-k]#需要总结的消息

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将下列历史对话压缩为一条保留关键消息的摘要信息，不丢失信息密度"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","请生成包含上述对话核心内容的摘要，保留重要事实和决策。")
    ])
    summarization_chain = summarization_prompt | llm
    #生成摘要
    summary_message = summarization_chain.invoke({"chat_history": messages_to_summarize})

    #重建历史记录：摘要+最后k条消息
    #FIXME：能否不影响数据库中存储的历史记录？
    chat_history.clear()
    chat_history.add_message(summary_message)
    for msg in last_k_messages:
        chat_history.add_message(msg)
    return True

#最终的链,使用RunnablePassthrough方法，默认将输入数据原样传递到下游，而.assign()方法允许在保留原始输入的同时，通过指定键对（message_summarized=summarization）将Dict中新加一个字段
from langchain_core.runnables import RunnablePassthrough
final_chain = RunnablePassthrough.assign(messages_summaried = summarize_messages) | chain_with_message_history

#配置文件，使大模型识别会话id
session_id = "KIAEr_1"
config = {"configurable": {"session_id": session_id}}

# result1 = final_chain.invoke({"input":"你好，我的名字叫张文凯", "config":config}, config=config)
# print(result1)

# result2 = final_chain.invoke({"input":"那你的呢？", "config":config}, config=config)
# print(result2)
result3 = final_chain.invoke({"input":"请你以我的口吻给未来的师弟介绍一下这个岗位的消息。", "config":config}, config=config)
print(result3)