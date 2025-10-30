import base64
import io

import gradio as gr
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from multipart import file_path
from zai import ZhipuAiClient
from langchain_openai import ChatOpenAI
from main import get_final_chain, get_config
from config import get_settings
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory

final_chain = get_final_chain()
config = get_config()
settings = get_settings()
'''使用Gradio库创建一个简单的Web界面，允许用户通过文本和语音与聊天机器人进行交互。'''
# TODO: 实现用户登录功能，为每一个用户赋予一个session_id，以便保存和区分不同用户的聊天记录
# TODO：每次用户登录时，将历史记录加载到界面中显示出来

#初始化全模态大模型
llm = ChatOpenAI(
    model=settings.dashscope_model,
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,
)

#初始化prompt
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个多模态AI助手，能够理解用户发送的文本、图片和音频消息，并进行有意义的回复。并根据用户的输入内容，结合上下文信息，生成准确且相关的回答。' ),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | llm

DB_URI = "postgresql://{}:{}@{}:{}/{}?sslmode=disable".format(
settings.postgres_myusername,
settings.postgres_mypassword,
settings.postgres_host,
settings.postgres_port,
settings.postgres_mydatabase,
)

def get_session_history_from_postgres(session_id: str):
    # print(f"🧾 正在加载历史记录，session_id = {session_id}")
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URI
    )

#配置带历史记录的处理链
chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history_from_postgres,
)

#配置config
session_id = "KKZ"
config = {"configurable": {"session_id": session_id}}

# user_msg = HumanMessage([{'type': 'text', 'text': '你知道机器学习是什么东西吗？'}])
# resp = chain_history.invoke({"messages":[user_msg]},config)
# print(resp)






def get_last_user_after_assistant(chat_history):
    """反向便利找到最后一个assistant的位置，并返回后面的所有user消息"""
    if not chat_history:
        return None
    if chat_history[-1]["role"] == "assistant":
        return None
    last_assistant_idx = -1
    for i in range(len(chat_history)-1, -1, -1):
        if chat_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    #若没找到assistant
    if last_assistant_idx == -1:
        return chat_history
    else:
        #从assistant位置向后查找第一个user
        return chat_history[last_assistant_idx+1:]

#web界面中的核心函数
'''
多模态版本 add_message
'''
def add_message(chat_history, user_messages):
    """将用户消息添加到聊天历史记录中"""
    print(user_messages)
    for msg in user_messages['files']:
        print(msg)
        chat_history.append({'role': "user", 'content': {'path': msg}})
    #处理文本消息
    if user_messages['text'] is not None:
        chat_history.append({'role': "user", 'content': user_messages['text']})#字典内的role对应的值必须是"user"或者"assistant"
    return chat_history, gr.MultimodalTextbox(value=None, interactive=False)

#处理语音文件函数
def transcribe_audio(audio_path):
    """
    使用base64处理语音转为
    目前全模态大模型的两种传参方式：1、base64字符串（本地） 2、网络访问的url地址（外网服务器上）如https://xxx.com/xx.wav
    """
    try:
        with open(audio_path,"rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_message = {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_data}",
                "duration": 30 #单位：秒（帮助模型优化处理）
            }
        }
        return audio_message
    except Exception as e:
        return {}

#处理图片文件函数
def transcribe_image(image_path):
    """
    将任意格式的图片转换为base64编码的data URL
    :param image_path: 图片文件的路径
    :return: 包含base64编码的字典
    """
    with Image.open(image_path) as img:
        #获取原始图片格式（如JPEG/PNG）
        img_format = img.format if img.format else "JPEG"
        buffered = io.BytesIO()
        #保留原始格式保存图片到内存缓冲区
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format.lower()};base64,{image_data}",#MIME类型标准
                "detail": "low"
            }
        }

def submit_message(chat_history):
    """提交用户消息并获取聊天机器人的响应"""
    user_messages = get_last_user_after_assistant(chat_history)
    print(user_messages)
    content = [] #HumanMessage的内容
    if user_messages:
        for x in user_messages:
            if isinstance(x['content'], str):#文字输入消息
                content.append({'type': 'text', 'text': x['content']})
            elif isinstance(x['content'], tuple):#多模态输入消息
                file_path = x['content'][0]#取得上传文件的路径
                if file_path.endswith('.wav'):#是录音文件
                    file_message = transcribe_audio(file_path)
                elif file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.jpeg'):#是图片
                   file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                pass
        input_message = HumanMessage(content)
        resp = chain_history.invoke(
            {"messages": input_message},
            config=config
        )
        chat_history.append({'role': "assistant", 'content': resp.content})
    return chat_history


def execute_chain(chat_history):
    """执行聊天链以获取响应"""
    print(chat_history)
    input = chat_history[-1]
    result = final_chain.invoke(
        {"input": input['content'], "config": config},
        config=config
    )
    chat_history.append({'role': "assistant", 'content': result.content})
    return chat_history

with gr.Blocks(title="多模态聊天机器人", theme = gr.themes.Soft()) as block:

    #聊天历史记录的组件
    chatbot = gr.Chatbot(type="messages", height=500, label = "聊天机器人", bubble_full_width=False)

    #创建多模态输入框
    chat_input = gr.MultimodalTextbox(
        interactive=True, #可交互
        file_types=['image', '.wav', '.mp4'], #支持的文件类型
        file_count="multiple",#允许多文件上传
        placeholder="请给ChatBot输入信息或者上传文件...",#输入框体术文本
        show_label=False,
        sources=["microphone", "upload",]#支持的输入源:麦克风与上传文件
    )

    chat_input.submit(
        add_message,
        [chatbot,chat_input],
        [chatbot,chat_input]
    ).then(
        submit_message,
        [chatbot],
        [chatbot],
    ).then( # 回复完成后激活输入框
        lambda: gr.MultimodalTextbox(interactive=True),
        None,#无输入
        [chat_input]#输出到输入框
    )


    if __name__ == "__main__":
        block.launch()