import gradio as gr
from langgraph.graph import add_messages
from zai import ZhipuAiClient

from main import get_final_chain, get_config

final_chain = get_final_chain()
config = get_config()
'''使用Gradio库创建一个简单的Web界面，允许用户通过文本和语音与聊天机器人进行交互。'''
# TODO: 优化界面，将语音输入与文字输入结合起来
# TODO: 实现用户登录功能，为每一个用户赋予一个session_id，以便保存和区分不同用户的聊天记录
# TODO：每次用户登录时，将历史记录加载到界面中显示出来

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
def add_message(chat_history, user_messages):
    """将用户消息添加到聊天历史记录中"""
    for msg in user_messages['files']:
        print(msg)
        chat_history.append({'role': "user", 'content': {'path': msg}})
    #处理文本消息
    if user_messages['text'] is not None:
        chat_history.append({'role': "user", 'content': user_messages['text']})#字典内的role对应的值必须是"user"或者"assistant"
    return chat_history, gr.MultimodalTextbox(value=None, interactive=False)
'''

'''无图片模态版本'''
def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({'role': "user", 'content': user_message})
        return chat_history, ''

def read_audio(audio_path):
    """读取音频文件并将其转换为文本"""
    if audio_path:
        client = ZhipuAiClient()
        with open(audio_path, "rb") as audio_data:
            response = client.audio.transcriptions.create(
                model="glm-asr",
                file=audio_data,
                stream=False
            )
        text = response.model_extra['text']
        print(text)
        return text
    return ''

def submit_message(chat_history):
    """提交用户消息并获取聊天机器人的响应"""
    user_messages = get_last_user_after_assistant(chat_history)
    print(user_messages)

def execute_chain(chat_history):
    """执行聊天链以获取响应"""
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

    with gr.Row():
        #文字输入的区域
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder="请给ChatBot发送消息...", label="文字输入", max_lines=5)

            submit_btn = gr.Button("发送",variant="primary")
        #语音输入的区域
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources="microphone", type="filepath", label="语音输入", format="wav")

    #文本框提交的事件
    chat_msg = user_input.submit(add_message, [chatbot, user_input], [chatbot, user_input])
    chat_msg.then(execute_chain,chatbot,chatbot)

    #语音输入框的改变事件
    audio_input.change(read_audio, [audio_input], [user_input])

    #按钮点击的事件
    submit_btn.click(
        add_message,
        [chatbot, user_input],
        [chatbot, user_input]
    ).then(execute_chain,chatbot,chatbot)

    if __name__ == "__main__":
        block.launch()