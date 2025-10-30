from zai import ZhipuAiClient
from config import get_settings

settings = get_settings()

client = ZhipuAiClient()  # 请填写您自己的 APIKey

input_wav_path = r"C:\Users\KIAEr\AppData\Local\Temp\gradio\ae5687798e0384225b492d8d2beae109da81785132e3d1dc08ca1531cfc6a231\audio.wav"  # 你的 WAV 文件路径

with open(input_wav_path, "rb") as audio_data:
    response = client.audio.transcriptions.create(
    model="glm-asr",
    file=audio_data,
    stream=False
    )

    print(response)
    # for chunk in response:
    #     if chunk.type == "transcript.text.delta":
    #         print(chunk.delta, end="", flush=True)