from autogen import AssistantAgent, config_list_from_json, UserProxyAgent
from autogen.code_utils import DEFAULT_MODEL, UNKNOWN, content_str, execute_code, extract_code, infer_lang
# Tạo một danh sách cấu hình, trong đó bao gồm thông tin API key và endpoint của Gemini
config_list = [
    {
        "model": "gemini-1.5",
        "api_key": "AIzaSyAPwIVqbSYQPXYTLMr14fRvd1gMX6gFcMI",  # Thay thế bằng API key của bạn
        #"base_url": "https://api.example.com/v1",  # Thay thế bằng endpoint của Gemini
        "tags": ["gemini1.5"]
    }
]
config_list_gemini = config_list_from_json(
    "configs/config_list.json",
    filter_dict={
        "model": ["gemini-1.5-pro"],
    },
)
seed = 25
assistant = AssistantAgent(
    "assistant", llm_config={"config_list": config_list_gemini, "seed": seed}, max_consecutive_auto_reply=3
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
)

result = user_proxy.initiate_chat(assistant, message="Sort the array with Bubble Sort: [4, 1, 5, 2, 3]")