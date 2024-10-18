import os
import random
import re
import time
import json
import argparse
from functools import wraps

import os.path as osp
from tqdm import tqdm

from autogen import (
    GroupChat,
    UserProxyAgent,
    ConversableAgent,
    AssistantAgent,
    GroupChatManager,
    config_list_from_json,
)

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Medagents Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_list.json",
        help="the llm models' config file",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default= "gemini_1.5_flash",#"llama-3.1-70b-versatile", 
        choices=  ["gemini_1.5_flash002", "gemini_1.5_flash"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=  "gemini_1.5_flash",#"llama-3.1-70b-versatile", 
        choices= ["gemini_1.5_flash002", "gemini_1.5_flash"], 
        help="the llm models",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="inital",
        choices=["inital", "follow_up"],
        help="choice different stages",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="choice different stages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_user",
        help="log file",
    )
    parser.add_argument(
        "--num_specialists", type=int, default=3, help="number of experts"
    )
    parser.add_argument("--n_round", type=int, default=13, help="attempt_vote")
    parser.add_argument("--query_round", type=int, default=1, help="query times")

    args = parser.parse_args()

    return args

def process_single_case(
    args, user_input,output_dir, model_config, query_model_config
):
    case_info = {}
    case_crl = random.randint(1000, 9999)
 
    json_name = f"{case_crl}.json"
    conversation_name = f"{case_crl}_conversation.json"
    identify = f"{args.num_specialists}-{args.n_round}"

    output_dir = osp.join(
        output_dir,
        "MAC_WS_user",
        args.stage,
        args.model_name,
        identify,
        str(args.times),
    )

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    file_names = os.listdir(output_dir)

    json_files = [file for file in file_names if file.endswith(".json")]

    if json_name in json_files and conversation_name in json_files:
        return

    # user_proxy_agent = UserProxyAgent(  name="user", 
    #                                     human_input_mode="ALWAYS",
    #                                     is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    #                                     code_execution_config={                                     
    #                                     "use_docker": False,#Set to True if you want to use docker 
    #                                   })
    
    case_presentation = user_input
       
    coordinator = ConversableAgent(
        "Medical_Coordinator",
        system_message="You are a Medical Coordinator. Your role is to provide the patient's medical history and ask questions to determine the appropriate specialist. You should seek clarification and ensure all relevant information is covered.",
        llm_config=query_model_config,
        human_input_mode="NEVER",  # Never ask for human input.
    )

    consultant = ConversableAgent(
        "Senior_Medical_Consultant",
        system_message="You are a Senior Medical Consultant. Your role is to answer the Medical Coordinator's questions, recommend the appropriate specialist based on the medical history provided, and correct any misconceptions.",
        llm_config=query_model_config,
        human_input_mode="NEVER",  # Never ask for human input.
    )

    consultant_message = get_consultant_message(case_presentation, int(args.num_specialists))

    result = coordinator.initiate_chat(
        consultant, message=consultant_message, max_turns=args.query_round
    )
    top_k_specialists = prase_json(result.chat_history[-1]["content"])[
        "top_k_specialists"
    ]
    assert len(top_k_specialists) == args.num_specialists
    
    Docs = []
    for specialist in top_k_specialists:
        name = specialist.replace(" ", "_")
        doc_system_message = get_doc_system_message(
            doctor_name=name, stage=args.stage)

        Doc = AssistantAgent(
            name=name,
            llm_config=model_config,
            system_message=doc_system_message,
        )
        Docs.append(Doc)


    supervisor_system_message = get_supervisor_system_message(
        stage=args.stage, use_specialist=True, specialists=top_k_specialists
    )

    Supervisor = AssistantAgent(
        name="Supervisor",
        llm_config=model_config,
        system_message=supervisor_system_message,
    )

    agents = Docs + [Supervisor]
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=args.n_round,
        speaker_selection_method="auto",  
        admin_name="Critic",
        select_speaker_auto_verbose=False,
        allow_repeat_speaker=True,
        send_introductions=False,
        max_retries_for_selecting_speaker=args.n_round // (1 + args.num_specialists),
    )
    time.sleep(5)
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=model_config,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    inital_message = get_inital_message(patient_history=case_presentation, stage=args.stage)

    output = Supervisor.initiate_chat(
        manager,
        message=inital_message,
    )

    conversation_path = osp.join(output_dir, conversation_name)
    with open(conversation_path, "w",  encoding="utf-8") as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)

    critic_output = [
        item
        for i, item in enumerate(output.chat_history)
        if item.get("name") == None
        and '"Most Likely Diagnosis":' in item.get("content")
    ]

    syn_report = critic_output[-1]["content"]

    json_output = prase_json(syn_report)

    case_info["Crl"] = case_crl
    case_info["Presentation"] = case_presentation
    case_info["Most Likely"] = json_output.get("Most Likely Diagnosis")
    case_info["Other Possible"] = json_output.get("Differential") or json_output.get(
        "Differential Diagnosis"
    )

    if args.stage == "inital":
        case_info["Recommend Tests"] = json_output.get(
            "Recommend Tests"
        ) or json_output.get("Recommended Tests")

    recorder_path = osp.join(output_dir, json_name)
    with open(recorder_path, "w",  encoding="utf-8") as file:
        json.dump(case_info, file, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    
    query_filter_criteria = {
        "tags": [args.query_model_name],
    }

    filter_criteria = {
        "tags": [args.model_name],
    }

    query_config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=query_filter_criteria
    )

    config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria
    )

    query_model_config = {
        "cache_seed": None,
        "temperature": 0,
        "config_list": query_config_list,
        "timeout": 120,
    }

    model_config = {
        "cache_seed": None,
        "temperature": 1,
        "config_list": config_list,
        "timeout": 300,
    }

    output_dir = args.output_dir
    

    # Gọi hàm để nhận đầu vào từ người dùng
    user_input = '''Tôi cảm thấy mình thường xuyên:
                Hay quên: Tôi quên các việc quan trọng như cuộc hẹn, công việc cần làm, và thậm chí là những việc vừa mới xảy ra. Ví dụ, tôi có thể vào bếp lấy nước uống nhưng lại quên lý do tại sao mình vào đó.
                Khó tập trung: Tôi gặp khó khăn trong việc duy trì sự tập trung vào một việc trong thời gian dài, đặc biệt là khi công việc đòi hỏi sự chú ý cao. Tôi thường bắt đầu một việc nhưng dễ bị phân tâm và chuyển sang làm việc khác trước khi hoàn thành.
                Luôn cảm thấy bồn chồn: Tôi thường cảm thấy không thể ngồi yên một chỗ quá lâu, tôi có thói quen rung chân, hoặc luôn phải làm gì đó, ví dụ như chơi với bút hay điện thoại dù đang làm việc gì khác.
                Bị trì hoãn: Tôi có xu hướng trì hoãn các công việc quan trọng cho đến phút cuối cùng, dù biết rằng mình có thời gian để hoàn thành chúng sớm hơn.
                Gặp khó khăn trong việc tổ chức: Tôi khó sắp xếp công việc và thời gian của mình một cách có tổ chức. Ví dụ, khi làm việc nhóm, tôi dễ bị lạc hướng và không biết làm thế nào để bắt đầu hay phân chia thời gian hợp lý."
    '''
    try:
        process_single_case(
            args, user_input,output_dir, model_config, query_model_config
        )
    except Exception as e:
        print(f"Failed to process case after all attempts: {str(e)}")
        

if __name__ == "__main__":
    main()