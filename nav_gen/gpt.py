# GPT API
import base64
import requests


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_make(prompt_path, ex_prompt):    # set prompt
    with open(prompt_path, "r", encoding='utf-8') as f:  # open the file
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]  # read the user line
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        # print(prompt_system)
        # print(prompt)
        return prompt_system, prompt


def gpt4(args, prompt_path, ex_prompt):   # It is gpt4-o
    # OpenAI API Key
    api_key = args.API_KEY

    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    output = response.json()
    print(output["usage"])
    print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]


def gpt3_5(args, prompt_path, ex_prompt):  # It is gpt4-o-mini
    # OpenAI API Key
    api_key = args.API_KEY

    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    output = response.json()
    print(output["usage"])
    print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]


def gpt4_vision(args, prompt_path, ex_prompt, img_path):  # VLM
    base64_image = encode_image(img_path)
    # OpenAI API Key
    api_key = args.API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system",
             "content": prompt_system
             },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    output = response.json()

    # print(output["usage"])
    # print(output["choices"][0]['message'])
    return output["choices"][0]['message']["content"]
