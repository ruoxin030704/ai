import openai

# 設置 OpenAI API 密鑰
openai.api_key = 'sk-proj-PcHViqv3eFmSnozPFjhiT3BlbkFJ7sEQktgwIZ3icqGTWGzq'

def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def main():
    print("歡迎來到 GPT-3 聊天機器人！輸入 'exit' 以退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        response = chat_with_gpt(user_input)
        print(f"GPT-3: {response}")

if __name__ == "__main__":
    main()
