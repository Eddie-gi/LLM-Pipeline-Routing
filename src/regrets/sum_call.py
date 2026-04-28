from api.client import get_client
from utils.helper import send_file_request

def get_summary(prompt, model_name):
    if model_name == "assistant":
        instructions = ("You are to summarize inputted medical reports from patients and include the most important information useful for diagnosis.")
        summary = send_file_request(prompt, instructions)
        return summary
    else:
        client = get_client("base")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content