import re
from api.client import get_client
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI

def get_rating_score(context, prompt, rubric, model="gpt-4o", client_category="base"):
    """
    Constructs the messages using the provided rubric and context,
    calls the chat completion API, and extracts a numeric score.
    """
    client = get_client(client_category)
    messages = [
        {"role": "system", "content": rubric},
        {"role": "user", "content": f"The context is: {context} output is: {prompt}."}
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    score_text = response.choices[0].message.content
    match = re.search(r"Score:\s*(\d+)", score_text)
    return int(match.group(1)) if match else -5000

def send_file_request(prompt, instructions, input_path="data/input.txt", output_path="data/outputs.txt", client_category="assistants", model="Assistant"):
    """
    Handles the common file-based flow:
      - Writes the prompt to a file.
      - Creates a vector store and uploads the file.
      - Updates the assistant to include the file-search tool.
      - Creates a thread with the provided instructions.
      - Streams until done and returns the assistant's output.
    """
    client = get_client(client_category)
    assistant = client.beta.assistants.create(
        name="Diagnosis Summarizer",
        instructions=instructions,
        tools=[{"type": "file_search"}],
        model=model
    )
    vector_store = client.beta.vector_stores.create(name="Diagnosis Reports")
    with open(input_path, "w") as f:
        f.write(prompt)
    file_paths = [input_path]
    file_streams = [open(path, "rb") for path in file_paths]
    client.beta.vector_stores.file_batches.upload_and_poll(vector_store_id=vector_store.id, files=file_streams)
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )
    message_file = client.files.create(file=open(input_path, "rb"), purpose="assistants")
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": instructions,
            "attachments": [{"file_id": message_file.id, "tools": [{"type": "file_search"}]}]
        }]
    )
    class EventHandler(client.beta.threads.runs.EventHandler):
        def on_message_done(self, message) -> None:
            message_content = message.content[0].text
            # Optionally process citations here if needed.
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(message_content.value)
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()
    with open(output_path, "r", encoding="utf-8") as d:
        output = d.read()
    return output
