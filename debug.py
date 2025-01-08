from models.attr_predictor import predict_static_or_dynamic
from openai import OpenAI
import os

llm = OpenAI(
    base_url=os.getenv("INTERWEB_HOST", "https://interweb.l3s.uni-hannover.de"),
    api_key=os.getenv("INTERWEB_APIKEY"),
)


# Sample query batch
batch = {
    "query": [
        "What is the population of Japan?",
        "What time is the next train to New York?",
        "Who is the President of the United States?",
        "When will the next solar eclipse occur?",
    ]
}

# Few-shot examples (optional for your task)
few_shots = [
    {"query": "What is the capital of France?", "static_or_dynamic": "static"},
    {"query": "What is the weather in Berlin today?", "static_or_dynamic": "dynamic"},
]

# Call the predict_static_or_dynamic method
answers = predict_static_or_dynamic(
    llm=llm, batch=batch, sample_num=1, few_shots=few_shots
)

# Print results
print(answers)


# List available models
models = llm.models.list()
print(models)


'''
# Define the test case
test_batch = {
    "query": ["What is the population of Paris?", "Who won the 2024 Olympics?"]
}

# Call predict_static_or_dynamic with only one query
test_few_shots = [
    {"query": "What is the capital of France?", "static_or_dynamic": "static"},
    {"query": "What is the weather in Paris today?", "static_or_dynamic": "dynamic"}
]

# Construct a single test query
system_prompt = """You will be provided with a question. Your task is to identify whether this question is a static question or a dynamic question. A static question is that the answer is fixed and will not change over time. A dynamic question is that the answer will change over time or needs time information. You **MUST** choose from one of the following choices: ["static", "dynamic"]. You **MUST** give the question type succinctly, using the fewest words possible."""

for example in test_few_shots:
    system_prompt += (
        "------\n### Question: {}\n### Static or Dynamic: {}\n\n".format(
            example["query"], example["static_or_dynamic"]
        )
    )

query = "What is the weather in Paris today?"  # Example query to test
user_prompt = f"""Here is the question: {query}\nRemember your rule: You **MUST** choose from the following choices: ["static", "dynamic"].\nWhat is the static or dynamic of this question?"""

prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# Send the request
response = llm.chat.completions.create(
    model="llama3.3:70b",  # Adjust to the appropriate model you are using
    messages=prompt,
    n=1,
    temperature=0.1,
    top_p=0.9,
    max_tokens=10
)

# Iterate through all completions
for i, choice in enumerate(response.choices):
    print(f"Response {i+1}: {choice.message.content}")
'''
