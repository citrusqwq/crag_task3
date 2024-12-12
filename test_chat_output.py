from models.rag_llama_baseline_custom import CustomRAGModel

rag_model = CustomRAGModel("llama3-8b-chat")

def gen(query: str) -> str:
    prompt = """You are given a quesition. Your goal is to answer the question in as few words as possible.
### Question
{query}
### Answer
""".format(query=query)
    result = rag_model.generation_pipe(prompt)
    return result
prompt = """who is the american singer-songwriter who has won 14 grammy awards and is known for her unique blend of country, pop, and rock music, including her hit songs "you belong with me" and "we are never ever getting back together"?"""
res = gen(prompt)
print(res)
import pdb
pdb.set_trace()
