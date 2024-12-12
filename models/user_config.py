# from models.dummy_model import DummyModel
# from models.rag_llama_baseline import RAGModel
# from models.rag_llama_baseline_custom import CustomRAGModel
# from models.rag_llama_lcw_dev import LcwDevModel
#from models.rag_llama_baseline_batch import BatchRAGModel
#from models.rag_llama_batch_with_calculator import BatchRAGModelWithCalculator
#from models.rag_llama_batch_with_calculator_reasoning import BatchRAGModelWithCalculatorReasoning
#from models.rag_llama_batch_with_calculator_reasoning_optim import OptimBatchRAGModelWithCalculatorReasoning
#from models.model_task_1_add_direct_answer import OptimBatchRAGModelWithCalculatorReasoningDirectAnswer
#from models.model_task_1_paragraph import ParagraphBatchRAGModelWithCalculatorReasoning
#from models.model_task_2 import KGRAGModel
from models.model_task_3_kg import KGRAGModelTask3
#from models.rag_llama_baseline_batch_with_reasoning import BatchReasoningRAGModel
#from models.rag_llama_baseline_batch_custom import BatchRAGModelCustom
#from functools import partial

#from models.model_task_3 import OptimBatchRAGModelWithCalculatorReasoningTask3

# UserModel = BatchRAGModel
# UserModel = BatchRAGModelWithCalculatorReasoning
# UserModel = BatchReasoningRAGModel
# UserModel = BatchRAGModelCustom

# UserModel = OptimBatchRAGModelWithCalculatorReasoningTask3
# UserModel = BatchRAGModelWithCalculator
# UserModel = ParagraphBatchRAGModelWithCalculatorReasoning

UserModel = KGRAGModelTask3

##############################################
# UserModel = partial(CustomRAGModel, model="llama3-8b-groq")
# UserModel = DummyModel
# UserModel = RAGModel
# UserModel = LcwDevModel
# Uncomment the lines below to use the Vanilla LLAMA baseline
# from models.vanilla_llama_baseline import ChatModel
# UserModel = ChatModel


# Uncomment the lines below to use the RAG LLAMA baseline
# from models.rag_llama_baseline import RAGModel
# UserModel = RAGModel
