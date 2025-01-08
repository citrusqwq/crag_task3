import os
from utils.cragapi_wrapper import CRAG

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "https://demo3.kbs.uni-hannover.de")

print("API server: ", CRAG_MOCK_API_URL)
api = CRAG(server=CRAG_MOCK_API_URL)
print(api.open_search_entity_by_name("florida"))
