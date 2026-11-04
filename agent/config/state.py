from typing import Annotated, TypedDict,List,Any

from langgraph.graph import add_messages

class State(TypedDict):
    messages:Annotated[List[Any],add_messages]
    next:str
    