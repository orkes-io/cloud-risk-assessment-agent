from typing import Literal
import json

# Langchain imports
from langgraph.types import Command # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore

# Local imports
from src.utils.utils import token_count, read_prompt, read_file_prompt, messages_token_count, load_chat_model, get_latest_human_message, reasoning_prompt, parse_report_command
from src.db.db_query import generate_query, is_valid_query, query_summary
from src.core.agent_state import AgentState
from src.db.db_setup import setup_database_connections
from src.conductor.workers import classify_query, generate_summary, create_insights, write_conclusion, execute_ai_query, explain_yourself


app_context = setup_database_connections()

#-------------------------------
# Model setup
#-------------------------------
model = load_chat_model()
final_model = load_chat_model().with_config(tags=["final_node"])

SYSTEM_PROMPT = read_file_prompt("./src/prompts/report_system_prompt.txt")

#-------------------------------
# Node Functions
#-------------------------------
async def classify_user_intent(state: AgentState) -> Command:
    """
    Classify the user's query as either a report request or a regular question.
    """
    messages = state["messages"]
    query = get_latest_human_message(messages)
    print(f"\n\nUSER QUERY: {query} \n")
    
    [data, destination] = classify_query(query)
    return Command(
        update=data,
        goto=destination
    )
        
async def generate_summary_report(state: AgentState) -> object:
    """Generate a summary report based on the specified category"""
    print("--------------do_summary---------------")
    category = state["category"]

    return generate_summary(category)

async def generate_insights(state: AgentState) -> object:
    """Generate insights based on the top 5 results"""
    print("--------------do_insight---------------")
    result = state["top5"]

    return create_insights(result)

async def finalize_conclusion(state: AgentState) -> object:
    """Generate a conclusion based on the full results"""
    print("--------------do_conclude---------------")
    messages = state["messages"]
    result = state["result_text"]

    return write_conclusion(messages, result)

async def execute_db_query(state: AgentState) -> Command[Literal["reason"]]:
    """
    Execute a database query based on the user's question
    """
    messages = state["messages"]
    user_query = state["user_query"]
    
    # Determine category if available
    category = state.get("category", "ALL").upper() if state.get("category") else "ALL"

    data = execute_ai_query(messages, user_query, category)
    
    return Command(
        update=data,
        goto="reason"
    )

async def provide_explanation(state: AgentState) -> Command:
    """
    Generate an explanation based on query results
    """
    try:
        user_query = state.get("user_query", "")
        sql_query = state.get("sql_query", "")
        query_results = state.get("query_results", "")

        # If the user query is missing, default to the latest human message
        if not user_query:
            user_query = get_latest_human_message(state["messages"])

        return Command(
            update={
                "user_query": None,
                "sql_query": None,
                "query_results": None,
                "messages": state["messages"] + [explain_yourself(user_query, sql_query, query_results)]
            }
        )

    except Exception as e:
        print(f"Error during explanation generation: {e}")
        return Command(
            update={
                "user_query": None,
                "sql_query": None,
                "query_results": None,
                "messages": state["messages"] + [
                    SystemMessage(content="An error occurred while generating the explanation. Please try again.")
                ]
            }
        )