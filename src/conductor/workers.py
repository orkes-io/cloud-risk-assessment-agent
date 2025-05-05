import json
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore
from src.utils.utils import read_prompt, read_file_prompt, load_chat_model, reasoning_prompt, parse_report_command
from src.db.db_setup import setup_database_connections
from src.db.db_query import query_summary, is_valid_query, generate_query

#-------------------------------
# Model setup
#-------------------------------
model = load_chat_model()
final_model = load_chat_model().with_config(tags=["final_node"])

SYSTEM_PROMPT = read_file_prompt("./src/prompts/report_system_prompt.txt")

app_context = setup_database_connections()

def classify_query(query: str) -> tuple[object, str]:
    """
    Classify the user's query as either a report request or a regular question.
    """
    try:
        # Try to parse as a report command
        category = parse_report_command(query)
        return [{"category": category}, "summary"]
    except ValueError:
        # Process as a regular question
        content = reasoning_prompt(
            "./src/prompts/intent_classification_prompt.txt", 
            question=query
        )
        intent_response = model.invoke([HumanMessage(content=content)])
        
        try:
            res = json.loads(intent_response.content)
            score = res.get("Score", 0)
            
            if score > 30:
                return [{"intention": res, "user_query": query}, "querydb"]
            else:
                return [{"intention": res, "user_query": None}, "reason"]
            
        except json.JSONDecodeError:
            # Handle invalid JSON response
            print("Failed to parse intent classification response")
            return [{"user_query": query}, "reason"];

def generate_summary(category) -> object:
    """Generate a summary report based on the specified category"""
    print("--------------do_summary---------------")

    # Query database for summary data
    summary_df, details_df = query_summary(app_context.get_connection(), category)
    
    # Convert results to string format
    result = details_df.to_string(index=False)
    top5_result = details_df.to_string()
    summary = summary_df.to_string(index=False)

    # Format prompt for the model
    template = read_prompt("summary")
    prompt = PromptTemplate(
        template=template,
        input_variables=["category", "summary", "result"]
    )
    formatted_prompt = prompt.format(
        category=category, 
        summary=summary, 
        result=result
    )
    
    # Create messages for the model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=formatted_prompt)
    ]

    # Get response from the model
    response = final_model.invoke(messages)

    # Store results in state
    df_str = details_df.to_csv(index=False)
    return {
        "dataframe": df_str, 
        "result_text": result, 
        "top5": top5_result, 
        "messages": [response]
    }

def create_insights(top5: object) -> object:
    """Generate insights based on the top 5 results"""
    print("--------------do_insight---------------")
    
    # Format prompt for insights
    template = read_prompt("insight")
    prompt = PromptTemplate(
        template=template,
        input_variables=["result"]
    )
    formatted_prompt = prompt.format(result=top5)
    
    # Create messages for the model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=formatted_prompt)
    ]
    
    # Get response from the model
    response = final_model.invoke(messages)

    return {"messages": [response]}

def write_conclusion(messages, result) -> object:
    # Add conclusion prompt to messages
    template = read_prompt("conclude")
    messages.append(HumanMessage(content=template))
      
    # Get response from the model
    response = final_model.invoke(messages)
    
    return {"messages": [HumanMessage(content=result), response]}

def execute_ai_query(messages, user_query: str, category: str) -> object:
    try:
        # Generate a database query using the model
        generated_query = generate_query(user_query, category, model)

        # Validate the generated query
        if not is_valid_query(generated_query, app_context.get_engine()):
            print("Generated query is invalid or potentially unsafe.\n\n")
            return {"user_query": user_query}

        # Execute the validated query
        print("Executing query...\n\n")
        cursor = app_context.get_connection().cursor()
        cursor.execute(generated_query)
        records = cursor.fetchall()

        # Prepare query results
        if records:
            columns = [desc[0] for desc in cursor.description]
            results_str = "\n".join(str(dict(zip(columns, row))) for row in records)
        else:
            results_str = "No results returned."

        print("Query results prepared.\n\n")
        return {
                "user_query": user_query, 
                "sql_query": generated_query, 
                "query_results": results_str, 
                "messages": messages + [SystemMessage(content="Query executed successfully.")]
            }

    except Exception as e:
        print(f"Error during query execution: {e}\n\n")
        return {"user_query": user_query}
    
def explain_yourself(user_query: str, sql_query: str = "", query_results: str ="") -> str:
        # Format the explanation prompt
        template = read_prompt("explanation")
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "sql_query", "scan_results"]
        )
        formatted_prompt = prompt.format(
            question=user_query, 
            sql_query=sql_query, 
            scan_results=query_results
        )
        
        # Limit prompt size to prevent token overflow
        if len(formatted_prompt) > 80000:
            formatted_prompt = formatted_prompt[:80000]

        # Get response from the model
        return model.invoke([HumanMessage(content=formatted_prompt)])