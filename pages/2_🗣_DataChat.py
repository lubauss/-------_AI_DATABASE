from tiktoken import get_encoding, encoding_for_model
from weaviate_interface_v3 import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from openai import BadRequestError
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from reranker import ReRanker
from openai import OpenAI

from loguru import logger 
import streamlit as st
import os
import system_prompts
import base64
import json

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="Personal Database MSM",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)
##############
# START CODE #
##############

def encode_image(uploaded_file):
  return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

## DATA + CACHE
data_path = 'data/impact_theory_data.json'
cache_path = '/Users/luismi/Downloads/impact_theory_expanded.parquet'
data = load_data(data_path)
cache = None  # Initialize cache as None

# Check if the cache file exists before attempting to load it
if os.path.exists(cache_path):
    cache = load_content_cache(cache_path)
else:
    logger.warning(f"Cache file {cache_path} not found. Proceeding without cache.")

#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

with st.sidebar:
    st.subheader("Selecciona tu Base de datos 🗃️")
    client_type = st.radio(
        "Selecciona el modo de acceso:",
        ('Cloud', 'Local')
    )
if client_type == 'Cloud':
    api_key = st.secrets['WEAVIATE_CLOUD_API_KEY']
    url = st.secrets['WEAVIATE_CLOUD_ENDPOINT']

    weaviate_client = WeaviateClient(
        endpoint=url,
        api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(weaviate_client.show_classes())
    logger.info(available_classes)
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")
elif client_type == 'Local':
    url = st.secrets['WEAVIATE_LOCAL_ENDPOINT']
    weaviate_client = WeaviateClient(
        endpoint=url,
        # api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(weaviate_client.show_classes())
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def main():
    
    # Define the available user selected options
    available_models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']
    # Define system prompts
    system_prompt_list = ["🤖ChatGPT","🧙🏾‍♂️Professor Synapse", "👩🏼‍💼Marketing Jane"]


    # Initialize selected options in session state
    if "openai_data_model" not in st.session_state:
        st.session_state["openai_data_model"] = available_models[0]
    if "system_prompt_data_list" not in st.session_state and "system_prompt_data_model" not in st.session_state:
        # This should be the emoji string the user selected
        st.session_state["system_prompt_data_list"] = system_prompt_list[0]
        # Now we get the corresponding prompt variable using the selected emoji string
        st.session_state["system_prompt_data_model"] = system_prompts.prompt_mapping[system_prompt_list[0]]

    # logger.debug(f"Assistant: {st.session_state['system_prompt_sync_list']}")
    # logger.debug(f"System Prompt: {st.session_state['system_prompt_sync_model']}")

    if 'class_name' not in st.session_state:
        st.session_state['class_name'] = None

    with st.sidebar:
        model_choice = st.selectbox(
            label="Choose an OpenAI model",
            options=available_models,
            index= available_models.index(st.session_state["openai_data_model"]),
        )
        
        system_prompt = st.selectbox(
                label="Choose an Assistant",
                options=system_prompt_list,
                index=system_prompt_list.index(st.session_state["system_prompt_data_list"]),
        )

        st.session_state['class_name'] = st.selectbox(
            label='Repositorio:',
            options=available_classes,
            index=None,
            placeholder='Repositorio'
        )
        
        with st.expander("Filters"):
            guest_input = st.selectbox(
                label='Select Guest',
                options=guest_list,
                index=None,
                placeholder='Select Guest'
            )
        with st.expander("Search Parameters"):
            retriever_choice = st.selectbox(
            label="Choose a retriever",
            options=["Hybrid", "Vector", "Keyword"]
            )
            
            reranker_enabled = st.checkbox(
                label="Enable Reranker",
                value=True
            )

            alpha_input = st.slider(
                label='Alpha for Hybrid',
                min_value=0.00,
                max_value=1.00,
                value=0.40,
                step=0.05
            )
            
            retrieval_limit = st.slider(
                label='Reranked Retrieval Results',
                min_value=10,
                max_value=300,
                value=100,
                step=10
            )
            
            top_k_limit = st.slider(
                label='Top K Limit',
                min_value=1, 
                max_value=5, 
                value=3, 
                step=1
            )
            
            temperature_input = st.slider(
                label='Temperature of LLM',
                min_value=0.0,
                max_value=2.0,
                value=0.10,
                step=0.10
            )

    # Update the model choice in session state
    if st.session_state["openai_data_model"]!=model_choice:
        st.session_state["openai_data_model"] = model_choice
    logger.info(f"Data model: {st.session_state['openai_data_model']}")

    # Update the system prompt choice in session state
    if st.session_state["system_prompt_data_list"] != system_prompt:
        # This should be the emoji string the user selected
        st.session_state["system_prompt_data_list"] = system_prompt  
        # Now we get the corresponding prompt variable using the selected emoji string
        selected_prompt_variable = system_prompts.prompt_mapping[system_prompt]
        st.session_state['system_prompt_data_model'] = selected_prompt_variable
        # logger.info(f"System Prompt: {selected_prompt_variable}")
    logger.info(f"Assistant: {st.session_state['system_prompt_data_list']}")
    # logger.info(f"System Prompt: {st.session_state['system_prompt_sync_model']}")
    
    
    # Check if the collection name has been selected
    class_name = st.session_state['class_name']
    if class_name:
        st.success(f"Repositorio seleccionado ✅: {st.session_state['class_name']}")

    else:
        st.warning("🎗️ No olvides seleccionar el repositorio 👆 a consultar 🗄️.")
        st.stop()  # Stop execution of the script

    weaviate_client.display_properties.append('summary')
    logger.info(weaviate_client.display_properties)


    def database_search(query):
        # Determine the appropriate limit based on reranking
        search_limit = retrieval_limit if reranker_enabled else top_k_limit
        
        # make hybrid call to weaviate
        guest_filter = WhereFilter(
            path=['guest'],
            operator='Equal',
            valueText=guest_input).todict() if guest_input else None

        try:
            # Perform the search based on retriever_choice
            if retriever_choice == "Keyword":
                query_results = weaviate_client.keyword_search(
                    request=query,
                    class_name=class_name,
                    limit=search_limit,
                    where_filter=guest_filter
                )
            elif retriever_choice == "Vector":
                query_results = weaviate_client.vector_search(
                    request=query,
                    class_name=class_name,
                    limit=search_limit,
                    where_filter=guest_filter
                )
            elif retriever_choice == "Hybrid":
                query_results = weaviate_client.hybrid_search(
                    request=query,
                    class_name=class_name,
                    alpha=alpha_input,
                    limit=search_limit,
                    properties=["content"],
                    where_filter=guest_filter
                )
            else:
                return json.dumps({"error": "Invalid retriever choice"})


            ## RERANKER
            reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
            model_name = model_choice
            encoding = encoding_for_model(model_name)
            
            # Rerank the results if enabled
            if reranker_enabled:
                search_results = reranker.rerank(
                    results=query_results,
                    query=query,
                    apply_sigmoid=True,
                    top_k=top_k_limit
                )
            
            else:
                # Use the results directly if reranking is not enabled
                search_results = query_results

            # logger.debug(search_results)
            # Save search results to session state for later use
            # st.session_state['search_results'] = search_results
            add_to_search_history(query=query, search_results=search_results)
            expanded_response = expand_content(search_results, cache, content_key='doc_id', create_new_list=True)

            # validate token count is below threshold
            token_threshold = 8000
            valid_response = validate_token_threshold(
                ranked_results=expanded_response,
                base_prompt=question_answering_prompt_series,
                query=query,
                tokenizer=encoding,
                token_threshold=token_threshold,
                verbose=True
            )
            
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)

            return json.dumps({
                    "query": query,
                    "Search Results": prompt,
                })
    
        except Exception as e:
            # Handle any exceptions and return a JSON formatted error message
            return json.dumps({
                "error": "An error occurred during the search",
                "details": str(e)
            })

    ########################
    ## SETUP MAIN DISPLAY ##
    ########################
    st.image('./assets/impact-theory-logo.png', width=200)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, col2 = st.columns([50,50])
    # Initialize chat history
    if "data_chat_history" not in st.session_state:
        st.session_state["data_chat_history"] = []
    
    if "data_search_history" not in st.session_state:
        st.session_state["data_search_history"] = []

    # When a new message is added, include the type and content
    def add_to_search_history(query, search_results):
        st.session_state["data_search_history"].append({
            "query": query,
            "search_results": search_results,
        })
    
    # Function to display search results
    def display_search_results():
        # Loop through each item in the search history
        for search in st.session_state['data_search_history']:
            query = search["query"]
            search_results = search["search_results"]
            # Create an expander for each search query
            with st.expander(f"Query: {query}", expanded=False):
                for i, hit in enumerate(search_results):
                    col1, col2 = st.columns([7, 3], gap='large')
                    episode_url = hit['episode_url']
                    title = hit['title']
                    guest = hit['guest']
                    show_length = hit['length']
                    time_string = convert_seconds(show_length)
                    content = hit['content']

                    with col1:
                        st.write(search_result(i=i, 
                                            url=episode_url,
                                            guest=guest,
                                            title=title,
                                            content=content,
                                            length=time_string),
                                unsafe_allow_html=True)
                        st.write('\n\n')

                    with col2:
                        image = hit['thumbnail_url']
                        st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                        st.markdown(f'''
                                    <p style="text-align: right;">
                                        <b>Episode:</b> {title.split('|')[0]}<br>
                                        <b>Guest:</b> {guest}<br>
                                        <b>Length:</b> {time_string}
                                    </p>''', unsafe_allow_html=True)

    with col1:
        st.write("Chat History:")
        with st.container(height=500, border=True):
            # Display chat messages from history on app rerun
            for message in st.session_state["data_chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
        st.session_state["data_chat_history"].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)


        tools = [
            {
                "type": "function",
                "function": {
                    "name": "database_search",
                    "description": "Takes the users query about the database and returns the results, extracting info to answer the user's question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                                "query": {"type": "string", "description": "query"},  
                                
                            },
                        "required": ["query"],
                    },
                }
            }
        ]

        # Display live assistant response in chat message container
        with st.chat_message(
            name="assistant",
            avatar="assets/openai_purple_logo_hres.jpeg"):
            message_placeholder = st.empty()

        # Building the messages payload with proper OPENAI API structure
        messages=[
                {"role": "system", "content": st.session_state["system_prompt_data_model"]}
            ] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state["data_chat_history"]
            ]
        logger.debug(f"Initial Messages: {messages}")
        # call the OpenAI API to get the response
        
        RESPONSE = client.chat.completions.create(
            model=st.session_state["openai_data_model"],
            temperature=0.8,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
            stream=True
        )
        logger.debug(f"First Response: {RESPONSE}") 


        FULL_RESPONSE = ""
        tool_calls = []
        # build up the response structs from the streamed response, simultaneously sending message chunks to the browser
        for chunk in RESPONSE:
            delta = chunk.choices[0].delta
            # logger.debug(f"chunk: {delta}")

            

            if delta and delta.content:
                text_chunk = delta.content
                FULL_RESPONSE += str(text_chunk)
                message_placeholder.markdown(FULL_RESPONSE + "▌")
            
            elif delta and delta.tool_calls:
                tcchunklist = delta.tool_calls
                for tcchunk in tcchunklist:
                    if len(tool_calls) <= tcchunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": { "name": "", "arguments": "" } })
                    tc = tool_calls[tcchunk.index]

                    if tcchunk.id:
                        tc["id"] += tcchunk.id
                    if tcchunk.function.name:
                        tc["function"]["name"] += tcchunk.function.name
                    if tcchunk.function.arguments:
                        tc["function"]["arguments"] += tcchunk.function.arguments
        if tool_calls:
            logger.debug(f"tool_calls: {tool_calls}")
            # Define a dictionary mapping function names to actual functions
            available_functions = {
                "database_search": database_search,
                # Add other functions as necessary
            }
            available_functions = {
                "database_search": database_search,
            }  # only one function in this example, but you can have multiple
            logger.debug(f"FuncCall Before messages: {messages}")
            # Process each tool call
            for tool_call in tool_calls:
                # Get the function name and arguments from the tool call
                function_name = tool_call['function']['name']
                function_args = json.loads(tool_call['function']['arguments'])
                
                # Get the actual function to call
                function_to_call = available_functions[function_name]

                # Call the function and get the response
                function_response = function_to_call(**function_args)

                # Append the function response to the messages list
                messages.append({
                    "role": "assistant",
                    "tool_call_id": tool_call['id'],
                    "name": function_name,
                    "content": function_response,
                })
            logger.debug(f"FuncCall After messages: {messages}")

            RESPONSE = client.chat.completions.create(
                model=st.session_state["openai_data_model"],
                temperature=0.8,
                messages=messages,
                stream=True
            )
            logger.debug(f"Second Response: {RESPONSE}") 

            # build up the response structs from the streamed response, simultaneously sending message chunks to the browser
            for chunk in RESPONSE:
                delta = chunk.choices[0].delta
                # logger.debug(f"chunk: {delta}")

                if delta and delta.content:
                    text_chunk = delta.content
                    FULL_RESPONSE += str(text_chunk)
                    message_placeholder.markdown(FULL_RESPONSE + "▌")
        # Add assistant response to chat history
        st.session_state["data_chat_history"].append({"role": "assistant", "content": FULL_RESPONSE})
        logger.debug(f"chat_history: {st.session_state['data_chat_history']}")
    
# Next block of code...

        
    ####################
    ## Search Results ##
    ####################
    # st.subheader(subheader_msg)
    with col2:
        st.write("Search Results:")
        with st.container(height=500, border=True):
            # Check if 'data_search_history' is in the session state and not empty
                if 'data_search_history' in st.session_state and st.session_state['data_search_history']:
                    display_search_results()
                    # # Extract the latest message from the search history
                    #     latest_search = st.session_state['data_search_history'][-1]
                    #     query = latest_search["query"]
                    #     with st.expander(query, expanded=False):
                    #         # Extract the latest message from the search history
                    #         latest_search = st.session_state['data_search_history'][-1]
                    #         query = latest_search["query"]
                    #         for i, hit in enumerate(latest_search["search_results"]):
                    #             col1, col2 = st.columns([7, 3], gap='large')
                    #             episode_url = hit['episode_url']
                    #             title = hit['title']
                    #             guest=hit['guest']
                    #             show_length = hit['length']
                    #             time_string = convert_seconds(show_length)
                    #             # content = ranked_response[i]['content'] # Get 'content' from the same index in ranked_response
                    #             content = hit['content']
                            
                    #             with col1:
                    #                 st.write( search_result(i=i, 
                    #                                         url=episode_url,
                    #                                         guest=guest,
                    #                                         title=title,
                    #                                         content=content,
                    #                                         length=time_string),
                    #                                         unsafe_allow_html=True)
                    #                 st.write('\n\n')

                    #                 # with st.container("Episode Summary:"):
                    #                 #     try:
                    #                 #         ep_summary = hit['summary']
                    #                 #         st.write(ep_summary)
                    #                 #     except Exception as e:
                    #                 #         st.error(f"Error displaying summary: {e}")

                    #             with col2:
                    #                 image = hit['thumbnail_url']
                    #                 st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    #                 st.markdown(f'''
                    #                             <p style="text-align: right;">
                    #                                 <b>Episode:</b> {title.split('|')[0]}<br>
                    #                                 <b>Guest:</b> {hit['guest']}<br>
                    #                                 <b>Length:</b> {time_string}
                    #                             </p>''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()