from tiktoken import get_encoding, encoding_for_model
from weaviate_interface_v3 import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from openai import BadRequestError
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="👨🏻‍💻 AI Database ",
                   page_icon="👨🏻‍💻",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

## DATA + CACHE
data_path = 'data/impact_theory_data.json'
cache_path = 'data/impact_theory_expanded.parquet'
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

    client = WeaviateClient(
        endpoint=url,
        api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(client.show_classes())
    logger.info(available_classes)
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")
elif client_type == 'Local':
    url = st.secrets['WEAVIATE_LOCAL_ENDPOINT']
    client = WeaviateClient(
        endpoint=url,
        # api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(client.show_classes())
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")

def main():
    
    # Define the available user selected options
    available_models = ['gpt-3.5-turbo', 'gpt-4-1106-preview']
    # Define system prompts

    # Initialize selected options in session state
    if "openai_data_model" not in st.session_state:
        st.session_state["openai_data_model"] = available_models[0]
    
    if 'class_name' not in st.session_state:
        st.session_state['class_name'] = None

    with st.sidebar:
        st.session_state['class_name'] = st.selectbox(
            label='Repositorio:',
            options=available_classes,
            index=None,
            placeholder='Repositorio'
            )
        
        model_choice = st.selectbox(
            label="Choose an OpenAI model",
            options=available_models,
            index= available_models.index(st.session_state["openai_data_model"]),
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
    # Check if the collection name has been selected
    class_name = st.session_state['class_name']
    if class_name:
        st.success(f"Repositorio seleccionado ✅: {st.session_state['class_name']}")

    else:
        st.warning("🎗️ No olvides seleccionar el repositorio 👆 a consultar 🗄️.")
        st.stop()  # Stop execution of the script

    client.display_properties.append('summary')
    logger.info(client.display_properties)

    def perform_search(client, retriever_choice, query, class_name, search_limit, guest_filter, display_properties, alpha_input):
        if retriever_choice == "Keyword":
            return client.keyword_search(
                request=query,
                class_name=class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Keyword Search results: "
        elif retriever_choice == "Vector":
            return client.vector_search(
                request=query,
                class_name=class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Vector Search results: "
        elif retriever_choice == "Hybrid":
            return client.hybrid_search(
                request=query,
                class_name=class_name,
                alpha=alpha_input,
                limit=search_limit,
                properties=["content"],
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Hybrid Search results: "


    ## RERANKER
    reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

    ## LLM
    model_name = model_choice
    llm = GPT_Turbo(model=model_name, api_key=st.secrets['OPENAI_API_KEY'])
    encoding = encoding_for_model(model_name)


    ########################
    ## SETUP MAIN DISPLAY ##
    ########################
    st.image('./assets/impact-theory-logo.png', width=200)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        st.sidebar.make_llm_call = st.checkbox(
            label="Enable LLM",
        )
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        ############
        ## SEARCH ##
        ############
        if query:
            # make hybrid call to weaviate
            guest_filter = WhereFilter(
                path=['guest'],
                operator='Equal',
                valueText=guest_input).todict() if guest_input else None
            
            
            # Determine the appropriate limit based on reranking
            search_limit = retrieval_limit if reranker_enabled else top_k_limit

            # Perform the search
            query_response, subheader_msg = perform_search(
                client=client,
                retriever_choice=retriever_choice,
                query=query,
                class_name=class_name,
                search_limit=search_limit,
                guest_filter=guest_filter,
                display_properties=client.display_properties,
                alpha_input=alpha_input if retriever_choice == "Hybrid" else None
                )
            
            
            # Rerank the results if enabled
            if reranker_enabled:
                search_results = reranker.rerank(
                    results=query_response,
                    query=query,
                    apply_sigmoid=True,
                    top_k=top_k_limit
                )
                subheader_msg += " Reranked"
            else:
                # Use the results directly if reranking is not enabled
                search_results = query_response

            # logger.info(search_results)
            expanded_response = expand_content(search_results, cache, content_key='doc_id', create_new_list=True)

            # validate token count is below threshold
            token_threshold = 8000 if model_name == 'gpt-3.5-turbo-16k' else 3500
            valid_response = validate_token_threshold(
                ranked_results=expanded_response,
                base_prompt=question_answering_prompt_series,
                query=query,
                tokenizer=encoding,
                token_threshold=token_threshold,
                verbose=True
            )

            #########
            ## LLM ##
            #########
            make_llm_call = st.sidebar.make_llm_call
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                # Creates container for LLM response
                chat_container, response_box = [], st.empty()

                # generate LLM prompt
                prompt = generate_prompt_series(query=query, results=valid_response)
                # logger.info(prompt)
                if make_llm_call:

                    try:
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            max_tokens=350, # expand for more verbose answers
                            show_response=True,
                            stream=True):

                            # inserts chat stream from LLM
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                    except BadRequestError:
                        logger.info('Making request with smaller context...')
                        valid_response = validate_token_threshold(
                            ranked_results=search_results,
                            base_prompt=question_answering_prompt_series,
                            query=query,
                            tokenizer=encoding,
                            token_threshold=token_threshold,
                            verbose=True
                        )

                        # generate LLM prompt
                        prompt = generate_prompt_series(query=query, results=valid_response)
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            max_tokens=350, # expand for more verbose answers
                            show_response=True,
                            stream=True):

                            try:
                                # inserts chat stream from LLM
                                with response_box:
                                    content = resp.choices[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                            except Exception as e:
                                print(e)

            ####################
            ## Search Results ##
            ####################
            st.subheader(subheader_msg)
            for i, hit in enumerate(search_results):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                guest=hit['guest']
                show_length = hit['length']
                time_string = convert_seconds(show_length)
                # content = ranked_response[i]['content'] # Get 'content' from the same index in ranked_response
                content = hit['content']
            
                with col1:
                    st.write( search_result(i=i, 
                                            url=episode_url,
                                            guest=guest,
                                            title=title,
                                            content=content,
                                            length=time_string),
                                            unsafe_allow_html=True)
                    st.write('\n\n')

                    with st.expander("Click Here for Episode Summary:"):
                        try:
                            ep_summary = hit['summary']
                            st.write(ep_summary)
                        except Exception as e:
                            print(e)

                with col2:
                    image = hit['thumbnail_url']
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'''
                                <p style="text-align: right;">
                                    <b>Episode:</b> {title.split('|')[0]}<br>
                                    <b>Guest:</b> {hit['guest']}<br>
                                    <b>Length:</b> {time_string}
                                </p>''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()