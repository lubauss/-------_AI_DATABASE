from weaviate_interface_v3 import WeaviateClient, WhereFilter
from reranker import ReRanker
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from tiktoken import get_encoding, encoding_for_model

class AppConfig:
    def __init__(self, weaviate_client)
        # Initialize all configurations with None or default values
        self.retriever_choice = None
        self.query = None  # This might be set for each search operation
        self.class_name = None
        self.retrieval_limit = None
        self.top_k_limit = None
        self.reranker_enabled = None
        self.guest_input = None
        self.display_properties = None
        self.alpha_input = None  # You might want to set a default value here

        # Add a reference to the weaviate client
        self.weaviate_client = weaviate_client

    # Example method to update retriever choice
    def update_retriever_choice(self, retriever_choice):
        self.retriever_choice = retriever_choice
    def update_class_name(self, class_name):
        self.class_name = class_name
    def update_query(self, query):
        self.query = query
    def update_retrieval_limit(self, search_limit):
        self.search_limit = search_limit
    def update_top_k_limit(self, top_k_limit):
        self.top_k_limit = top_k_limit
    def update_guest_input(self, guest_input):
        self.guest_input = guest_input
    def update_display_properties(self, display_properties):
        self.display_properties = display_properties
    def update_alpha_input(self, alpha_input):
        self.alpha_input = alpha_input
    def update_reranker_enabled(self, reranker_enabled):
        self.reranker_enabled = reranker_enabled

     def perform_search(self, query):
        # Determine the appropriate limit based on reranking
        search_limit = self.retrieval_limit if self.reranker_enabled else self.top_k_limit
        
        # make hybrid call to weaviate
        guest_filter = WhereFilter(
            path=['guest'],
            operator='Equal',
            valueText=self.guest_input).todict() if self.guest_input else None

        if self.retriever_choice == "Keyword":
            query_results = self.weaviate_client.keyword_search(
                request=query,
                class_name=self.class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=self.display_properties
            )
            return query_results , "Keyword Search results: "
        elif self.retriever_choice == "Vector":
            query_results = self.weaviate_client.vector_search(
                request=query,
                class_name=self.class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=self.display_properties
            ) 
            subheader_msg = "Vector Search results: "
            return query_results, subheader_msg
        elif self.retriever_choice == "Hybrid":
            query_results = self.weaviate_client.hybrid_search(
                request=query,
                class_name=self.class_name,
                alpha=self.alpha_input,
                limit=search_limit,
                properties=["content"],
                where_filter=guest_filter,
                display_properties=self.display_properties
            ) 
            subheader_msg = "Hybrid Search results: "
            return query_results, subheader_msg


        ## RERANKER
        reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        
        # Rerank the results if enabled
        if self.reranker_enabled:
            search_results = reranker.rerank(
                results=query_results,
                query=query,
                apply_sigmoid=True,
                top_k=top_k_limit
            )
            subheader_msg += " Reranked"
        else:
            # Use the results directly if reranking is not enabled
            search_results = query_results

        # logger.info(search_results)
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

        return search_results, valid_response, prompt, subheader_msg