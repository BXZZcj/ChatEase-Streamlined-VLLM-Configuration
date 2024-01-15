My expectation for this project is that it will offer a simple way to configure a chatbot on your server, one that supports RAG and multi-modality.

Specifically, I envision this project demonstrating the following effect: A user can pull an implementation of some Large Language Model (LLM) from Hugging Face - including weights and configuration files - onto his server. Then, he can easily turn the LLM into a runnable chatbot with a tidy interface accessible via a browser, just by using this project for configuration.



## Notice:

 The configuration steps outlined below are incomplete, and the code in this repository has not been fully organized.

Clearer code and more detailed configuration steps will be provided soon ...



### Configuration Steps

1. Install dependencies.
2. Mark the 'code' folder as the source root.
3. Run 'code/llm/llm_server.py'.
4. Execute 'chainlit run code/web_ui/streamlit_chat.py --port 23'.
5. Then, you can access the chatbot by opening 'localhost:23' (or the corresponding URL).