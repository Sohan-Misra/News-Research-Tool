## NEWS RESEACH TOOL
This project takes multiple URLs from the user as input then uses a vector database to store and retrieve the contents of the URLs as per user query using an llm.

## TCHSTACK USED
1. Python
2. Langchain
3. FAISS (Facebook AI Similarity Search)
4. Streamlit
5. Google Gemini-1.5-flash.

## STEPS TO EXECUTE
1. Install the required libraries preferrably in a virtual environment (like Conda).
2. Create a `.env` file with your own GOOGLE `API key` (for Gemini). For using other llms refer to their respective documentation and the `langchain` documentation.
3. Run `main.py` using the command `streamlit run main.py`.
4. Paste the URL's in the box and press `process`.
5. Wait for processing to finish and then enter your query. 