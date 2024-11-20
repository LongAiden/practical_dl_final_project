# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import pinecone
import os
import warnings
warnings.filterwarnings('ignore')


def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=1000, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30)
    chunks = text_splitter.split_documents(data)
    return chunks


# Insert to Pinecone index
def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client
    embeddings = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")

    # loading from existing index
    if (index_name in pc.list_indexes().names()):
        print(
            f'Index {index_name} already exists. Loading embeddings ... ', end='')
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        PineconeVectorStore.from_documents(
            chunks, embeddings, index_name=index_name)

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object.

        print('Ok')

    vector_store = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)

    return vector_store


def delete_pinecone_index(index_name='all', exclude_indexes=['documents-embeddings-index']):
    if index_name == 'all':
        indexes = [i for i in pc.list_indexes().names()
                   if i not in exclude_indexes]
        print('Deleting all indexes ... ')
        for index in indexes:
            print(index)
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


def get_chain(vector_store, k):
    '''
    Creates a retrieval-based Q&A chain with conversation memory

    Args:
        vector_store: Vector store containing embedded documents
        k: Number of relevant documents to retrieve for each query

    Returns:
        RetrievalQA: A chain that combines document retrieval with LLM for Q&A
    '''
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    llm = ChatOllama(model="llama3.2:3b", use_gpu=True,
                     max_tokens=2048, temperature=0.5, top_p=0.95)

    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=retriever,
        verbose=False)

    return chain


def ask_and_get_answer_v2(chain):
    llm = ChatOllama(model="llama3.2:3b", use_gpu=True,
                     max_tokens=2048, temperature=0.5, top_p=0.95)
    retriever = vector_store.as_retriever(search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return chain, answer['result']


def ask_and_get_answer(vector_store, q, k=5):
    llm = ChatOllama(model="llama3.2:3b", use_gpu=True,
                     max_tokens=2048, temperature=0.5, top_p=0.95)
    retriever = vector_store.as_retriever(search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    # Load environment variable
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(
        "D:/Online_Learning/Practical_DL/langchain_udemy/ice_breaker/.env"), override=True)

    # Retrieve the API key and index name from environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    st.image(r"C:\Users\ADMIN\Pictures\Adrien_Long_city_in_the_sky_fighting_with_giant_monsters._hyper_91482e6a-63f5-4ff5-845f-db8bee06b978.png", width=100)
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # file uploader widget
        uploaded_file = st.file_uploader(
            'Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input(
            'Chunk size:', min_value=100, max_value=2048, value=1000, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20,
                            value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        # index name text input widget
        index_name_input = st.text_input('Index Name:', value='test_index')

        # delete index button widget
        delete_index = st.button(
            'Delete Current Index', on_click=clear_history)

        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                os.makedirs('./doc_rag', exist_ok=True)
                file_name = os.path.join('./doc_rag', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = insert_or_fetch_embeddings(
                    index_name_input, chunks)

                print("Finish chunking and embedding the file.")
                print(vector_store)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

        if delete_index:  # if the user clicked the delete index button
            with st.spinner('Deleting current index ...'):
                delete_pinecone_index(index_name=index_name_input)
                st.success('Current index deleted successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q:  # if the user entered a question and hit enter
        if q.lower() == 'exit':
            st.write('Session closed.')
            st.stop()
        else:
            q = f"{q}"
            # if there's the vector store (user uploaded, split and embedded a file)
            if 'vs' in st.session_state:
                vector_store = st.session_state.vs

                # Create chain if not in session state
                if 'chain' not in st.session_state:
                    st.session_state.chain = get_chain(vector_store, k)

                st.write(f'k: {k}')
                # Use the chain with memory to get the answer
                answer = st.session_state.chain.invoke({"query": q})["result"]

                # text area widget for the LLM answer
                st.text_area('LLM Answer: ', value=answer)

                st.divider()

                # if there's no chat history in the session state, create it
                if 'history' not in st.session_state:
                    st.session_state.history = ''

                # the current question and answer
                value = f'Q: {q} \nA: {answer}'

                st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
                h = st.session_state.history

                # text area widget for the chat history
                st.text_area(label='Chat History', value=h,
                             key='history', height=400)

# run the app: streamlit run ./project_streamlit_rag.py
# Styling for message containers
st.markdown("""
<style>
.user-message {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    float: right;
    clear: both;
    max-width: 70%;
}
.bot-message {
    background-color: #E8E8E8;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    float: left;
    clear: both;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

def display_message(text, is_user=False):
    message_class = "user-message" if is_user else "bot-message"
    st.markdown(f'<div class="{message_class}">{text}</div>', unsafe_allow_html=True)
    st.markdown("<div style='clear: both'></div>", unsafe_allow_html=True)

# Add chat display in the main interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

if q and 'vs' in st.session_state:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": q})
    
    # Add bot response
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for message in st.session_state.messages:
    display_message(message["content"], message["role"] == "user")