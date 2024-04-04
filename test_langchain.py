from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

def load_text(query_text,text_path):

    """
    加载文档
    """

    loader = TextLoader(f"./{text_path}", encoding="utf-8")
    docs = loader.load()

    """
    分割文档
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    text_strs = []
    for text in texts:
        text_strs.append(text.page_content)

    """
    文档向量化
    """
    

    embeddings = HuggingFaceEmbeddings()
    query_text = query_text
    query_result = embeddings.embed_query(query_text)
    doc_result = embeddings.embed_documents(text_strs)

    """
    向量化存储
    """
    
    db = FAISS.from_documents(texts, embeddings)
    docs = db.similarity_search_by_vector(query_result)
    for doc in docs:
        print(doc)
    db.save_local('./db')


def search_text(query_text):

    """
    查询文本
    """
    

    embeddings = HuggingFaceEmbeddings()
    query_text = query_text

    """
    向量数据库查询
    """
    from langchain.vectorstores import FAISS

    db = FAISS.load_local('./db', embeddings,allow_dangerous_deserialization=True)
    docs = db.similarity_search(query_text)
    for doc in docs:
        print(doc)

    doc_strs = []
    for doc in docs:
        doc_strs.append(doc.page_content)
    doc_strs = '\n'.join(doc_strs)

    return doc_strs

    """
    构建提示词
    """

    print(doc_strs)

    prompt = PromptTemplate.from_template("请根据以下内容写小说\n内容：{doc}\n提示：{query}\n\n")
    prompt = prompt.format(doc=doc_strs, query=query_text)
    print(prompt)





if __name__ == '__main__':
    
    search_text()
