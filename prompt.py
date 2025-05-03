from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

template_text = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question.\ 
    If you don't know the answer, just say that you don't know.\ 
    Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer: """

prompt = ChatPromptTemplate(input_variables=['context', 'question'],
                            messages=[
                                HumanMessagePromptTemplate(
                                    prompt=PromptTemplate(input_variables=['context', 'question'],
                                                          template=template_text)
                                )])