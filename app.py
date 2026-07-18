import gradio as gr
import main

main.initialize_rag()

def ask_question(question):

    if not question.strip():
        return "", "Please enter a question."

    docs = main.retrieval_documents(question)

    context = ""

    for i, doc in enumerate(docs, start=1):

        context += f"""
        Result {i}

        Type   : {doc.metadata.get("type")}
        Source : {doc.metadata.get("source")}

        {doc.page_content}

        {'-'*60}

        """

    answer = main.chat(question, docs)

    return context, answer


with gr.Blocks(title="MultiModal RAG Bot") as demo:
        gr.Markdown(
        """
            # 📄 MultiModal RAG Bot

            ### Ask questions about the **Attention Is All You Need** paper.

            Supports:

            - 📄 Text
            - 📊 Tables
            - 🖼 Figures & Diagrams

            """
                )
        
        question = gr.Textbox(
            label="Question",
            placeholder="Example: What is Multi-Head Attention?",
            lines=2
        )        

        with gr.Row():

            ask = gr.Button(
                "Ask",
                variant="primary"
            )

            clear = gr.Button("Clear")


        answer = gr.Textbox(
            label="Answer",
            lines=8
        )

        retrieved = gr.Textbox(
            label="Retrieved Context",
             lines=16
         )

        ask.click(
             ask_question,
             inputs=question,
            outputs=[retrieved, answer]
        )

        clear.click(
            lambda: ("", "", ""),
                outputs=[question, retrieved, answer]
        )

demo.launch()