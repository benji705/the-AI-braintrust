import os
import re
from flask import Flask, render_template, request
from docling.document_converter import DocumentConverter
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.readers import ExtractiveReader
from haystack.components.preprocessors import DocumentSplitter

app = Flask(__name__)

# --- INITIALISATION ---
file_name = "7a3b28ef.pdf"
print("🛡️ Démarrage du service Expert Assur'IA...")

document_store = InMemoryDocumentStore()
converter = DocumentConverter()
result = converter.convert(file_name)

all_docs = []
for page_no, page in result.document.pages.items():
    page_text = result.document.export_to_markdown(page_no=page_no)
    doc = Document(content=page_text, meta={"page": str(page_no)})
    all_docs.append(doc)

# Découpage en blocs pour isoler les paragraphes
splitter = DocumentSplitter(split_by="word", split_length=350, split_overlap=50)
docs_decoupes = splitter.run(documents=all_docs)
document_store.write_documents(docs_decoupes["documents"])

retriever = InMemoryBM25Retriever(document_store=document_store, top_k=3)
reader = ExtractiveReader(model="deepset/roberta-base-squad2")

pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("reader", reader)
pipe.connect("retriever", "reader.documents")

print("✅ IA prête et connectée au contrat.")

# --- ROUTE PRINCIPALE ---

@app.route('/')
def home():
    return render_template('index.html')

from groq import Groq

# Remplacez par votre clé Groq
client_llm = Groq(api_key"Clé")

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    
    try:
        # 1. Recherche du paragraphe (Haystack)
        result_ia = pipe.run(data={"retriever": {"query": user_question}, "reader": {"query": user_question, "top_k": 1}})

        # Initialisation par défaut si rien n'est trouvé
        explication_bot = "Désolé, je ne trouve pas d'information spécifique à ce sujet dans le contrat."
        paragraphe_brut = ""
        page_no = "?"
        section_nom = "Général"

        if result_ia["reader"]["answers"]:
            ans = result_ia["reader"]["answers"][0]
            # Vérifier si l'IA est assez sûre d'elle (score > 0.35)
            if ans.score > 0.30:
                paragraphe_brut = ans.document.content
                page_no = ans.document.meta.get("page", "?")
                
                # 2. Le Bot Groq rédige l'explication
                chat_completion = client_llm.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Tu es l'Expert Assur'IA. Réponds de manière courte et simple à partir du texte fourni."},
                        {"role": "user", "content": f"Question: {user_question} \n Texte: {paragraphe_brut}"}
                    ],
                    model="llama-3.3-70b-versatile",
                )
                explication_bot = chat_completion.choices[0].message.content

        return render_template('index.html', 
                               explication=explication_bot, 
                               paragraphe=paragraphe_brut,
                               page=page_no,
                               section=section_nom, # Ajout de la variable manquante
                               question=user_question)

    except Exception as e:
        print(f"❌ ERREUR DÉTAILLÉE : {e}")
        return render_template('index.html', explication=f"Erreur technique : contactez l'assistance pour plus de renseignement.")
if __name__ == '__main__':
    app.run(debug=False, port=5000)
