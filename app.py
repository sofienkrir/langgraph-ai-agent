import os

import streamlit as st
from dotenv import load_dotenv

from agent import documents
from agent.coverage import MAX_SEARCH_ROUNDS, MIN_SOURCES_PER_SUB_QUESTION
from agent.graph import build_research_graph
from agent.llm import active_provider, model_for

load_dotenv()

st.set_page_config(
    page_title="LangGraph Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .35rem 0;}
.small-muted {color: #7b8492; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

provider = active_provider()


@st.cache_data(show_spinner=False)
def index_pdf_cached(data: bytes, filename: str) -> dict:
    """Index one PDF. Cached so a Streamlit rerun does not re-embed the file."""
    return documents.index_pdf(data, filename)


with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption("LangGraph · recherche sourcée")

    st.markdown("**Fournisseur**")
    st.code(
        f"{provider}\nplanner      : {model_for('planner', provider)}\n"
        f"synthesizer  : {model_for('synthesizer', provider)}",
        language="text",
    )
    st.caption("Change `LLM_PROVIDER` dans `.env` pour basculer (`groq` ou `anthropic`).")

    st.divider()
    st.markdown("**📄 Corpus local (PDF)**")
    uploads = st.file_uploader(
        "Ajouter des PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads:
        for upload in uploads:
            with st.spinner(f"Indexation de {upload.name}…"):
                try:
                    result = index_pdf_cached(upload.getvalue(), upload.name)
                except Exception as exc:
                    st.error(f"{upload.name} : {exc}")
                    continue
            if result["skipped"]:
                st.caption(f"↩️ {upload.name} — déjà indexé")
            elif result["chunks"]:
                st.caption(
                    f"✅ {upload.name} — {result['pages']} pages, {result['chunks']} passages"
                )
            else:
                st.warning(
                    f"{upload.name} : aucun texte extractible. "
                    "Un PDF scanné nécessiterait de l'OCR, hors périmètre ici."
                )

    stats = documents.index_stats()
    if stats["files"]:
        st.markdown(
            f"<span class='small-muted'>{stats['chunks']} passages indexés</span>",
            unsafe_allow_html=True,
        )
        for name in stats["files"]:
            st.markdown(f"<span class='small-muted'>· {name}</span>", unsafe_allow_html=True)
        if st.button("Vider le corpus", use_container_width=True):
            documents.clear_index()
            index_pdf_cached.clear()
            st.rerun()
    else:
        st.caption("Aucun document. L'agent cherchera uniquement sur le Web.")

    st.divider()
    st.markdown("**Seuils de couverture**")
    st.markdown(
        f"- {MIN_SOURCES_PER_SUB_QUESTION} sources minimum par sous-question\n"
        f"- {MAX_SEARCH_ROUNDS} tours de recherche maximum"
    )
    st.caption("Contrôle en Python, sans score de confiance généré par un LLM.")

    st.divider()
    llm_key = os.getenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "GROQ_API_KEY", "")
    search_key = os.getenv("TAVILY_API_KEY", "")
    st.markdown("**Clés**")
    st.markdown(f"- LLM ({provider}) : {'✅' if llm_key.strip() else '❌ manquante'}")
    st.markdown(f"- Tavily : {'✅' if search_key.strip() else '⚠️ absente (Web désactivé)'}")

st.markdown("""
<div class="hero">
  <h1>🔎 LangGraph Research Agent</h1>
  <div class="small-muted">Décompose la question, cherche dans tes PDF et sur le Web,
  vérifie la couverture des sources, puis rédige un rapport cité — en signalant ce
  qu'il n'a pas trouvé.</div>
</div>
""", unsafe_allow_html=True)

has_docs = bool(stats["files"])

if not llm_key.strip():
    st.info("Ajoute ta clé LLM dans le fichier `.env` local pour démarrer.")
    st.code(
        ("ANTHROPIC_API_KEY" if provider == "anthropic" else "GROQ_API_KEY") + "=...",
        language="bash",
    )
    st.markdown("[Clé Groq](https://console.groq.com/keys)")
    st.stop()

if not search_key.strip() and not has_docs:
    st.info(
        "Sans clé Tavily, l'agent n'a aucune source. Ajoute `TAVILY_API_KEY` dans `.env` "
        "pour la recherche Web, ou importe au moins un PDF dans la barre latérale."
    )
    st.code("TAVILY_API_KEY=...", language="bash")
    st.markdown("[Clé Tavily](https://app.tavily.com)")
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None

question = st.text_area(
    "Question de recherche",
    placeholder="Quelles sont les principales méthodes de prévision de la production solaire par IA ?",
    height=100,
)

if st.button("Lancer la recherche", type="primary", disabled=not question.strip()):
    graph = build_research_graph()
    state = {"question": question.strip()}

    with st.status("Recherche en cours…", expanded=True) as status:
        final = None
        try:
            for step in graph.stream(state, stream_mode="values"):
                final = step
                for line in step.get("trace", [])[-2:]:
                    st.write(line)
            status.update(label="Recherche terminée", state="complete")
        except Exception as exc:
            status.update(label="Recherche interrompue", state="error")
            st.error(f"Échec : {exc}")
            final = None
    st.session_state.result = final

result = st.session_state.result
if result:
    report_tab, sources_tab, trace_tab = st.tabs(["Rapport", "Sources", "Déroulé"])

    with report_tab:
        st.markdown(result.get("report", "_Aucun rapport._"))

    with sources_tab:
        sources = result.get("sources", [])
        local_count = sum(1 for s in sources if s.get("kind") == "document")
        st.caption(
            f"{len(sources)} sources ({local_count} document, {len(sources) - local_count} web) "
            "— la numérotation correspond aux citations du rapport."
        )
        for index, source in enumerate(sources, start=1):
            if source.get("kind") == "document":
                label = f"📄 {source['title']}"
            else:
                label = f"🌐 [{source['title']}]({source['url']})"
            st.markdown(
                f"**[{index}]** {label}  \n"
                f"<span class='small-muted'>{source['sub_question_id']} · "
                f"pertinence {source['score']:.2f}</span>",
                unsafe_allow_html=True,
            )

    with trace_tab:
        for sub_question in result.get("sub_questions", []):
            flag = "⚠️" if sub_question["id"] in result.get("gaps", []) else "✅"
            st.markdown(f"{flag} **{sub_question['id']}** — {sub_question['text']}")
        st.divider()
        st.code("\n".join(result.get("trace", [])), language="text")

    st.download_button(
        "Télécharger le rapport (Markdown)",
        data=result.get("report", ""),
        file_name="rapport.md",
        mime="text/markdown",
    )
