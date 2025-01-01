from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import logging
from typing import List, Dict
import numpy as np

class EvidenceCollector:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=api_key,
            model="gpt-4"
        )
        # Optimized chunk settings for evidence collection
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for more precise evidence
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def _prepare_chunks(self, documents: Dict[str, str]) -> List[Dict]:
        """Prepare document chunks with source tracking and metadata"""
        chunks = []
        for filename, content in documents.items():
            texts = self.text_splitter.split_text(content)
            for i, text in enumerate(texts):
                chunks.append({
                    'content': text,
                    'metadata': {
                        'source': filename,
                        'chunk_id': i,
                        'is_quote': '"' in text or '"' in text or '"' in text,
                        'has_numbers': any(char.isdigit() for char in text)
                    }
                })
        return chunks

    def _create_vectorstore(self, chunks: List[Dict]) -> FAISS:
        """Create FAISS vectorstore with metadata"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def _score_evidence(self, evidence: Dict, topic: str) -> float:
        """Score evidence relevance based on multiple factors"""
        score = 0.0
        content = evidence['content']
        metadata = evidence['metadata']

        # Relevance factors
        if metadata['is_quote']:  # Direct quotes are valuable
            score += 0.3
        if metadata['has_numbers']:  # Numerical evidence is valuable
            score += 0.3
        if len(content.split()) >= 10:  # Sufficient context
            score += 0.2
        if any(keyword in content.lower() for keyword in topic.lower().split()):  # Direct topic mention
            score += 0.2

        return score

    def collect_evidence(self, documents: Dict[str, str], topics: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """Collect relevant evidence for each topic using RAG"""
        try:
            # Prepare chunks with metadata
            chunks = self._prepare_chunks(documents)
            vectorstore = self._create_vectorstore(chunks)
            
            evidence_by_topic = {}
            
            # Process each category and its topics
            for category, topic_list in topics.items():
                if category == 'quotes':  # Skip the quotes dictionary
                    continue
                    
                for topic in topic_list:
                    # Get initial set of relevant chunks
                    results = vectorstore.similarity_search_with_score(
                        topic,
                        k=10,  # Get more candidates for filtering
                        fetch_k=20  # Cast wider net
                    )

                    # Process and score evidence
                    scored_evidence = []
                    for doc, similarity_score in results:
                        evidence = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'similarity': similarity_score
                        }
                        relevance_score = self._score_evidence(evidence, topic)
                        final_score = (similarity_score + relevance_score) / 2
                        scored_evidence.append((evidence, final_score))

                    # Sort by score and select best evidence
                    scored_evidence.sort(key=lambda x: x[1], reverse=True)
                    best_evidence = [e[0] for e in scored_evidence[:5]]  # Top 5 pieces of evidence

                    # Validate evidence quality
                    validation_prompt = f"""
                    Topic: {topic}
                    Category: {category}
                    
                    Review these pieces of evidence and confirm they support the topic:
                    {[e['content'] for e in best_evidence]}
                    
                    Return only the indices of relevant evidence (e.g., 0,2,4).
                    """
                    
                    response = self.llm.invoke(validation_prompt)
                    try:
                        valid_indices = [int(i) for i in response.content.strip().split(',')]
                        validated_evidence = [best_evidence[i] for i in valid_indices if i < len(best_evidence)]
                    except:
                        validated_evidence = best_evidence

                    evidence_by_topic[topic] = [{
                        'text': e['content'],
                        'source': e['metadata']['source'],
                        'relevance': 'direct_quote' if e['metadata']['is_quote'] else 'supporting_evidence'
                    } for e in validated_evidence]

                    logging.info(f"Collected {len(validated_evidence)} pieces of evidence for topic: {topic}")

            return evidence_by_topic

        except Exception as e:
            logging.error(f"Error collecting evidence: {str(e)}")
            return {} 