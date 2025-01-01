from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
from typing import List, Dict

class StoryGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=api_key,
            model="gpt-4"
        )

    def generate_story(self, category: str, topics: List[str], evidence: Dict[str, List[Dict]]) -> Dict:
        """Generate a coherent story with proper evidence integration"""
        try:
            # Organize evidence by topic
            evidence_by_topic = {}
            for topic in topics:
                topic_evidence = evidence.get(topic, [])
                
                # Separate quotes and supporting evidence
                quotes = [e for e in topic_evidence if e['relevance'] == 'direct_quote']
                support = [e for e in topic_evidence if e['relevance'] == 'supporting_evidence']
                
                evidence_by_topic[topic] = {
                    'quotes': quotes,
                    'support': support
                }

            # Create story prompt with structured evidence
            story_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a technical product analyst writing data-driven stories.
                Focus on creating clear, actionable insights supported by evidence.
                
                Story Structure:
                1. Headline - Capture key insight
                2. Executive Summary - Main finding with key metrics
                3. Analysis - Detailed discussion with evidence
                4. Technical Context - Relevant technical details
                5. Action Items - Clear next steps
                6. Evidence Summary - Supporting data and quotes"""),
                
                ("user", """Generate a product story for category: {category}

                Topics to cover:
                {topics}

                Available Evidence:
                {evidence}

                Requirements:
                - Write 200-300 words
                - Include specific metrics and data points
                - Use direct quotes where relevant
                - Maintain journalistic style
                - Maintain technical accuracy
                - Provide clear action items
                - Cite sources properly""")
            ])

            # Format evidence for prompt
            evidence_text = ""
            for topic in topics:
                evidence_text += f"\nFor {topic}:\n"
                if evidence_by_topic[topic]['quotes']:
                    evidence_text += "Quotes:\n"
                    for quote in evidence_by_topic[topic]['quotes']:
                        evidence_text += f"- {quote['text']} (Source: {quote['source']})\n"
                if evidence_by_topic[topic]['support']:
                    evidence_text += "Supporting Evidence:\n"
                    for support in evidence_by_topic[topic]['support']:
                        evidence_text += f"- {support['text']} (Source: {support['source']})\n"

            # Generate story
            response = self.llm.invoke(
                story_prompt.format_messages(
                    category=category,
                    topics="\n".join(f"- {topic}" for topic in topics),
                    evidence=evidence_text
                )
            )

            # Parse and structure the response
            story_content = response.content
            
            return {
                'category': category,
                'topics': topics,
                'story': story_content,
                'evidence_used': evidence_by_topic,
                'word_count': len(story_content.split())
            }

        except Exception as e:
            logging.error(f"Error generating story: {str(e)}")
            return None 