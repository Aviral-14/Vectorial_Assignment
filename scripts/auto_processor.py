from typing import Dict, List
import logging
from .topic_extractor import TopicExtractor
from .evidence_collector import EvidenceCollector
from .story_generator import StoryGenerator
import json
import os
from datetime import datetime

class AutoDocumentProcessor:
    def __init__(self, api_key: str):
        self.topic_extractor = TopicExtractor(api_key)
        self.evidence_collector = EvidenceCollector(api_key)
        self.story_generator = StoryGenerator(api_key)
        logging.basicConfig(level=logging.INFO)

    async def process_documents(self, documents: Dict[str, str]) -> Dict:
        """Process documents with improved workflow and error handling"""
        try:
            # Extract topics
            logging.info(f"Processing {len(documents)} documents...")
            topics = self.topic_extractor.extract_topics(documents)
            
            if not any(topics.get(key) for key in ['concerns', 'wins', 'opportunities']):
                raise ValueError("No topics were extracted from documents")

            # Collect evidence
            evidence = self.evidence_collector.collect_evidence(documents, topics)

            # Generate stories
            stories = []
            categories = ['concerns', 'wins', 'opportunities']
            
            for category in categories:
                if topics.get(category):
                    story = self.story_generator.generate_story(
                        category=category,
                        topics=topics[category],
                        evidence=evidence
                    )
                    if story:
                        stories.append({
                            "category": category,
                            "story": story
                        })

            return {
                "status": "success",
                "stories": stories,
                "metadata": {
                    "document_count": len(documents),
                    "generated_stories": len(stories)
                }
            }

        except Exception as e:
            logging.error(f"Error in document processing: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "stories": []
            } 