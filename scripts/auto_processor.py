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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)

    def save_intermediate_results(self, stage: str, data: any) -> None:
        """Save intermediate results for debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}_{stage}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_readable_output(self, stories: List[Dict]) -> None:
        """Save stories in a human-readable format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}_readable_stories.txt"
        with open(filename, 'w') as f:
            for story in stories:
                f.write(f"\n{'='*80}\n")
                f.write(f"Category: {story.get('category', 'Unknown')}\n")
                f.write(f"Story:\n{story.get('story', 'No story generated')}\n")

    async def process_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """Process documents with improved workflow and error handling"""
        try:
            # Save input documents
            self.save_intermediate_results("01_input_documents", documents)
            logging.info(f"Processing {len(documents)} documents...")

            # Extract topics
            logging.info("Starting topic extraction...")
            topics = self.topic_extractor.extract_topics(documents)
            self.save_intermediate_results("02_topic_extraction", topics)
            
            if not any(topics.get(key) for key in ['concerns', 'wins', 'opportunities']):
                raise ValueError("No topics were extracted from documents")

            # Collect evidence
            logging.info("Collecting evidence...")
            evidence = self.evidence_collector.collect_evidence(documents, topics)
            self.save_intermediate_results("03_evidence_collection", evidence)

            # Generate stories
            stories = []
            categories = ['concerns', 'wins', 'opportunities']
            
            for category in categories:
                if topics.get(category):
                    logging.info(f"Generating story for {category}")
                    story = self.story_generator.generate_story(
                        category=category,
                        topics=topics[category],
                        evidence=evidence
                    )
                    if story:
                        stories.append(story)

            # Save final stories
            self.save_intermediate_results("04_final_stories", stories)
            
            # Generate readable output
            self.save_readable_output(stories)
            
            return {
                "status": "success",
                "stories": stories,
                "metadata": {
                    "document_count": len(documents),
                    "generated_stories": len(stories),
                    "categories": categories
                }
            }

        except Exception as e:
            logging.error(f"Error in document processing: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "stories": []
            } 