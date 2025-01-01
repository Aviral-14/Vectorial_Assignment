from typing import List, Dict
from .topic_extractor import TopicExtractor
from .evidence_collector import EvidenceCollector
from .story_generator import StoryGenerator
import logging
import json
from pathlib import Path
from datetime import datetime

class AutoDocumentProcessor:
    def __init__(self, api_key: str):
        self.topic_extractor = TopicExtractor(api_key=api_key)
        self.evidence_collector = EvidenceCollector(api_key=api_key)
        self.story_generator = StoryGenerator(api_key=api_key)
        self.setup_output_directory()

    def setup_output_directory(self):
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/run_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {self.output_dir}")

    def save_intermediate_results(self, stage: str, data: dict):
        """Save intermediate results for analysis"""
        filepath = self.output_dir / f"{stage}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {stage} to {filepath}")

    def process_documents(self, documents: Dict[str, str]) -> List[Dict]:
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
            
            return stories

        except Exception as e:
            logging.error(f"Error in document processing: {str(e)}")
            raise

    def save_readable_output(self, stories: List[Dict]):
        """Save stories in human-readable format"""
        try:
            output = []
            for story in stories:
                output.append(f"{'='*80}")
                output.append(f"Category: {story['category'].upper()}")
                output.append("\nTopics:")
                for topic in story['topics']:
                    output.append(f"- {topic}")
                
                output.append("\nStory:")
                output.append(story['story'])
                
                output.append("\nEvidence Used:")
                for topic, evidence in story['evidence_used'].items():
                    output.append(f"\nFor topic: {topic}")
                    if evidence['quotes']:
                        output.append("Quotes:")
                        for quote in evidence['quotes']:
                            output.append(f"- {quote['text']}")
                            output.append(f"  Source: {quote['source']}")
                    if evidence['support']:
                        output.append("Supporting Evidence:")
                        for support in evidence['support']:
                            output.append(f"- {support['text']}")
                            output.append(f"  Source: {support['source']}")
                
                output.append(f"\n{'='*80}\n")

            # Save as text file
            with open(self.output_dir / "05_stories.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(output))

            # Save as markdown for better formatting
            with open(self.output_dir / "06_stories.md", 'w', encoding='utf-8') as f:
                f.write("\n".join(output))

        except Exception as e:
            logging.error(f"Error saving readable output: {str(e)}") 