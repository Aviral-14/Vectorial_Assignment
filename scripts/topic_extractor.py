from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
from typing import List, Dict
import tiktoken

class TopicExtractor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=api_key,
            model="gpt-4"
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 6000
        self.batch_size = 10  # Process 10 documents per batch

    def _initial_summary(self, content: str) -> str:
        """Create initial focused summary preserving key information"""
        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract only the most important information that could be used in product stories."),
                ("user", """Summarize this content focusing ONLY on:
                - Specific metrics and data points
                - Direct user quotes and feedback
                - Technical issues with impact
                - Feature requests with context
                  
                - Success metrics and outcomes
                - User pain points and their frequency
                - Positive feedback patterns
                - Success metrics and outcomes
                
                Exclude any general descriptions or non-essential information.
                Requirements:
                - Preserve exact numbers and statistics
                - Keep direct quotes that show impact
                - Maintain specific examples
                - Include frequency of issues/feedback
                - Note patterns across users
                
                Be concise but preserve all important details.
                Be extremely concise but precise.
                
                Content: {content}""")
            ])

            response = self.llm.invoke(summary_prompt.format_messages(content=content))
            return response.content

        except Exception as e:
            logging.error(f"Error in initial summary: {str(e)}")
            return ""

    def _batch_synthesis(self, summaries: List[str]) -> str:
        """Synthesize a batch of summaries into a coherent overview"""
        try:
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "Combine these summaries into a coherent overview that preserves all key information."),
                ("user", """Synthesize these summaries into a single overview:
                - Group related information
                - Combine similar metrics
                - Maintain specific quotes and evidence
                - Highlight patterns and trends
                - Preserve unique insights
                
                Summaries: {summaries}""")
            ])

            response = self.llm.invoke(synthesis_prompt.format_messages(summaries="\n\n".join(summaries)))
            return response.content

        except Exception as e:
            logging.error(f"Error in batch synthesis: {str(e)}")
            return ""

    def extract_topics(self, documents: Dict[str, str]) -> Dict:
        """Extract topics using two-stage summarization"""
        try:
            # Stage 1: Initial Summaries
            initial_summaries = {}
            for filename, content in documents.items():
                summary = self._initial_summary(content)
                if summary:
                    initial_summaries[filename] = summary
                    logging.info(f"Created initial summary for {filename}")

            # Stage 2: Batch Processing and Synthesis
            final_summaries = []
            doc_items = list(initial_summaries.items())
            
            for i in range(0, len(doc_items), self.batch_size):
                batch = dict(doc_items[i:i + self.batch_size])
                
                # Synthesize batch
                batch_text = "\n\n".join([f"File: {k}\n{v}" for k, v in batch.items()])
                batch_synthesis = self._batch_synthesis(batch_text)
                
                # Check tokens
                tokens = self._count_tokens(batch_synthesis)
                if tokens <= self.max_tokens:
                    final_summaries.append(batch_synthesis)
                    logging.info(f"Processed batch {i//self.batch_size + 1}")

            # Final topic extraction from synthesized summaries
            combined_text = "\n\n".join(final_summaries)
            
            topic_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract key product insights both positive and negative from these summaries."),
                ("user", """Analyze this information and identify:

                CONCERNS:
                - List major and even minor issues with supporting evidence 
                - Include impact and metrics where available

                WINS:
                - List significant successes with metrics
                - Include user feedback and outcomes

                OPPORTUNITIES:
                - List potential improvements
                - Include user needs and business value

                Use specific data points and quotes when available.
                
                Information to analyze: {documents}""")
            ])

            response = self.llm.invoke(topic_prompt.format_messages(documents=combined_text))
            return self._parse_topics(response.content)

        except Exception as e:
            logging.error(f"Error in topic extraction: {str(e)}")
            return self._get_empty_result()

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _get_empty_result(self) -> Dict:
        return {
            'concerns': [],
            'wins': [],
            'opportunities': [],
            'quotes': {
                'concerns': [],
                'wins': [],
                'opportunities': []
            }
        }

    def _parse_topics(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            result = self._get_empty_result()
            
            current_section = None
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.upper() in ['CONCERNS:', 'WINS:', 'OPPORTUNITIES:']:
                    current_section = line.lower().replace(':', '')
                elif line.startswith('- ') and current_section:
                    topic = line[2:].strip()
                    result[current_section].append(topic)
                elif line.startswith('> ') and current_section:
                    quote = line[2:].strip()
                    result['quotes'][current_section].append(quote)
            
            return result

        except Exception as e:
            logging.error(f"Error parsing topics: {str(e)}")
            return self._get_empty_result() 