import json
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms.ollama import Ollama
from app.config import config
import logging
logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM-based filter refinement using LangChain."""

    def __init__(self):
        self.llm = Ollama(
            base_url=config.OLLAMA_URL,
            model=config.OLLAMA_LLM_MODEL,
            temperature=0.15,
            num_gpu=1
        )
        self.setup_chain()

    def setup_chain(self):
        """Setup LangChain prompt and output parser."""

        # Define response schema
        response_schemas = [
            ResponseSchema(
                name="reducedFilters",
                description="Dictionary with reduced filter selections per category. For hierarchical categories, use nested dictionaries."
            ),
            ResponseSchema(
                name="confidence",
                description="Dictionary with confidence scores (0-1) for each category"
            ),
            ResponseSchema(
                name="reasoning",
                description="Dictionary with reasoning for each category's selection"
            )
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.output_parser.get_format_instructions()

        # Define prompt template
        template = """
            You are an expert filter recommendation system for a Knowledge Management System.
            User Query: "{query}"
            Available filter candidates (top similar filters from embedding search):
            {candidates}

            Your task:
            1. Analyze the user query and understand the intent
            2. From the candidates provided, select the 5-10 MOST RELEVANT filters per category
            3. For hierarchical categories (like domain-competence, domain-speciality), maintain the hierarchical structure
            4. Provide confidence scores (0-1) for each category
            5. Explain your reasoning for each category

            Important rules:
            - Only select filters that are truly relevant to the query
            - Maintain parent-child relationships in hierarchical filters
            - Higher confidence for explicit matches, lower for inferred matches
            - If a category has no relevant filters, omit it

            {format_instructions}

            Return ONLY the JSON output, no additional text."""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["query", "candidates"],
            partial_variables={"format_instructions": format_instructions}
        )

        self.chain = self.prompt | self.llm | self.output_parser

    def refine_filters(self, query: str, candidates: Dict[str, Any], max_per_category: int = config.MAX_FILTERS_PER_CATEGORY) -> Dict[str, Any]:
        """
        Refine filter candidates using LLM.

        Args:
            query: User query
            candidates: Candidate filters from embedding search
            max_per_category: Maximum filters to return per category

        Returns:
            Refined filter selection with confidence and reasoning
        """
        # Format candidates for the prompt
        candidates_str = self._format_candidates(candidates, max_per_category)

        try:
            # Run the chain
            result = self.chain.invoke({
                "query": query,
                "candidates": candidates_str
            })
            # logger.info(f"   ✅ LLM refinement result: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ LLM refinement error: {e}")
            # Fallback: return top candidates without LLM refinement
            return self._fallback_selection(candidates, max_per_category)

    def _format_candidates(self, candidates: Dict[str, Any], max_per_category: int) -> str:
        """Format candidates for prompt."""
        formatted = {}

        for category, items in candidates.items():
            if not items:
                continue

            # Take top items based on score
            top_items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)[:max_per_category * 2]

            formatted[category] = []
            for item in top_items:
                value = item.get('value', {})
                subcategory = item.get('subcategory')

                formatted_item = {
                    'name': value.get('name', ''),
                    'description': value.get('description', ''),
                    'score': round(item.get('score', 0), 3)
                }

                if subcategory:
                    formatted_item['subcategory'] = subcategory

                formatted[category].append(formatted_item)

        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _fallback_selection(self, candidates: Dict[str, Any], max_per_category: int) -> Dict[str, Any]:
        """Fallback selection if LLM fails."""
        reduced_filters = {}
        confidence = {}
        reasoning = {}

        for category, items in candidates.items():
            if not items:
                continue

            # Sort by score and take top items
            top_items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)[:max_per_category]

            # Check if hierarchical
            has_subcategory = any(item.get('subcategory') for item in top_items)

            if has_subcategory:
                # Group by subcategory
                grouped = {}
                for item in top_items:
                    subcategory = item.get('subcategory', 'Other')
                    if subcategory not in grouped:
                        grouped[subcategory] = []
                    grouped[subcategory].append(item['value'])
                reduced_filters[category] = grouped
            else:
                # Simple list
                reduced_filters[category] = [item['value'] for item in top_items]

            # Calculate average confidence
            avg_score = sum(item.get('score', 0) for item in top_items) / len(top_items)
            confidence[category] = round(avg_score, 2)
            reasoning[category] = f"Selected based on embedding similarity (fallback mode)"

        return {
            'reducedFilters': reduced_filters,
            'confidence': confidence,
            'reasoning': reasoning
        }