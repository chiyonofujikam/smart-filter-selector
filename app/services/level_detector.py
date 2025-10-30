import json
import os
import logging
from typing import Any, Dict, List

from langchain.llms.ollama import Ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate




from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LevelDetector:
    """Service for detecting expertise/proficiency levels from query."""

    def __init__(self):
        self.levels_data = self._load_levels()
        self.llm = Ollama(
            base_url=config.OLLAMA_URL,
            model=config.OLLAMA_LLM_MODEL,
            temperature=0.15,
            num_gpu=1
        )
        self.setup_chain()

    def _load_levels(self) -> Dict[str, List[str]]:
        """Load levels configuration."""
        levels_path = os.path.join('data', 'levels.json')
        try:
            with open(levels_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.info(f"⚠️  Could not load levels.json: {e}")
            return {
                "language-level": ["A1", "A2", "B1", "B2", "C1", "C2", "native"],
                "experience": ["Experts", "Seniors", "Confirmed", "Medium", "Juniors"],
                "tool-expertise": ["basic", "limited", "intermediate", "advanced", "expert"]
            }

    def setup_chain(self):
        """Setup LangChain prompt and output parser for level detection."""

        response_schemas = [
            ResponseSchema(
                name="detectedLevels",
                description="Dictionary with detected level categories and their values"
            ),
            ResponseSchema(
                name="confidence",
                description="Dictionary with confidence scores (0-1) for each detected level"
            ),
            ResponseSchema(
                name="reasoning",
                description="Explanation for why these levels were detected"
            )
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.output_parser.get_format_instructions()

        template = """
            You are an expert at analyzing job queries to detect expertise and proficiency levels.
            User Query: "{query}"

            Available level categories:
            {levels}

            Your task:
            1. Analyze the query for explicit or implicit level indicators
            2. Detect which level categories are relevant (experience, tool-expertise, language-level)
            3. For each relevant category, select the most appropriate level(s)
            4. Provide confidence scores (0-1) for each detection

            Level detection rules:
            - "expert", "specialist", "senior" → experience: Experts/Seniors
            - "junior", "beginner" → experience: Juniors
            - "confirmed", "mid-level" → experience: Confirmed
            - If query mentions tools with "expert in", "mastery of" → tool-expertise: expert/advanced
            - If query mentions tools without level indicators → tool-expertise: intermediate
            - Only detect language-level if languages are explicitly mentioned (e.g., "fluent in French", "native English")
            - If no level is mentioned, infer from context (e.g., "railway engineer" likely Confirmed or Seniors)

            {format_instructions}

            Return ONLY the JSON output.
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["query", "levels"],
            partial_variables={"format_instructions": format_instructions}
        )

        self.chain = self.prompt | self.llm | self.output_parser

    def detect_levels(self, query: str) -> Dict[str, Any]:
        """
            Detect levels from user query.

            Args:
                query: User query

            Returns:
                Dictionary with detected levels, confidence, and reasoning
        """
        try:
            levels_str = json.dumps(self.levels_data, indent=2)

            result = self.chain.invoke({
                "query": query,
                "levels": levels_str
            })

            return result

        except Exception as e:
            logger.info(f"⚠️  Level detection error: {e}")
            return self._fallback_detection(query)

    def _fallback_detection(self, query: str) -> Dict[str, Any]:
        """Fallback level detection using keyword matching."""
        query_lower = query.lower()
        detected_levels = {}
        confidence = {}
        reasoning = {}

        # Experience level detection
        experience_keywords = {
            "Experts": ["expert", "specialist", "highly experienced", "senior expert"],
            "Seniors": ["senior", "experienced", "seasoned"],
            "Confirmed": ["confirmed", "mid-level", "intermediate experience"],
            "Medium": ["medium", "moderate"],
            "Juniors": ["junior", "beginner", "entry-level", "graduate"]
        }

        for level, keywords in experience_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_levels["experience"] = [level]
                confidence["experience"] = 0.7
                reasoning["experience"] = f"Detected '{level}' from keywords in query"
                break

        # Tool expertise detection
        tool_expertise_keywords = {
            "expert": ["expert in", "mastery of", "proficient in", "skilled in"],
            "advanced": ["advanced", "extensive experience"],
            "intermediate": ["experience with", "knowledge of", "familiar with"]
        }

        for level, keywords in tool_expertise_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_levels["tool-expertise"] = [level]
                confidence["tool-expertise"] = 0.6
                reasoning["tool-expertise"] = f"Inferred '{level}' tool expertise from query context"
                break

        # Language level detection (only if languages mentioned)
        language_keywords = ["fluent", "native", "bilingual", "language", "french", "english", "spanish"]
        if any(kw in query_lower for kw in language_keywords):
            if "native" in query_lower or "mother tongue" in query_lower:
                detected_levels["language-level"] = ["native"]
                confidence["language-level"] = 0.8
                reasoning["language-level"] = "Native language detected"
            elif "fluent" in query_lower or "c2" in query_lower:
                detected_levels["language-level"] = ["C2"]
                confidence["language-level"] = 0.7
                reasoning["language-level"] = "Fluent/C2 language level detected"

        # Default: if no experience level detected, infer from job type
        if "experience" not in detected_levels:
            if "engineer" in query_lower or "developer" in query_lower:
                detected_levels["experience"] = ["Confirmed"]
                confidence["experience"] = 0.5
                reasoning["experience"] = "Inferred Confirmed level from professional context"

        return {
            "detectedLevels": detected_levels,
            "confidence": confidence,
            "reasoning": reasoning
        }
