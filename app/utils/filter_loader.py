import json
from typing import Dict, Any

class FilterLoader:
    """Load and manage filter configuration data."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.filter_data = None

    def load_filters(self) -> Dict[str, Any]:
        """
        Load filter configuration from JSON file.

        Returns:
            Dictionary containing filter configuration
        """
        if self.filter_data is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.filter_data = json.load(f)
        return self.filter_data

    def get_all_filter_values(self) -> Dict[str, list]:
        """
        Extract all filter values organized by category.

        Returns:
            Dictionary with categories as keys and lists of values
        """
        filters = self.load_filters()
        all_values = {}

        for category, values in filters.items():
            if isinstance(values, list):
                # Simple list of values (tool, environnement-domain)
                all_values[category] = values
            elif isinstance(values, dict):
                # Hierarchical structure (environnement-context, domain-competence, etc.)
                all_values[category] = values

        return all_values

    def flatten_filter_values(self) -> list:
        """
        Flatten all filter values into a single list for embedding generation.

        Returns:
            List of tuples (category, subcategory, value_dict)
        """
        filters = self.load_filters()
        flattened = []

        for category, values in filters.items():
            if isinstance(values, list):
                # Simple list (tool, environnement-domain)
                for value in values:
                    if isinstance(value, dict):
                        flattened.append((category, None, value))
                    else:
                        flattened.append((category, None, {"name": value, "description": ""}))
            elif isinstance(values, dict):
                # Hierarchical structure
                for subcategory, subvalues in values.items():
                    for value in subvalues:
                        if isinstance(value, dict):
                            flattened.append((category, subcategory, value))
                        else:
                            flattened.append((category, subcategory, {"name": value, "description": ""}))

        return flattened