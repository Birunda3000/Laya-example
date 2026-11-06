"""
Schema definition and state builder for Laya Anonymization Decision Engine.
"""

from typing import Any, Dict

QUESTIONS: Dict[str, Any] = {
    "should_anonymize": {
        "type": "noul",
        "instructions": (
            "Given the dataset context (domain, purpose, row and column counts) and the column's "
            "metadata, statistical quantiles, and PII detector flags, should this column "
            "be anonymized to protect personal, sensitive, or re-identifiable information?"
        ),
        "criteria": {
            "true": (
                "The column contains direct PII (e.g., CPF, CNPJ, SSN, email, phone, person name) "
                "or quasi-identifiers (dates, locations, ZIP/CEP codes, demographic or high-cardinality "
                "attributes) that pose a privacy or re-identification risk in this dataset's domain and purpose."
            ),
            "false": (
                "The column contains non-sensitive data, generic system metrics, aggregated values, "
                "random/technical identifiers, or public attributes that do not identify individuals in this context."
            ),
        },
        "labels": {
            "true": "A",
            "false": "B",
        },
    }
}


def build_column_state(
    dataset_meta: Dict[str, Any],
    column_data: Dict[str, Any],
    drop_semantic_type: bool = False,
) -> Dict[str, Any]:
    """
    Builds a sanitized state representation for a single column.
    
    Strictly filters out data leakage targets:
      - 'should_anonymize' (target label)
      - 'reason' (human justification)
      - 'dataset_id' (split identifier)
      
    Optionally drops 'semantic_type' when drop_semantic_type=True.
    """
    col_dict: Dict[str, Any] = {
        "name": column_data.get("name"),
        "description": column_data.get("description"),
        "data_type": column_data.get("data_type"),
    }
    
    if not drop_semantic_type and "semantic_type" in column_data:
        col_dict["semantic_type"] = column_data.get("semantic_type")
        
    col_dict["null_percentage"] = column_data.get("null_percentage")
    col_dict["cardinality"] = column_data.get("cardinality")
    
    if "uniqueness_ratio" in column_data:
        col_dict["uniqueness_ratio"] = column_data.get("uniqueness_ratio")
        
    col_dict["quantiles"] = column_data.get("quantiles")
    col_dict["detectors"] = column_data.get("detectors")

    state = {
        "dataset": {
            "dataset_name": dataset_meta.get("dataset_name"),
            "domain": dataset_meta.get("domain"),
            "purpose": dataset_meta.get("purpose"),
            "num_rows": dataset_meta.get("num_rows"),
            "num_cols": dataset_meta.get("num_cols"),
        },
        "column": col_dict,
    }
    return state
