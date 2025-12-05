"""
Functional classification module for TLR4 ligands.

This module provides the FunctionalClassifier class for:
- Parsing assay descriptions to extract functional keywords
- Classifying compounds as agonists, antagonists, or unknown
- Prioritizing NF-κB, cytokine, and reporter gene assays

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import re
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import FunctionalClass

logger = logging.getLogger(__name__)


class AssayPriority(int, Enum):
    """Priority levels for different assay types."""
    NF_KB = 1       # Highest priority: NF-κB activation assays
    CYTOKINE = 2    # High priority: Cytokine production assays
    REPORTER = 3    # Medium priority: Reporter gene assays
    BINDING = 4     # Lower priority: Direct binding assays
    OTHER = 5       # Lowest priority: Other assay types


@dataclass
class FunctionalEvidence:
    """Evidence for functional classification from a single assay."""
    classification: str  # "agonist", "antagonist", or "unknown"
    assay_type: AssayPriority
    confidence: float  # 0.0 to 1.0
    keywords_found: List[str]
    assay_description: str


# Keyword patterns for functional classification
# Requirements: 3.1 - Parse assay descriptions for functional keywords
AGONIST_KEYWORDS = [
    r'\bagonist\b',
    r'\bactivat(?:or|ion|ing|es?)\b',
    r'\bstimulat(?:or|ion|ing|es?)\b',
    r'\binducer?\b',
    r'\bpotentiat(?:or|ion|ing|es?)\b',
    r'\benhancer?\b',
    r'\bpositive\s+modulator\b',
]

ANTAGONIST_KEYWORDS = [
    r'\bantagonist\b',
    r'\binhibit(?:or|ion|ing|s)?\b',
    r'\bblock(?:er|ing|s)?\b',
    r'\bsuppress(?:or|ion|ing|es)?\b',
    r'\bnegative\s+modulator\b',
    r'\battenuator?\b',
    r'\bdown-?regulat(?:or|ion|ing|es?)\b',
]

# Assay type patterns for prioritization
# Requirements: 3.2 - Prioritize NF-κB, cytokine, reporter gene assays
NF_KB_PATTERNS = [
    r'\bNF-?[κk]?B\b',
    r'\bnf-?kappa-?b\b',
    r'\bnuclear\s+factor\s+kappa\b',
    r'\bNFKB\b',
    r'\bNF-kB\b',
    r'\bp65\b',
    r'\bRelA\b',
    r'\bI[κk]B\b',
    r'\bIKK\b',
]

CYTOKINE_PATTERNS = [
    r'\bcytokine\b',
    r'\bTNF-?α?\b',
    r'\bIL-?\d+\b',
    r'\binterleukin\b',
    r'\bIFN-?[αβγ]?\b',
    r'\binterferon\b',
    r'\bchemokine\b',
    r'\bCXCL\d+\b',
    r'\bCCL\d+\b',
    r'\bMCP-?\d?\b',
    r'\bRANTES\b',
    r'\bIP-?10\b',
]

REPORTER_PATTERNS = [
    r'\breporter\b',
    r'\bluciferase\b',
    r'\bGFP\b',
    r'\bgreen\s+fluorescent\b',
    r'\bβ-?galactosidase\b',
    r'\blacZ\b',
    r'\bSEAP\b',
]

BINDING_PATTERNS = [
    r'\bbinding\b',
    r'\baffinity\b',
    r'\bKd\b',
    r'\bKi\b',
    r'\bIC50\b',
    r'\bEC50\b',
    r'\bdisplacement\b',
    r'\bcompetition\b',
]


class FunctionalClassifier:
    """
    Classifies TLR4 ligands as agonists, antagonists, or unknown.
    
    This class provides methods to:
    - Parse assay descriptions for functional keywords
    - Classify compounds based on available evidence
    - Prioritize NF-κB, cytokine, and reporter gene assays
    - Flag ambiguous classifications for manual curation
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    
    def __init__(self):
        """Initialize the FunctionalClassifier with compiled regex patterns."""
        # Compile keyword patterns for efficiency
        self._agonist_patterns = [
            re.compile(p, re.IGNORECASE) for p in AGONIST_KEYWORDS
        ]
        self._antagonist_patterns = [
            re.compile(p, re.IGNORECASE) for p in ANTAGONIST_KEYWORDS
        ]
        
        # Compile assay type patterns
        self._nfkb_patterns = [
            re.compile(p, re.IGNORECASE) for p in NF_KB_PATTERNS
        ]
        self._cytokine_patterns = [
            re.compile(p, re.IGNORECASE) for p in CYTOKINE_PATTERNS
        ]
        self._reporter_patterns = [
            re.compile(p, re.IGNORECASE) for p in REPORTER_PATTERNS
        ]
        self._binding_patterns = [
            re.compile(p, re.IGNORECASE) for p in BINDING_PATTERNS
        ]

    def parse_assay_description(self, description: str) -> Optional[str]:
        """
        Extract functional classification from assay description text.
        
        Parses the assay description for functional keywords (agonist, antagonist,
        inhibition, activation) and returns the corresponding classification.
        
        Args:
            description: Assay description text to parse
        
        Returns:
            Functional classification: "agonist", "antagonist", or None if
            no functional keywords are found
        
        Requirements: 3.1
        """
        if not description or not isinstance(description, str):
            return None
        
        description = description.strip()
        if not description:
            return None
        
        # Search for agonist keywords
        agonist_matches = []
        for pattern in self._agonist_patterns:
            matches = pattern.findall(description)
            agonist_matches.extend(matches)
        
        # Search for antagonist keywords
        antagonist_matches = []
        for pattern in self._antagonist_patterns:
            matches = pattern.findall(description)
            antagonist_matches.extend(matches)
        
        # Determine classification based on keyword matches
        has_agonist = len(agonist_matches) > 0
        has_antagonist = len(antagonist_matches) > 0
        
        if has_antagonist and not has_agonist:
            return FunctionalClass.ANTAGONIST.value
        elif has_agonist and not has_antagonist:
            return FunctionalClass.AGONIST.value
        elif has_agonist and has_antagonist:
            # Both keywords present - need context analysis
            # Check which appears more frequently or in more prominent position
            if len(antagonist_matches) > len(agonist_matches):
                return FunctionalClass.ANTAGONIST.value
            elif len(agonist_matches) > len(antagonist_matches):
                return FunctionalClass.AGONIST.value
            else:
                # Equal matches - return None to indicate ambiguity
                return None
        
        return None
    
    def _get_assay_priority(self, description: str) -> AssayPriority:
        """
        Determine the priority level of an assay based on its description.
        
        Prioritizes NF-κB, cytokine, and reporter gene assays as specified
        in Requirements 3.2.
        
        Args:
            description: Assay description text
        
        Returns:
            AssayPriority enum value
        """
        if not description:
            return AssayPriority.OTHER
        
        # Check for NF-κB assays (highest priority)
        for pattern in self._nfkb_patterns:
            if pattern.search(description):
                return AssayPriority.NF_KB
        
        # Check for cytokine assays
        for pattern in self._cytokine_patterns:
            if pattern.search(description):
                return AssayPriority.CYTOKINE
        
        # Check for reporter gene assays
        for pattern in self._reporter_patterns:
            if pattern.search(description):
                return AssayPriority.REPORTER
        
        # Check for binding assays
        for pattern in self._binding_patterns:
            if pattern.search(description):
                return AssayPriority.BINDING
        
        return AssayPriority.OTHER
    
    def _extract_evidence(
        self,
        assay_description: str,
    ) -> FunctionalEvidence:
        """
        Extract functional evidence from a single assay description.
        
        Args:
            assay_description: Assay description text
        
        Returns:
            FunctionalEvidence object with classification and metadata
        """
        classification = self.parse_assay_description(assay_description)
        assay_type = self._get_assay_priority(assay_description)
        
        # Collect all matched keywords
        keywords_found = []
        
        for pattern in self._agonist_patterns:
            keywords_found.extend(pattern.findall(assay_description or ""))
        for pattern in self._antagonist_patterns:
            keywords_found.extend(pattern.findall(assay_description or ""))
        
        # Calculate confidence based on assay type and keyword matches
        confidence = self._calculate_confidence(
            classification,
            assay_type,
            len(keywords_found),
        )
        
        return FunctionalEvidence(
            classification=classification or FunctionalClass.UNKNOWN.value,
            assay_type=assay_type,
            confidence=confidence,
            keywords_found=keywords_found,
            assay_description=assay_description or "",
        )
    
    def _calculate_confidence(
        self,
        classification: Optional[str],
        assay_type: AssayPriority,
        num_keywords: int,
    ) -> float:
        """
        Calculate confidence score for a functional classification.
        
        Args:
            classification: The determined classification
            assay_type: Priority level of the assay
            num_keywords: Number of functional keywords found
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if classification is None:
            return 0.0
        
        # Base confidence from assay type
        assay_confidence = {
            AssayPriority.NF_KB: 0.9,
            AssayPriority.CYTOKINE: 0.85,
            AssayPriority.REPORTER: 0.8,
            AssayPriority.BINDING: 0.6,
            AssayPriority.OTHER: 0.5,
        }
        
        base = assay_confidence.get(assay_type, 0.5)
        
        # Boost confidence based on number of keywords (up to 0.1 bonus)
        keyword_bonus = min(num_keywords * 0.02, 0.1)
        
        return min(base + keyword_bonus, 1.0)

    def classify_compound(self, compound_data: Dict[str, Any]) -> str:
        """
        Assign functional label based on available evidence.
        
        Classifies a compound as agonist, antagonist, or unknown based on
        assay descriptions and other available data. Prioritizes evidence
        from NF-κB, cytokine, and reporter gene assays.
        
        Args:
            compound_data: Dictionary containing compound information with keys:
                - assay_description: str (required)
                - assay_type: str (optional)
                - assay_descriptions: List[str] (optional, for multiple assays)
        
        Returns:
            Functional classification: "agonist", "antagonist", or "unknown"
        
        Requirements: 3.2, 3.3, 3.4
        """
        # Collect all available assay descriptions
        descriptions = []
        
        # Single assay description
        if 'assay_description' in compound_data:
            desc = compound_data['assay_description']
            if desc and isinstance(desc, str):
                descriptions.append(desc)
        
        # Multiple assay descriptions
        if 'assay_descriptions' in compound_data:
            for desc in compound_data.get('assay_descriptions', []):
                if desc and isinstance(desc, str):
                    descriptions.append(desc)
        
        if not descriptions:
            return FunctionalClass.UNKNOWN.value
        
        # Extract evidence from all assay descriptions
        evidence_list: List[FunctionalEvidence] = []
        for desc in descriptions:
            evidence = self._extract_evidence(desc)
            evidence_list.append(evidence)
        
        # Sort evidence by assay priority (lower value = higher priority)
        evidence_list.sort(key=lambda e: (e.assay_type.value, -e.confidence))
        
        # Collect classifications with their priorities
        agonist_evidence: List[FunctionalEvidence] = []
        antagonist_evidence: List[FunctionalEvidence] = []
        
        for evidence in evidence_list:
            if evidence.classification == FunctionalClass.AGONIST.value:
                agonist_evidence.append(evidence)
            elif evidence.classification == FunctionalClass.ANTAGONIST.value:
                antagonist_evidence.append(evidence)
        
        # If no functional evidence found, return unknown
        if not agonist_evidence and not antagonist_evidence:
            return FunctionalClass.UNKNOWN.value
        
        # If only one type of evidence, use it
        if agonist_evidence and not antagonist_evidence:
            return FunctionalClass.AGONIST.value
        if antagonist_evidence and not agonist_evidence:
            return FunctionalClass.ANTAGONIST.value
        
        # Both types of evidence present - use prioritization
        # Requirements: 3.2 - Prioritize NF-κB, cytokine, reporter gene assays
        best_agonist = agonist_evidence[0]  # Already sorted by priority
        best_antagonist = antagonist_evidence[0]
        
        # Compare by assay priority first
        if best_agonist.assay_type.value < best_antagonist.assay_type.value:
            return FunctionalClass.AGONIST.value
        elif best_antagonist.assay_type.value < best_agonist.assay_type.value:
            return FunctionalClass.ANTAGONIST.value
        
        # Same priority - compare by confidence
        if best_agonist.confidence > best_antagonist.confidence:
            return FunctionalClass.AGONIST.value
        elif best_antagonist.confidence > best_agonist.confidence:
            return FunctionalClass.ANTAGONIST.value
        
        # Equal priority and confidence - return unknown (ambiguous)
        # Requirements: 3.3 - Flag ambiguous compounds
        return FunctionalClass.UNKNOWN.value
    
    def flag_ambiguous(self, compound_data: Dict[str, Any]) -> bool:
        """
        Identify compounds requiring manual curation due to ambiguous classification.
        
        A compound is flagged as ambiguous if:
        - It has conflicting evidence (both agonist and antagonist keywords)
        - Evidence comes from low-priority assays only
        - Classification confidence is below threshold
        
        Args:
            compound_data: Dictionary containing compound information
        
        Returns:
            True if compound requires manual curation, False otherwise
        
        Requirements: 3.3
        """
        # Collect all available assay descriptions
        descriptions = []
        
        if 'assay_description' in compound_data:
            desc = compound_data['assay_description']
            if desc and isinstance(desc, str):
                descriptions.append(desc)
        
        if 'assay_descriptions' in compound_data:
            for desc in compound_data.get('assay_descriptions', []):
                if desc and isinstance(desc, str):
                    descriptions.append(desc)
        
        if not descriptions:
            # No assay descriptions - flag for manual review
            return True
        
        # Extract evidence from all descriptions
        evidence_list = [self._extract_evidence(desc) for desc in descriptions]
        
        # Check for conflicting evidence
        has_agonist = any(
            e.classification == FunctionalClass.AGONIST.value 
            for e in evidence_list
        )
        has_antagonist = any(
            e.classification == FunctionalClass.ANTAGONIST.value 
            for e in evidence_list
        )
        
        if has_agonist and has_antagonist:
            # Conflicting evidence - flag for manual review
            return True
        
        # Check if all evidence is from low-priority assays
        all_low_priority = all(
            e.assay_type.value >= AssayPriority.BINDING.value 
            for e in evidence_list
        )
        
        if all_low_priority and (has_agonist or has_antagonist):
            # Only low-priority evidence - flag for review
            return True
        
        # Check confidence threshold
        max_confidence = max(e.confidence for e in evidence_list)
        if max_confidence < 0.6:
            return True
        
        return False
    
    def get_classification_details(
        self,
        compound_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get detailed classification information for a compound.
        
        Returns comprehensive information about the classification including
        all evidence, confidence scores, and ambiguity flags.
        
        Args:
            compound_data: Dictionary containing compound information
        
        Returns:
            Dictionary with classification details
        """
        # Collect all available assay descriptions
        descriptions = []
        
        if 'assay_description' in compound_data:
            desc = compound_data['assay_description']
            if desc and isinstance(desc, str):
                descriptions.append(desc)
        
        if 'assay_descriptions' in compound_data:
            for desc in compound_data.get('assay_descriptions', []):
                if desc and isinstance(desc, str):
                    descriptions.append(desc)
        
        # Extract evidence
        evidence_list = [self._extract_evidence(desc) for desc in descriptions]
        
        # Get final classification
        classification = self.classify_compound(compound_data)
        is_ambiguous = self.flag_ambiguous(compound_data)
        
        # Calculate overall confidence
        if evidence_list:
            relevant_evidence = [
                e for e in evidence_list 
                if e.classification == classification
            ]
            if relevant_evidence:
                max_confidence = max(e.confidence for e in relevant_evidence)
            else:
                max_confidence = 0.0
        else:
            max_confidence = 0.0
        
        return {
            'classification': classification,
            'confidence': max_confidence,
            'is_ambiguous': is_ambiguous,
            'evidence_count': len(evidence_list),
            'agonist_evidence': sum(
                1 for e in evidence_list 
                if e.classification == FunctionalClass.AGONIST.value
            ),
            'antagonist_evidence': sum(
                1 for e in evidence_list 
                if e.classification == FunctionalClass.ANTAGONIST.value
            ),
            'evidence_details': [
                {
                    'classification': e.classification,
                    'assay_type': e.assay_type.name,
                    'confidence': e.confidence,
                    'keywords': e.keywords_found,
                }
                for e in evidence_list
            ],
        }
