"""
Data collection module for TLR4 binding affinity prediction.

This module provides the DataCollector class for:
- Querying ChEMBL for TLR4 and related TLR binding data
- Querying PubChem BioAssay for dose-response data
- Standardizing activity values to kcal/mol
- Merging data from multiple sources with duplicate detection

Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3
"""

import time
import logging
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import CompoundRecord, AffinitySource, FunctionalClass
from ..config.api_config import get_api_config, APIConfig

logger = logging.getLogger(__name__)


# Physical constants for thermodynamic calculations
R = 1.987e-3  # Gas constant in kcal/(mol·K)
T = 298.15    # Standard temperature in Kelvin (25°C)


# TLR target IDs in ChEMBL
TLR_TARGETS = {
    "TLR4": ["CHEMBL5896", "CHEMBL2047"],
    "TLR2": ["CHEMBL5372"],
    "TLR7": ["CHEMBL5600"],
    "TLR8": ["CHEMBL5608"],
    "TLR9": ["CHEMBL5842"],
}

# PubChem BioAssay IDs for TLR4
PUBCHEM_ASSAYS = {
    "TLR4": [1053197, 588834, 651635],
}


@dataclass
class ActivityMeasurement:
    """Represents a single activity measurement before standardization."""
    value: float
    unit: str
    activity_type: str  # IC50, EC50, Ki, Kd
    
    def to_molar(self) -> float:
        """Convert activity value to molar concentration."""
        unit_conversions = {
            'nM': 1e-9,
            'uM': 1e-6,
            'mM': 1e-3,
            'pM': 1e-12,
            'M': 1.0,
            'nm': 1e-9,
            'um': 1e-6,
            'mm': 1e-3,
            'pm': 1e-12,
        }
        
        multiplier = unit_conversions.get(self.unit, None)
        if multiplier is None:
            logger.warning(f"Unknown unit: {self.unit}, assuming nM")
            multiplier = 1e-9
        
        return self.value * multiplier


class DataCollector:
    """
    Collects and standardizes TLR binding data from multiple sources.
    
    This class provides methods to:
    - Query ChEMBL for compounds with binding data for TLR targets
    - Query PubChem BioAssay for dose-response data
    - Convert all activity values to consistent units (kcal/mol)
    - Merge data from multiple sources with duplicate detection
    
    Attributes:
        config: API configuration for rate limiting and timeouts
        session: Requests session with retry logic
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize the DataCollector.
        
        Args:
            config: Optional API configuration. Uses global config if not provided.
        """
        self.config = config or get_api_config()
        self.session = self._create_session()
        self._last_chembl_request = 0.0
        self._last_pubchem_request = 0.0
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.chembl_max_retries,
            backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit_chembl(self) -> None:
        """Apply rate limiting for ChEMBL API."""
        elapsed = time.time() - self._last_chembl_request
        if elapsed < self.config.chembl_rate_limit:
            time.sleep(self.config.chembl_rate_limit - elapsed)
        self._last_chembl_request = time.time()
    
    def _rate_limit_pubchem(self) -> None:
        """Apply rate limiting for PubChem API."""
        elapsed = time.time() - self._last_pubchem_request
        if elapsed < self.config.pubchem_rate_limit:
            time.sleep(self.config.pubchem_rate_limit - elapsed)
        self._last_pubchem_request = time.time()
    
    def query_chembl(
        self,
        target_ids: Optional[List[str]] = None,
        activity_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query ChEMBL for compounds with binding data for specified targets.
        
        Extracts compounds with IC50, EC50, Ki, Kd binding data for the
        specified TLR targets. Handles pagination and rate limiting.
        
        Args:
            target_ids: List of ChEMBL target IDs. Defaults to TLR4 targets.
            activity_types: List of activity types to include. 
                           Defaults to ['IC50', 'EC50', 'Ki', 'Kd'].
        
        Returns:
            DataFrame with columns: smiles, compound_id, target_id, target_name,
            activity_value, activity_unit, activity_type, assay_description
        
        Requirements: 1.1
        """
        if target_ids is None:
            target_ids = TLR_TARGETS["TLR4"]
        
        if activity_types is None:
            activity_types = ['IC50', 'EC50', 'Ki', 'Kd']
        
        all_records = []
        
        for target_id in target_ids:
            logger.info(f"Querying ChEMBL for target {target_id}")
            records = self._query_chembl_target(target_id, activity_types)
            all_records.extend(records)
            logger.info(f"Retrieved {len(records)} records for {target_id}")
        
        if not all_records:
            logger.warning("No records retrieved from ChEMBL")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        logger.info(f"Total ChEMBL records: {len(df)}")
        
        return df
    
    def _query_chembl_target(
        self,
        target_id: str,
        activity_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Query ChEMBL for a single target with pagination."""
        records = []
        offset = 0
        limit = 1000
        
        while True:
            self._rate_limit_chembl()
            
            # Build query URL for activity endpoint
            url = f"{self.config.chembl_base_url}/activity.json"
            params = {
                'target_chembl_id': target_id,
                'standard_type__in': ','.join(activity_types),
                'limit': limit,
                'offset': offset,
            }
            
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=self.config.get_chembl_headers(),
                    timeout=self.config.chembl_timeout,
                )
                response.raise_for_status()
                data = response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"ChEMBL API error for {target_id}: {e}")
                break
            
            activities = data.get('activities', [])
            if not activities:
                break
            
            for activity in activities:
                # Skip if no SMILES
                smiles = activity.get('canonical_smiles')
                if not smiles:
                    continue
                
                # Skip if no valid activity value
                value = activity.get('standard_value')
                if value is None:
                    continue
                
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                
                record = {
                    'smiles': smiles,
                    'compound_id': activity.get('molecule_chembl_id'),
                    'compound_name': activity.get('molecule_pref_name'),
                    'target_id': target_id,
                    'target_name': activity.get('target_pref_name'),
                    'activity_value': value,
                    'activity_unit': activity.get('standard_units', 'nM'),
                    'activity_type': activity.get('standard_type'),
                    'assay_description': activity.get('assay_description', ''),
                    'assay_type': activity.get('assay_type', ''),
                    'source': AffinitySource.CHEMBL.value,
                }
                records.append(record)
            
            # Check if more pages exist
            if len(activities) < limit:
                break
            
            offset += limit
            logger.debug(f"Fetched {offset} records for {target_id}")
        
        return records
    
    def query_pubchem(
        self,
        assay_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Query PubChem BioAssay for dose-response data.
        
        Extracts dose-response data from relevant TLR4 bioassays.
        
        Args:
            assay_ids: List of PubChem assay IDs. Defaults to TLR4 assays.
        
        Returns:
            DataFrame with columns: smiles, compound_id, assay_id,
            activity_value, activity_unit, activity_type, assay_description
        
        Requirements: 1.2
        """
        if assay_ids is None:
            assay_ids = PUBCHEM_ASSAYS["TLR4"]
        
        all_records = []
        
        for assay_id in assay_ids:
            logger.info(f"Querying PubChem for assay AID {assay_id}")
            records = self._query_pubchem_assay(assay_id)
            all_records.extend(records)
            logger.info(f"Retrieved {len(records)} records for AID {assay_id}")
        
        if not all_records:
            logger.warning("No records retrieved from PubChem")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        logger.info(f"Total PubChem records: {len(df)}")
        
        return df
    
    def _query_pubchem_assay(self, assay_id: int) -> List[Dict[str, Any]]:
        """Query PubChem for a single assay."""
        records = []
        
        # First, get assay description
        self._rate_limit_pubchem()
        assay_desc = self._get_pubchem_assay_description(assay_id)
        
        # Get active compounds from assay
        self._rate_limit_pubchem()
        
        url = f"{self.config.pubchem_base_url}/assay/aid/{assay_id}/cids/JSON"
        params = {'cids_type': 'active'}
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=self.config.get_pubchem_headers(),
                timeout=self.config.pubchem_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PubChem API error for AID {assay_id}: {e}")
            return records
        
        cids = data.get('InformationList', {}).get('Information', [{}])[0].get('CID', [])
        
        if not cids:
            logger.warning(f"No active compounds found for AID {assay_id}")
            return records
        
        # Get SMILES and activity data for compounds in batches
        batch_size = 100
        for i in range(0, len(cids), batch_size):
            batch_cids = cids[i:i + batch_size]
            batch_records = self._get_pubchem_compound_data(
                batch_cids, assay_id, assay_desc
            )
            records.extend(batch_records)
        
        return records
    
    def _get_pubchem_assay_description(self, assay_id: int) -> str:
        """Get assay description from PubChem."""
        url = f"{self.config.pubchem_base_url}/assay/aid/{assay_id}/description/JSON"
        
        try:
            response = self.session.get(
                url,
                headers=self.config.get_pubchem_headers(),
                timeout=self.config.pubchem_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            descriptions = data.get('PC_AssayContainer', [{}])
            if descriptions:
                assay_info = descriptions[0].get('assay', {}).get('descr', {})
                name = assay_info.get('name', '')
                desc = assay_info.get('description', [''])[0] if assay_info.get('description') else ''
                return f"{name}: {desc}"
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not get description for AID {assay_id}: {e}")
        
        return ""
    
    def _get_pubchem_compound_data(
        self,
        cids: List[int],
        assay_id: int,
        assay_desc: str,
    ) -> List[Dict[str, Any]]:
        """Get SMILES and activity data for a batch of compounds."""
        records = []
        
        if not cids:
            return records
        
        self._rate_limit_pubchem()
        
        # Get SMILES for compounds
        cid_str = ','.join(str(c) for c in cids)
        url = f"{self.config.pubchem_base_url}/compound/cid/{cid_str}/property/CanonicalSMILES/JSON"
        
        try:
            response = self.session.get(
                url,
                headers=self.config.get_pubchem_headers(),
                timeout=self.config.pubchem_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PubChem compound API error: {e}")
            return records
        
        properties = data.get('PropertyTable', {}).get('Properties', [])
        
        for prop in properties:
            cid = prop.get('CID')
            smiles = prop.get('CanonicalSMILES')
            
            if not smiles:
                continue
            
            # Get activity data for this compound in this assay
            activity_data = self._get_pubchem_activity(cid, assay_id)
            
            if activity_data:
                record = {
                    'smiles': smiles,
                    'compound_id': f"CID{cid}",
                    'target_id': f"AID{assay_id}",
                    'target_name': 'TLR4',
                    'activity_value': activity_data.get('value'),
                    'activity_unit': activity_data.get('unit', 'uM'),
                    'activity_type': activity_data.get('type', 'IC50'),
                    'assay_description': assay_desc,
                    'assay_type': 'Functional',
                    'source': AffinitySource.PUBCHEM.value,
                }
                records.append(record)
        
        return records
    
    def _get_pubchem_activity(
        self,
        cid: int,
        assay_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Get activity value for a compound in an assay."""
        self._rate_limit_pubchem()
        
        url = f"{self.config.pubchem_base_url}/assay/aid/{assay_id}/cid/{cid}/JSON"
        
        try:
            response = self.session.get(
                url,
                headers=self.config.get_pubchem_headers(),
                timeout=self.config.pubchem_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Could not get activity for CID {cid} in AID {assay_id}: {e}")
            return None
        
        # Parse activity data from response
        try:
            pc_assay = data.get('PC_AssaySubmit', {})
            assay_data = pc_assay.get('assay', {}).get('results', [])
            
            for result in assay_data:
                if result.get('sid') or result.get('cid'):
                    # Look for AC50, IC50, or EC50 values
                    for outcome in result.get('data', []):
                        tid = outcome.get('tid')
                        value = outcome.get('value', {})
                        
                        # Extract numeric value
                        if 'fval' in value:
                            return {
                                'value': value['fval'],
                                'unit': 'uM',
                                'type': 'IC50',
                            }
                        elif 'ival' in value:
                            return {
                                'value': float(value['ival']),
                                'unit': 'uM',
                                'type': 'IC50',
                            }
            
        except (KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error parsing activity data: {e}")
        
        return None

    def standardize_activity(
        self,
        df: pd.DataFrame,
        unit: str = "kcal/mol",
    ) -> pd.DataFrame:
        """
        Convert all activity values to consistent units using ΔG = RT ln(Kd).
        
        Supports conversion from IC50, EC50, Ki, Kd to binding free energy
        in kcal/mol. Uses the Gibbs relationship: ΔG = RT ln(Kd)
        
        Args:
            df: DataFrame with activity_value, activity_unit, activity_type columns
            unit: Target unit for standardization. Currently only "kcal/mol" supported.
        
        Returns:
            DataFrame with additional binding_affinity column in kcal/mol
        
        Requirements: 1.3
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Convert activity values to binding free energy
        binding_affinities = []
        
        for _, row in df.iterrows():
            try:
                measurement = ActivityMeasurement(
                    value=float(row['activity_value']),
                    unit=str(row.get('activity_unit', 'nM')),
                    activity_type=str(row.get('activity_type', 'IC50')),
                )
                
                # Convert to molar concentration
                molar_value = measurement.to_molar()
                
                # Convert to binding free energy using ΔG = RT ln(Kd)
                # For IC50/EC50, we approximate Kd ≈ IC50/2 under certain conditions
                # For Ki/Kd, use directly
                if measurement.activity_type in ['IC50', 'EC50']:
                    # Approximate Kd from IC50 (Cheng-Prusoff approximation)
                    kd_approx = molar_value / 2.0
                else:
                    kd_approx = molar_value
                
                # Avoid log of zero or negative
                if kd_approx <= 0:
                    binding_affinities.append(np.nan)
                    continue
                
                # ΔG = RT ln(Kd)
                delta_g = R * T * math.log(kd_approx)
                binding_affinities.append(delta_g)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Could not convert activity value: {e}")
                binding_affinities.append(np.nan)
        
        df['binding_affinity'] = binding_affinities
        
        # Store original values
        df['original_value'] = df['activity_value']
        df['original_unit'] = df['activity_unit']
        df['original_type'] = df['activity_type']
        
        # Remove rows with invalid binding affinity
        valid_count = df['binding_affinity'].notna().sum()
        logger.info(f"Standardized {valid_count}/{len(df)} activity values to kcal/mol")
        
        return df
    
    def merge_sources(
        self,
        dataframes: List[pd.DataFrame],
        conflict_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Merge data from multiple sources, flagging duplicates and conflicts.
        
        Identifies duplicate compounds by canonical SMILES and flags entries
        with conflicting measurements (difference > threshold) for manual review.
        
        Args:
            dataframes: List of DataFrames to merge
            conflict_threshold: Maximum allowed difference in binding affinity
                               (kcal/mol) before flagging as conflict
        
        Returns:
            Merged DataFrame with has_conflict and conflict_sources columns
        
        Requirements: 1.4
        """
        if not dataframes:
            return pd.DataFrame()
        
        # Filter out empty dataframes
        dataframes = [df for df in dataframes if not df.empty]
        
        if not dataframes:
            return pd.DataFrame()
        
        # Concatenate all dataframes
        merged = pd.concat(dataframes, ignore_index=True)
        
        if merged.empty:
            return merged
        
        # Ensure we have canonical SMILES
        if 'canonical_smiles' not in merged.columns:
            merged['canonical_smiles'] = merged['smiles'].apply(
                self._canonicalize_smiles
            )
        
        # Group by canonical SMILES to find duplicates
        merged['has_conflict'] = False
        merged['conflict_sources'] = [[] for _ in range(len(merged))]
        
        # Find duplicates and check for conflicts
        smiles_groups = merged.groupby('canonical_smiles')
        
        conflict_indices = []
        
        for smiles, group in smiles_groups:
            if len(group) > 1:
                # Check for conflicting measurements
                if 'binding_affinity' in group.columns:
                    affinities = group['binding_affinity'].dropna()
                    
                    if len(affinities) > 1:
                        max_diff = affinities.max() - affinities.min()
                        
                        if max_diff > conflict_threshold:
                            # Flag all entries for this compound as conflicting
                            sources = group['source'].unique().tolist()
                            
                            for idx in group.index:
                                conflict_indices.append(idx)
                                merged.at[idx, 'has_conflict'] = True
                                merged.at[idx, 'conflict_sources'] = sources
        
        # Log conflict statistics
        n_conflicts = len(set(conflict_indices))
        n_unique = merged['canonical_smiles'].nunique()
        logger.info(
            f"Merged {len(merged)} records into {n_unique} unique compounds. "
            f"{n_conflicts} compounds have conflicting measurements."
        )
        
        return merged
    
    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize a SMILES string using RDKit."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except ImportError:
            logger.warning("RDKit not available, using original SMILES")
        except Exception as e:
            logger.warning(f"Could not canonicalize SMILES: {e}")
        
        return smiles
    
    def query_related_tlrs(
        self,
        tlr_types: Optional[List[str]] = None,
        activity_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Collect binding data from related TLR targets for transfer learning.
        
        Extends the collector for TLR2, TLR7, TLR8, TLR9 targets and stores
        target labels to enable target-specific fine-tuning.
        
        Args:
            tlr_types: List of TLR types to query. Defaults to TLR2, TLR7, TLR8, TLR9.
            activity_types: List of activity types to include.
        
        Returns:
            DataFrame with binding data from related TLRs, including target_label column
        
        Requirements: 2.1, 2.2, 2.3
        """
        if tlr_types is None:
            tlr_types = ["TLR2", "TLR7", "TLR8", "TLR9"]
        
        if activity_types is None:
            activity_types = ['IC50', 'EC50', 'Ki', 'Kd']
        
        all_records = []
        
        for tlr_type in tlr_types:
            target_ids = TLR_TARGETS.get(tlr_type, [])
            
            if not target_ids:
                logger.warning(f"No target IDs found for {tlr_type}")
                continue
            
            logger.info(f"Querying ChEMBL for {tlr_type} targets: {target_ids}")
            
            df = self.query_chembl(target_ids, activity_types)
            
            if not df.empty:
                # Add target label for transfer learning
                df['target_label'] = tlr_type
                all_records.append(df)
                logger.info(f"Retrieved {len(df)} records for {tlr_type}")
        
        if not all_records:
            logger.warning("No records retrieved from related TLRs")
            return pd.DataFrame()
        
        # Concatenate all related TLR data
        merged = pd.concat(all_records, ignore_index=True)
        
        # Apply standardization
        merged = self.standardize_activity(merged)
        
        logger.info(
            f"Total related TLR records: {len(merged)} from "
            f"{merged['target_label'].nunique()} TLR types"
        )
        
        return merged
    
    def collect_tlr4_dataset(
        self,
        include_pubchem: bool = True,
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Collect complete TLR4 binding dataset from all sources.
        
        Convenience method that queries ChEMBL and optionally PubChem,
        standardizes activity values, and merges sources.
        
        Args:
            include_pubchem: Whether to include PubChem data
            standardize: Whether to standardize activity values
        
        Returns:
            Complete TLR4 binding dataset
        """
        dataframes = []
        
        # Query ChEMBL
        logger.info("Collecting TLR4 data from ChEMBL...")
        chembl_df = self.query_chembl()
        if not chembl_df.empty:
            if standardize:
                chembl_df = self.standardize_activity(chembl_df)
            dataframes.append(chembl_df)
        
        # Query PubChem
        if include_pubchem:
            logger.info("Collecting TLR4 data from PubChem...")
            pubchem_df = self.query_pubchem()
            if not pubchem_df.empty:
                if standardize:
                    pubchem_df = self.standardize_activity(pubchem_df)
                dataframes.append(pubchem_df)
        
        # Merge sources
        if dataframes:
            merged = self.merge_sources(dataframes)
            return merged
        
        return pd.DataFrame()
    
    def to_compound_records(self, df: pd.DataFrame) -> List[CompoundRecord]:
        """
        Convert DataFrame to list of CompoundRecord objects.
        
        Args:
            df: DataFrame with compound data
        
        Returns:
            List of CompoundRecord objects
        """
        records = []
        
        for _, row in df.iterrows():
            try:
                record = CompoundRecord(
                    smiles=row.get('smiles', ''),
                    canonical_smiles=row.get('canonical_smiles', row.get('smiles', '')),
                    binding_affinity=row.get('binding_affinity', 0.0),
                    affinity_source=row.get('source', AffinitySource.CHEMBL.value),
                    functional_class=row.get('functional_class', FunctionalClass.UNKNOWN.value),
                    assay_type=row.get('assay_type', ''),
                    quality_score=row.get('quality_score', 1.0),
                    is_pains=row.get('is_pains', False),
                    scaffold=row.get('scaffold', ''),
                    compound_id=row.get('compound_id'),
                    compound_name=row.get('compound_name'),
                    target_id=row.get('target_id'),
                    target_name=row.get('target_name'),
                    original_value=row.get('original_value'),
                    original_unit=row.get('original_unit'),
                    original_type=row.get('original_type'),
                    has_conflict=row.get('has_conflict', False),
                    conflict_sources=row.get('conflict_sources', []),
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Could not create CompoundRecord: {e}")
        
        return records


# Utility functions for activity conversion

def kd_to_delta_g(kd: float, temperature: float = T) -> float:
    """
    Convert Kd (in molar) to binding free energy (kcal/mol).
    
    Uses the Gibbs relationship: ΔG = RT ln(Kd)
    
    Args:
        kd: Dissociation constant in molar
        temperature: Temperature in Kelvin (default 298.15 K)
    
    Returns:
        Binding free energy in kcal/mol
    """
    if kd <= 0:
        raise ValueError("Kd must be positive")
    
    return R * temperature * math.log(kd)


def delta_g_to_kd(delta_g: float, temperature: float = T) -> float:
    """
    Convert binding free energy (kcal/mol) to Kd (in molar).
    
    Uses the inverse Gibbs relationship: Kd = exp(ΔG/RT)
    
    Args:
        delta_g: Binding free energy in kcal/mol
        temperature: Temperature in Kelvin (default 298.15 K)
    
    Returns:
        Dissociation constant in molar
    """
    return math.exp(delta_g / (R * temperature))


def ic50_to_ki(ic50: float, ligand_conc: float, kd_ligand: float) -> float:
    """
    Convert IC50 to Ki using Cheng-Prusoff equation.
    
    Ki = IC50 / (1 + [L]/Kd)
    
    Args:
        ic50: IC50 value in molar
        ligand_conc: Concentration of competing ligand in molar
        kd_ligand: Kd of competing ligand in molar
    
    Returns:
        Ki value in molar
    """
    return ic50 / (1 + ligand_conc / kd_ligand)
