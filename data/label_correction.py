"""
Label Auto-Correction Algorithms

Advanced auto-correction system for gravitational wave detection labels using:
- Fuzzy string matching with configurable thresholds
- ML embedding similarity (Sentence Transformers)
- Smart correction strategies combining multiple approaches
- Comprehensive benchmarking and evaluation

Features:
- Fuzzy matching with difflib and SequenceMatcher
- ML embedding similarity using Sentence Transformers
- Cached computations for performance
- Smart correction combining multiple strategies
- Comprehensive benchmarking tools
"""

from typing import Dict, List, Union, Optional, Tuple, Any
import logging
from functools import lru_cache
import numpy as np

from .label_enums import (
    GWSignalType,
    CANONICAL_LABELS,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_ML_THRESHOLD,
    ML_MODEL_CONFIGS,
    normalize_label_value
)

from .label_validation import LabelError

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    from difflib import get_close_matches, SequenceMatcher
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False
    logger.warning("difflib not available - fuzzy matching disabled")

try:
    from sentence_transformers import SentenceTransformer
    ML_EMBEDDING_AVAILABLE = True
except ImportError:
    ML_EMBEDDING_AVAILABLE = False
    SentenceTransformer = None
    logger.warning("sentence-transformers not available - ML embedding disabled")

try:
    import sklearn.metrics.pairwise
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - some ML features disabled")


@lru_cache(maxsize=1000)
def _cached_similarity_score(label: str, canonical_label: str) -> float:
    """
    Cached fuzzy similarity score between two labels.
    
    Args:
        label: Input label string
        canonical_label: Canonical label to compare against
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not FUZZY_MATCHING_AVAILABLE:
        # Exact match fallback
        return 1.0 if label.lower() == canonical_label.lower() else 0.0
    
    return SequenceMatcher(None, label.lower(), canonical_label.lower()).ratio()


@lru_cache(maxsize=500)
def _cached_ml_embedding_similarity(label1: str, label2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Cached ML embedding similarity between two labels.
    
    Args:
        label1: First label string
        label2: Second label string
        model_name: Sentence transformer model name
        
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    if not ML_EMBEDDING_AVAILABLE:
        logger.debug("ML embedding not available, falling back to fuzzy matching")
        return _cached_similarity_score(label1, label2)
    
    try:
        # Load model (cached by sentence_transformers)
        model = SentenceTransformer(model_name)
        
        # Compute embeddings
        embeddings = model.encode([label1, label2])
        
        # Compute cosine similarity
        if SKLEARN_AVAILABLE:
            similarity = sklearn.metrics.pairwise.cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        else:
            # Manual cosine similarity
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm_product = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            similarity = dot_product / norm_product if norm_product > 0 else 0.0
        
        return float(similarity)
        
    except Exception as e:
        logger.warning(f"ML embedding similarity failed: {e}")
        return _cached_similarity_score(label1, label2)


@lru_cache(maxsize=100)
def _cached_canonical_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Cache canonical label embeddings for efficient comparison.
    
    Args:
        model_name: Sentence transformer model name
        
    Returns:
        Array of embeddings for canonical labels, or None if unavailable
    """
    if not ML_EMBEDDING_AVAILABLE:
        return None
    
    try:
        model = SentenceTransformer(model_name)
        canonical_texts = list(CANONICAL_LABELS.values())
        embeddings = model.encode(canonical_texts)
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to cache canonical embeddings: {e}")
        return None


def fuzzy_match_labels(labels: List[str],
                      threshold: float = DEFAULT_FUZZY_THRESHOLD,
                      max_suggestions: int = 3) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find fuzzy matches for labels using string similarity.
    
    Args:
        labels: List of labels to match
        threshold: Minimum similarity threshold (0.0 to 1.0)
        max_suggestions: Maximum number of suggestions per label
        
    Returns:
        Dictionary mapping input labels to list of (match, score) tuples
    """
    if not FUZZY_MATCHING_AVAILABLE:
        logger.warning("Fuzzy matching not available")
        return {}
    
    canonical_texts = list(CANONICAL_LABELS.values())
    suggestions = {}
    
    for label in labels:
        label_suggestions = []
        
        # Get close matches using difflib
        close_matches = get_close_matches(
            label.lower(), 
            [c.lower() for c in canonical_texts],
            n=max_suggestions,
            cutoff=threshold
        )
        
        # Calculate exact similarity scores
        for match in close_matches:
            # Find original canonical form
            canonical = next(c for c in canonical_texts if c.lower() == match)
            score = _cached_similarity_score(label, canonical)
            
            if score >= threshold:
                label_suggestions.append((canonical, score))
        
        # Sort by score (highest first)
        label_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        if label_suggestions:
            suggestions[label] = label_suggestions
    
    return suggestions


def ml_embedding_similarity(label: str,
                           model_name: str = "all-MiniLM-L6-v2",
                           threshold: float = DEFAULT_ML_THRESHOLD) -> List[Tuple[str, float]]:
    """
    Find similar labels using ML embeddings (Sentence Transformers).
    
    Args:
        label: Input label to match
        model_name: Sentence transformer model name
        threshold: Minimum similarity threshold
        
    Returns:
        List of (canonical_label, similarity_score) tuples
    """
    if not ML_EMBEDDING_AVAILABLE:
        logger.warning("ML embedding not available, falling back to fuzzy matching")
        fuzzy_results = fuzzy_match_labels([label], threshold=threshold)
        return fuzzy_results.get(label, [])
    
    try:
        canonical_texts = list(CANONICAL_LABELS.values())
        similarities = []
        
        # Check cached embeddings first
        cached_embeddings = _cached_canonical_embeddings(model_name)
        
        if cached_embeddings is not None:
            # Use cached embeddings for efficiency
            model = SentenceTransformer(model_name)
            label_embedding = model.encode([label])
            
            if SKLEARN_AVAILABLE:
                scores = sklearn.metrics.pairwise.cosine_similarity(
                    label_embedding, cached_embeddings
                )[0]
            else:
                # Manual cosine similarity
                scores = []
                for canonical_emb in cached_embeddings:
                    dot_product = np.dot(label_embedding[0], canonical_emb)
                    norm_product = np.linalg.norm(label_embedding[0]) * np.linalg.norm(canonical_emb)
                    score = dot_product / norm_product if norm_product > 0 else 0.0
                    scores.append(score)
                scores = np.array(scores)
            
            # Create results
            for canonical, score in zip(canonical_texts, scores):
                if score >= threshold:
                    similarities.append((canonical, float(score)))
        else:
            # Fallback to individual comparisons
            for canonical in canonical_texts:
                score = _cached_ml_embedding_similarity(label, canonical, model_name)
                if score >= threshold:
                    similarities.append((canonical, score))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
        
    except Exception as e:
        logger.error(f"ML embedding similarity failed: {e}")
        # Fallback to fuzzy matching
        fuzzy_results = fuzzy_match_labels([label], threshold=threshold)
        return fuzzy_results.get(label, [])


def auto_correct_labels(labels: Union[List[str], List[int]],
                       method: str = 'fuzzy',
                       threshold: float = None,
                       confidence_threshold: float = 0.8) -> Tuple[List[int], Dict[str, Any]]:
    """
    Automatically correct labels using specified method.
    
    Args:
        labels: List of labels to correct
        method: Correction method ('fuzzy', 'ml', 'smart')
        threshold: Similarity threshold (method-specific default if None)
        confidence_threshold: Minimum confidence for auto-correction
        
    Returns:
        Tuple of (corrected_labels, correction_report)
    """
    # Set default thresholds
    if threshold is None:
        threshold = DEFAULT_FUZZY_THRESHOLD if method == 'fuzzy' else DEFAULT_ML_THRESHOLD
    
    corrected = []
    corrections_made = 0
    correction_log = []
    failed_corrections = []
    
    for i, label in enumerate(labels):
        try:
            # Try direct normalization first
            normalized = normalize_label_value(label)
            corrected.append(normalized)
            
            if str(label) != str(normalized):
                correction_log.append({
                    'index': i,
                    'original': label,
                    'corrected': normalized,
                    'method': 'direct_normalization',
                    'confidence': 1.0
                })
                corrections_made += 1
                
        except ValueError:
            # Direct normalization failed, try correction methods
            if isinstance(label, (int, np.integer)):
                # Integer label that's out of range
                corrected.append(0)  # Default to noise
                failed_corrections.append({
                    'index': i,
                    'original': label,
                    'error': 'Invalid integer label',
                    'fallback': 0
                })
                continue
            
            # String label correction
            label_str = str(label)
            suggestions = []
            
            if method == 'fuzzy':
                fuzzy_results = fuzzy_match_labels([label_str], threshold=threshold)
                suggestions = fuzzy_results.get(label_str, [])
                
            elif method == 'ml':
                suggestions = ml_embedding_similarity(label_str, threshold=threshold)
                
            elif method == 'smart':
                # Combine fuzzy and ML approaches
                fuzzy_results = fuzzy_match_labels([label_str], threshold=threshold)
                fuzzy_suggestions = fuzzy_results.get(label_str, [])
                ml_suggestions = ml_embedding_similarity(label_str, threshold=threshold)
                
                # Merge and weight suggestions
                suggestion_scores = {}
                for canonical, score in fuzzy_suggestions:
                    suggestion_scores[canonical] = suggestion_scores.get(canonical, 0) + score * 0.4
                for canonical, score in ml_suggestions:
                    suggestion_scores[canonical] = suggestion_scores.get(canonical, 0) + score * 0.6
                
                suggestions = [(k, v) for k, v in suggestion_scores.items()]
                suggestions.sort(key=lambda x: x[1], reverse=True)
            
            # Apply best suggestion if confidence is high enough
            if suggestions and suggestions[0][1] >= confidence_threshold:
                best_canonical = suggestions[0][0]
                best_score = suggestions[0][1]
                
                # Convert to integer label
                for signal_type, canonical in CANONICAL_LABELS.items():
                    if canonical == best_canonical:
                        corrected.append(int(signal_type))
                        correction_log.append({
                            'index': i,
                            'original': label,
                            'corrected': int(signal_type),
                            'method': method,
                            'confidence': best_score,
                            'canonical': best_canonical
                        })
                        corrections_made += 1
                        break
            else:
                # No confident correction found
                corrected.append(0)  # Default to noise
                failed_corrections.append({
                    'index': i,
                    'original': label,
                    'error': 'No confident correction found',
                    'suggestions': suggestions[:3],  # Top 3 suggestions
                    'fallback': 0
                })
    
    # Compile correction report
    correction_report = {
        'total_labels': len(labels),
        'corrections_made': corrections_made,
        'correction_rate': corrections_made / len(labels) if labels else 0,
        'failed_corrections': len(failed_corrections),
        'method_used': method,
        'threshold_used': threshold,
        'confidence_threshold': confidence_threshold,
        'correction_log': correction_log,
        'failed_corrections': failed_corrections
    }
    
    logger.info(f"Auto-correction completed: {corrections_made}/{len(labels)} labels corrected using {method}")
    
    return corrected, correction_report


def ml_auto_correct_labels(labels: Union[List[str], List[int]],
                          model_name: str = "all-MiniLM-L6-v2",
                          threshold: float = DEFAULT_ML_THRESHOLD) -> Tuple[List[int], Dict[str, Any]]:
    """
    ML-based auto-correction using Sentence Transformers.
    
    Args:
        labels: List of labels to correct
        model_name: Sentence transformer model name
        threshold: Similarity threshold
        
    Returns:
        Tuple of (corrected_labels, correction_report)
    """
    return auto_correct_labels(labels, method='ml', threshold=threshold)


def smart_label_correction(labels: Union[List[str], List[int]],
                          fuzzy_weight: float = 0.4,
                          ml_weight: float = 0.6,
                          confidence_threshold: float = 0.7) -> Tuple[List[int], Dict[str, Any]]:
    """
    Smart label correction combining fuzzy matching and ML embeddings.
    
    Args:
        labels: List of labels to correct
        fuzzy_weight: Weight for fuzzy matching scores
        ml_weight: Weight for ML embedding scores
        confidence_threshold: Minimum confidence for correction
        
    Returns:
        Tuple of (corrected_labels, correction_report)
    """
    corrected = []
    corrections_made = 0
    correction_log = []
    failed_corrections = []
    
    for i, label in enumerate(labels):
        try:
            # Try direct normalization first
            normalized = normalize_label_value(label)
            corrected.append(normalized)
            
            if str(label) != str(normalized):
                correction_log.append({
                    'index': i,
                    'original': label,
                    'corrected': normalized,
                    'method': 'direct_normalization',
                    'confidence': 1.0
                })
                corrections_made += 1
                
        except ValueError:
            # Smart correction for string labels
            if isinstance(label, (int, np.integer)):
                corrected.append(0)
                failed_corrections.append({
                    'index': i,
                    'original': label,
                    'error': 'Invalid integer label',
                    'fallback': 0
                })
                continue
            
            label_str = str(label)
            
            # Get fuzzy suggestions
            fuzzy_results = fuzzy_match_labels([label_str])
            fuzzy_suggestions = fuzzy_results.get(label_str, [])
            
            # Get ML suggestions
            ml_suggestions = ml_embedding_similarity(label_str)
            
            # Combine suggestions with weights
            combined_scores = {}
            
            for canonical, score in fuzzy_suggestions:
                combined_scores[canonical] = score * fuzzy_weight
            
            for canonical, score in ml_suggestions:
                if canonical in combined_scores:
                    combined_scores[canonical] += score * ml_weight
                else:
                    combined_scores[canonical] = score * ml_weight
            
            # Find best suggestion
            if combined_scores:
                best_canonical = max(combined_scores.keys(), key=lambda k: combined_scores[k])
                best_score = combined_scores[best_canonical]
                
                if best_score >= confidence_threshold:
                    # Convert to integer label
                    for signal_type, canonical in CANONICAL_LABELS.items():
                        if canonical == best_canonical:
                            corrected.append(int(signal_type))
                            correction_log.append({
                                'index': i,
                                'original': label,
                                'corrected': int(signal_type),
                                'method': 'smart_combined',
                                'confidence': best_score,
                                'canonical': best_canonical,
                                'fuzzy_score': next((s for c, s in fuzzy_suggestions if c == best_canonical), 0.0),
                                'ml_score': next((s for c, s in ml_suggestions if c == best_canonical), 0.0)
                            })
                            corrections_made += 1
                            break
                else:
                    corrected.append(0)
                    failed_corrections.append({
                        'index': i,
                        'original': label,
                        'error': 'Low confidence',
                        'best_score': best_score,
                        'fallback': 0
                    })
            else:
                corrected.append(0)
                failed_corrections.append({
                    'index': i,
                    'original': label,
                    'error': 'No suggestions found',
                    'fallback': 0
                })
    
    # Compile report
    correction_report = {
        'total_labels': len(labels),
        'corrections_made': corrections_made,
        'correction_rate': corrections_made / len(labels) if labels else 0,
        'failed_corrections': len(failed_corrections),
        'method_used': 'smart_combined',
        'fuzzy_weight': fuzzy_weight,
        'ml_weight': ml_weight,
        'confidence_threshold': confidence_threshold,
        'correction_log': correction_log,
        'failed_corrections': failed_corrections
    }
    
    logger.info(f"Smart correction completed: {corrections_made}/{len(labels)} labels corrected")
    
    return corrected, correction_report


def benchmark_correction_methods(test_labels: List[str],
                               ground_truth: List[int],
                               methods: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different correction methods against ground truth.
    
    Args:
        test_labels: List of corrupted/incorrect labels
        ground_truth: List of correct integer labels
        methods: List of methods to test
        
    Returns:
        Dictionary with performance metrics for each method
    """
    if methods is None:
        methods = ['fuzzy', 'ml', 'smart']
    
    if len(test_labels) != len(ground_truth):
        raise ValueError("test_labels and ground_truth must have same length")
    
    results = {}
    
    for method in methods:
        if method == 'ml' and not ML_EMBEDDING_AVAILABLE:
            logger.warning(f"Skipping {method} - ML embedding not available")
            continue
        
        try:
            corrected, report = auto_correct_labels(test_labels, method=method)
            
            # Calculate accuracy metrics
            correct = sum(1 for pred, true in zip(corrected, ground_truth) if pred == true)
            accuracy = correct / len(ground_truth)
            
            # Calculate precision, recall per class
            class_metrics = {}
            for signal_type in GWSignalType:
                class_id = int(signal_type)
                
                true_positives = sum(1 for pred, true in zip(corrected, ground_truth) 
                                   if pred == class_id and true == class_id)
                false_positives = sum(1 for pred, true in zip(corrected, ground_truth) 
                                    if pred == class_id and true != class_id)
                false_negatives = sum(1 for pred, true in zip(corrected, ground_truth) 
                                    if pred != class_id and true == class_id)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                class_metrics[f'class_{class_id}_precision'] = precision
                class_metrics[f'class_{class_id}_recall'] = recall
                class_metrics[f'class_{class_id}_f1'] = f1
            
            results[method] = {
                'accuracy': accuracy,
                'correction_rate': report['correction_rate'],
                'failed_rate': report['failed_corrections'] / len(test_labels),
                **class_metrics
            }
            
        except Exception as e:
            logger.error(f"Benchmarking failed for method {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


# Export main functions
__all__ = [
    'fuzzy_match_labels',
    'ml_embedding_similarity', 
    'auto_correct_labels',
    'ml_auto_correct_labels',
    'smart_label_correction',
    'benchmark_correction_methods',
    '_cached_similarity_score',
    '_cached_ml_embedding_similarity',
    '_cached_canonical_embeddings'
] 