#!/usr/bin/env python3


import json
import logging
import os
import time
import uuid
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("grace.cortex")


class PodID(str):
    """Type alias for Pod identifiers"""
    pass


class IntentID(str):
    """Type alias for Intent identifiers"""
    pass


class IntentStatus(Enum):
    """Status of an intent in the system"""
    REGISTERED = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REVOKED = auto()


class IntentPriority(Enum):
    """Priority levels for intents"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class EthicalCategory(Enum):
    """Categories for ethical policies"""
    PRIVACY = auto()
    FAIRNESS = auto()
    TRANSPARENCY = auto()
    SAFETY = auto()
    AUTONOMY = auto()
    BENEFICENCE = auto()
    JUSTICE = auto()
    ACCOUNTABILITY = auto()
    SUSTAINABILITY = auto()
    DIGNITY = auto()


class GlobalIntentRegistry:
    """
    Registry for managing pod intents across the GRACE system.
    Handles registration, validation, and tracking of intents.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the intent registry
        
        Args:
            storage_path: Path to store intent data
        """
        self.logger = logging.getLogger("grace.cortex.intent_registry")
        self.intents: Dict[IntentID, Dict[str, Any]] = {}
        self.pod_intents: Dict[PodID, Set[IntentID]] = {}
        self.intent_lock = Lock()
        
        # Set up storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "intents"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Intent registry initialized with storage at {self.storage_path}")
        
        # Load existing intents
        self._load_intents()

    def _load_intents(self) -> None:
        """Load intents from storage"""
        try:
            intent_files = list(self.storage_path.glob("*.json"))
            self.logger.info(f"Loading {len(intent_files)} intents from storage")
            
            for intent_file in intent_files:
                try:
                    with open(intent_file, "r") as f:
                        intent_data = json.load(f)
                        intent_id = intent_data.get("id")
                        if not intent_id:
                            self.logger.warning(f"Skipping intent file {intent_file} - missing ID")
                            continue
                        
                        pod_id = intent_data.get("pod_id")
                        if not pod_id:
                            self.logger.warning(f"Skipping intent file {intent_file} - missing pod ID")
                            continue
                        
                        # Convert string status to enum
                        if "status" in intent_data and isinstance(intent_data["status"], str):
                            try:
                                intent_data["status"] = IntentStatus[intent_data["status"]]
                            except KeyError:
                                intent_data["status"] = IntentStatus.REGISTERED
                        
                        # Convert string priority to enum
                        if "priority" in intent_data and isinstance(intent_data["priority"], str):
                            try:
                                intent_data["priority"] = IntentPriority[intent_data["priority"]]
                            except KeyError:
                                intent_data["priority"] = IntentPriority.MEDIUM
                        
                        with self.intent_lock:
                            self.intents[intent_id] = intent_data
                            if pod_id not in self.pod_intents:
                                self.pod_intents[pod_id] = set()
                            self.pod_intents[pod_id].add(intent_id)
                            
                except Exception as e:
                    self.logger.error(f"Error loading intent from {intent_file}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading intents: {str(e)}")

    def _save_intent(self, intent_id: IntentID) -> bool:
        """
        Save an intent to storage
        
        Args:
            intent_id: ID of the intent to save
            
        Returns:
            Success status
        """
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    return False
                
                intent_data = self.intents[intent_id].copy()
                
                # Convert enum values to strings for serialization
                if "status" in intent_data and isinstance(intent_data["status"], IntentStatus):
                    intent_data["status"] = intent_data["status"].name
                
                if "priority" in intent_data and isinstance(intent_data["priority"], IntentPriority):
                    intent_data["priority"] = intent_data["priority"].name
                
                file_path = self.storage_path / f"{intent_id}.json"
                with open(file_path, "w") as f:
                    json.dump(intent_data, f, indent=2)
                
                return True
        except Exception as e:
            self.logger.error(f"Error saving intent {intent_id}: {str(e)}")
            return False

    def register_intent(self, pod_id: PodID, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new intent from a pod
        
        Args:
            pod_id: ID of the pod registering the intent
            intent_data: Intent details
            
        Returns:
            Registered intent data
        """
        try:
            # Generate a unique ID if not provided
            intent_id = intent_data.get("id", str(uuid.uuid4()))
            
            # Create intent record
            timestamp = datetime.utcnow().isoformat()
            
            # Set default priority if not specified
            priority = intent_data.get("priority", IntentPriority.MEDIUM)
            if isinstance(priority, str):
                try:
                    priority = IntentPriority[priority]
                except KeyError:
                    priority = IntentPriority.MEDIUM
            
            intent_record = {
                "id": intent_id,
                "pod_id": pod_id,
                "description": intent_data.get("description", ""),
                "parameters": intent_data.get("parameters", {}),
                "dependencies": intent_data.get("dependencies", []),
                "priority": priority,
                "status": IntentStatus.REGISTERED,
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": intent_data.get("metadata", {})
            }
            
            # Register the intent
            with self.intent_lock:
                self.intents[intent_id] = intent_record
                
                if pod_id not in self.pod_intents:
                    self.pod_intents[pod_id] = set()
                
                self.pod_intents[pod_id].add(intent_id)
            
            # Save to storage
            self._save_intent(intent_id)
            
            self.logger.info(f"Registered intent {intent_id} from pod {pod_id}")
            return intent_record
            
        except Exception as e:
            self.logger.error(f"Error registering intent from pod {pod_id}: {str(e)}")
            raise ValueError(f"Failed to register intent: {str(e)}")

    def update_intent_status(self, intent_id: IntentID, status: IntentStatus, 
                            details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update the status of an intent
        
        Args:
            intent_id: ID of the intent to update
            status: New status
            details: Additional details about the status change
            
        Returns:
            Updated intent data
        """
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    raise ValueError(f"Intent {intent_id} not found")
                
                intent = self.intents[intent_id]
                intent["status"] = status
                intent["updated_at"] = datetime.utcnow().isoformat()
                
                if details:
                    if "status_history" not in intent:
                        intent["status_history"] = []
                    
                    intent["status_history"].append({
                        "status": status,
                        "timestamp": intent["updated_at"],
                        "details": details
                    })
                
                # Save changes
                self._save_intent(intent_id)
                
                self.logger.info(f"Updated intent {intent_id} status to {status.name}")
                return intent
                
        except Exception as e:
            self.logger.error(f"Error updating intent {intent_id} status: {str(e)}")
            raise ValueError(f"Failed to update intent status: {str(e)}")

    def get_intent(self, intent_id: IntentID) -> Dict[str, Any]:
        """
        Get an intent by ID
        
        Args:
            intent_id: ID of the intent to retrieve
            
        Returns:
            Intent data
        """
        with self.intent_lock:
            if intent_id not in self.intents:
                raise ValueError(f"Intent {intent_id} not found")
            
            return self.intents[intent_id].copy()

    def get_pod_intents(self, pod_id: PodID) -> List[Dict[str, Any]]:
        """
        Get all intents registered by a pod
        
        Args:
            pod_id: ID of the pod
            
        Returns:
            List of intent data
        """
        with self.intent_lock:
            if pod_id not in self.pod_intents:
                return []
            
            return [self.intents[intent_id].copy() for intent_id in self.pod_intents[pod_id]]

    def get_all_intents(self) -> List[Dict[str, Any]]:
        """
        Get all registered intents
        
        Returns:
            List of all intent data
        """
        with self.intent_lock:
            return [intent.copy() for intent in self.intents.values()]

    def get_intents_by_status(self, status: IntentStatus) -> List[Dict[str, Any]]:
        """
        Get intents with a specific status
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching intent data
        """
        with self.intent_lock:
            return [intent.copy() for intent in self.intents.values() 
                   if intent.get("status") == status]

    def validate_intent(self, intent_id: IntentID) -> Tuple[bool, str]:
        """
        Validate an intent
        
        Args:
            intent_id: ID of the intent to validate
            
        Returns:
            Validation results
        """
        # Check if intent exists and is valid
        valid, reason = True, ""
        
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    return False, "Intent not found"
                
                intent = self.intents[intent_id]
                
                # Check if pod exists
                pod_id = intent.get("pod_id")
                if not pod_id:
                    return False, "Intent has no associated pod"
                
                # Check dependencies
                dependencies = intent.get("dependencies", [])
                for dep_id in dependencies:
                    if dep_id not in self.intents:
                        return False, f"Dependency {dep_id} not found"
                    
                    dep_status = self.intents[dep_id].get("status")
                    if dep_status != IntentStatus.COMPLETED:
                        return False, f"Dependency {dep_id} is not completed"
                
                # Intent is valid
                return True, ""
                
        except Exception as e:
            self.logger.error(f"Error validating intent {intent_id}: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def revoke_intent(self, intent_id: IntentID, reason: str) -> Dict[str, Any]:
        """
        Revoke an intent
        
        Args:
            intent_id: ID of the intent to revoke
            reason: Reason for revocation
            
        Returns:
            Updated intent data
        """
        return self.update_intent_status(
            intent_id, 
            IntentStatus.REVOKED, 
            {"reason": reason}
        )

    def get_intent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about registered intents
        
        Returns:
            Intent statistics
        """
        with self.intent_lock:
            total = len(self.intents)
            
            # Count by status
            status_counts = {}
            for status in IntentStatus:
                status_counts[status.name] = 0
            
            # Count by priority
            priority_counts = {}
            for priority in IntentPriority:
                priority_counts[priority.name] = 0
            
            # Count by pod
            pod_counts = {pod_id: len(intents) for pod_id, intents in self.pod_intents.items()}
            
            # Calculate counts
            for intent in self.intents.values():
                status = intent.get("status")
                if status:
                    if isinstance(status, IntentStatus):
                        status_counts[status.name] += 1
                    elif isinstance(status, str):
                        status_counts[status] += 1
                
                priority = intent.get("priority")
                if priority:
                    if isinstance(priority, IntentPriority):
                        priority_counts[priority.name] += 1
                    elif isinstance(priority, str):
                        priority_counts[priority] += 1
            
            return {
                "total": total,
                "by_status": status_counts,
                "by_priority": priority_counts,
                "by_pod": pod_counts
            }


class TrustOrchestration:
    """
    Manages trust scores for pods and coordinates trust-based decisions.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the trust orchestration system
        
        Args:
            storage_path: Path to store trust data
        """
        self.logger = logging.getLogger("grace.cortex.trust")
        self.trust_scores: Dict[PodID, Dict[str, Any]] = {}
        self.trust_lock = Lock()
        
        # Set up storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
                    self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "trust"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Trust orchestration initialized with storage at {self.storage_path}")
        
        # Load existing trust scores
        self._load_trust_scores()

    def _load_trust_scores(self) -> None:
        """Load trust scores from storage"""
        try:
            trust_files = list(self.storage_path.glob("*.json"))
            self.logger.info(f"Loading {len(trust_files)} trust records from storage")
            
            for trust_file in trust_files:
                try:
                    with open(trust_file, "r") as f:
                        trust_data = json.load(f)
                        pod_id = trust_data.get("pod_id")
                        if not pod_id:
                            self.logger.warning(f"Skipping trust file {trust_file} - missing pod ID")
                            continue
                        
                        with self.trust_lock:
                            self.trust_scores[pod_id] = trust_data
                            
                except Exception as e:
                    self.logger.error(f"Error loading trust data from {trust_file}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading trust scores: {str(e)}")

    def _save_trust_score(self, pod_id: PodID) -> bool:
        """
        Save a trust score to storage
        
        Args:
            pod_id: ID of the pod
            
        Returns:
            Success status
        """
        try:
            with self.trust_lock:
                if pod_id not in self.trust_scores:
                    return False
                
                trust_data = self.trust_scores[pod_id]
                file_path = self.storage_path / f"{pod_id}.json"
                with open(file_path, "w") as f:
                    json.dump(trust_data, f, indent=2)
                
                return True
        except Exception as e:
            self.logger.error(f"Error saving trust score for pod {pod_id}: {str(e)}")
            return False

    def initialize_trust_score(self, pod_id: PodID, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize trust score for a new pod
        
        Args:
            pod_id: ID of the pod
            metadata: Additional metadata about the pod
            
        Returns:
            Initial trust record
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            trust_record = {
                "pod_id": pod_id,
                "trust_score": 0.5,  # Default initial score (neutral)
                "confidence": 0.3,   # Low initial confidence
                "components": {
                    "history": 0.5,
                    "verification": 0.5,
                    "consistency": 0.5,
                    "context": 0.5,
                    "source": 0.5
                },
                "history": [],
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": metadata or {}
            }
            
            with self.trust_lock:
                self.trust_scores[pod_id] = trust_record
            
            # Save to storage
            self._save_trust_score(pod_id)
            
            self.logger.info(f"Initialized trust score for pod {pod_id}")
            return trust_record
            
        except Exception as e:
            self.logger.error(f"Error initializing trust score for pod {pod_id}: {str(e)}")
            raise ValueError(f"Failed to initialize trust score: {str(e)}")

    def update_trust_score(self, pod_id: PodID, components: Dict[str, float], 
                          reason: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update trust score for a pod
        
        Args:
            pod_id: ID of the pod
            components: Updated component scores
            reason: Reason for the update
            evidence: Evidence supporting the update
            
        Returns:
            Updated trust record
        """
        try:
            with self.trust_lock:
                # Get or initialize trust record
                if pod_id not in self.trust_scores:
                    trust_record = self.initialize_trust_score(pod_id)
                else:
                    trust_record = self.trust_scores[pod_id]
                
                # Update component scores
                for component, score in components.items():
                    if component in trust_record["components"]:
                        # Ensure score is in valid range
                        score = max(0.0, min(1.0, score))
                        trust_record["components"][component] = score
                
                # Calculate new overall score (weighted average)
                weights = {
                    "history": 0.25,
                    "verification": 0.2,
                    "consistency": 0.2,
                    "context": 0.15,
                    "source": 0.2
                }
                
                weighted_sum = 0.0
                weight_total = 0.0
                
                for component, weight in weights.items():
                    if component in trust_record["components"]:
                        weighted_sum += trust_record["components"][component] * weight
                        weight_total += weight
                
                if weight_total > 0:
                    trust_record["trust_score"] = weighted_sum / weight_total
                
                # Increase confidence with each update, up to a maximum
                trust_record["confidence"] = min(0.95, trust_record["confidence"] + 0.05)
                
                # Record update in history
                timestamp = datetime.utcnow().isoformat()
                trust_record["updated_at"] = timestamp
                
                history_entry = {
                    "timestamp": timestamp,
                    "previous_score": trust_record.get("trust_score", 0.5),
                    "new_score": trust_record["trust_score"],
                    "reason": reason,
                    "components": components.copy()
                }
                
                if evidence:
                    history_entry["evidence"] = evidence
                
                trust_record["history"].append(history_entry)
                
                # Limit history size
                if len(trust_record["history"]) > 100:
                    trust_record["history"] = trust_record["history"][-100:]
                
                # Save changes
                self._save_trust_score(pod_id)
                
                self.logger.info(f"Updated trust score for pod {pod_id} to {trust_record['trust_score']:.2f}")
                return trust_record
                
        except Exception as e:
            self.logger.error(f"Error updating trust score for pod {pod_id}: {str(e)}")
            raise ValueError(f"Failed to update trust score: {str(e)}")

    def get_trust_score(self, pod_id: PodID) -> Dict[str, Any]:
        """
        Get trust score for a pod
        
        Args:
            pod_id: ID of the pod
            
        Returns:
            Trust record
        """
        with self.trust_lock:
            if pod_id not in self.trust_scores:
                raise ValueError(f"No trust record found for pod {pod_id}")
            
            return self.trust_scores[pod_id].copy()

    def get_all_trust_records(self) -> Dict[PodID, Dict[str, Any]]:
        """
        Get all trust records
        
        Returns:
            Dictionary of all trust records
        """
        with self.trust_lock:
            return {pod_id: record.copy() for pod_id, record in self.trust_scores.items()}

    def evaluate_trust_threshold(self, pod_id: PodID, threshold: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if a pod's trust score meets a threshold
        
        Args:
            pod_id: ID of the pod
            threshold: Trust threshold (0.0 to 1.0)
            
        Returns:
            Result and trust record
        """
        try:
            with self.trust_lock:
                if pod_id not in self.trust_scores:
                    return False, {"error": f"No trust record found for pod {pod_id}"}
                
                trust_record = self.trust_scores[pod_id].copy()
                trust_score = trust_record["trust_score"]
                
                meets_threshold = trust_score >= threshold
                
                result = {
                    "meets_threshold": meets_threshold,
                    "trust_score": trust_score,
                    "threshold": threshold,
                    "margin": trust_score - threshold,
                    "trust_record": trust_record
                }
                
                return meets_threshold, result
                
        except Exception as e:
            self.logger.error(f"Error evaluating trust threshold for pod {pod_id}: {str(e)}")
            return False, {"error": str(e)}

    def calculate_system_trust(self) -> Dict[str, Any]:
        """
        Calculate overall system trust metrics
        
        Returns:
            System trust metrics
        """
        with self.trust_lock:
            if not self.trust_scores:
                return {
                    "average_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "pod_count": 0,
                    "high_trust_count": 0,
                    "medium_trust_count": 0,
                    "low_trust_count": 0
                }
            
            scores = [record["trust_score"] for record in self.trust_scores.values()]
            
            high_trust = sum(1 for score in scores if score >= 0.7)
            medium_trust = sum(1 for score in scores if 0.4 <= score < 0.7)
            low_trust = sum(1 for score in scores if score < 0.4)
            
            return {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pod_count": len(scores),
                "high_trust_count": high_trust,
                "medium_trust_count": medium_trust,
                "low_trust_count": low_trust
            }


class EthicalFramework:
    """
    Manages and evaluates ethical policies for system actions.
    """

    def __init__(self, policies_path: str = None):
        """
        Initialize the ethical framework
        
        Args:
            policies_path: Path to ethical policies
        """
        self.logger = logging.getLogger("grace.cortex.ethics")
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.policy_lock = Lock()
        
        # Set up storage
        if policies_path:
            self.policies_path = Path(policies_path)
        else:
            self.policies_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "policies"
        
        self.policies_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ethical framework initialized with policies at {self.policies_path}")
        
        # Load existing policies
        self._load_policies()

    def _load_policies(self) -> None:
        """Load ethical policies from storage"""
        try:
            policy_files = list(self.policies_path.glob("*.json"))
            self.logger.info(f"Loading {len(policy_files)} ethical policies from storage")
            
            for policy_file in policy_files:
                try:
                    with open(policy_file, "r") as f:
                        policy_data = json.load(f)
                        policy_id = policy_data.get("id")
                        if not policy_id:
                            self.logger.warning(f"Skipping policy file {policy_file} - missing ID")
                            continue
                        
                        # Convert string categories to enum values
                        if "categories" in policy_data and isinstance(policy_data["categories"], list):
                            categories = []
                            for category in policy_data["categories"]:
                                if isinstance(category, str):
                                    try:
                                        categories.append(EthicalCategory[category])
                                    except KeyError:
                                        self.logger.warning(f"Unknown ethical category: {category}")
                            policy_data["categories"] = categories
                        
                        with self.policy_lock:
                            self.policies[policy_id] = policy_data
                            
                except Exception as e:
                    self.logger.error(f"Error loading policy from {policy_file}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading policies: {str(e)}")

    def _save_policy(self, policy_id: str) -> bool:
        """
        Save a policy to storage
        
        Args:
            policy_id: ID of the policy to save
            
        Returns:
            Success status
        """
        try:
            with self.policy_lock:
                if policy_id not in self.policies:
                    return False
                
                policy_data = self.policies[policy_id].copy()
                
                # Convert enum categories to strings for serialization
                if "categories" in policy_data and isinstance(policy_data["categories"], list):
                    categories = []
                    for category in policy_data["categories"]:
                        if isinstance(category, EthicalCategory):
                            categories.append(category.name)
                        else:
                            categories.append(str(category))
                    policy_data["categories"] = categories
                
                file_path = self.policies_path / f"{policy_id}.json"
                with open(file_path, "w") as f:
                    json.dump(policy_data, f, indent=2)
                
                return True
        except Exception as e:
            self.logger.error(f"Error saving policy {policy_id}: {str(e)}")
            return False

    def add_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new ethical policy
        
        Args:
            policy_data: Policy definition
            
        Returns:
            Added policy
        """
        try:
            # Generate a unique ID if not provided
            policy_id = policy_data.get("id", str(uuid.uuid4()))
            
            # Create policy record
            timestamp = datetime.utcnow().isoformat()
            
            # Convert string categories to enum values
            categories = []
            if "categories" in policy_data and isinstance(policy_data["categories"], list):
                for category in policy_data["categories"]:
                    if isinstance(category, str):
                        try:
                            categories.append(EthicalCategory[category])
                        except KeyError:
                            self.logger.warning(f"Unknown ethical category: {category}")
                    elif isinstance(category, EthicalCategory):
                        categories.append(category)
            
            policy_record = {
                "id": policy_id,
                "name": policy_data.get("name", "Unnamed Policy"),
                "description": policy_data.get("description", ""),
                "categories": categories,
                "rules": policy_data.get("rules", []),
                "created_at": timestamp,
                "updated_at": timestamp,
                "version": policy_data.get("version", "1.0.0"),
                "metadata": policy_data.get("metadata", {})
            }
                        # Add the policy
            with self.policy_lock:
                self.policies[policy_id] = policy_record
            
            # Save to storage
            self._save_policy(policy_id)
            
            self.logger.info(f"Added ethical policy {policy_id}: {policy_record['name']}")
            return policy_record
            
        except Exception as e:
            self.logger.error(f"Error adding ethical policy: {str(e)}")
            raise ValueError(f"Failed to add ethical policy: {str(e)}")

    def get_policy(self, policy_id: str) -> Dict[str, Any]:
        """
        Get a policy by ID
        
        Args:
            policy_id: ID of the policy to retrieve
            
        Returns:
            Policy data
        """
        with self.policy_lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            return self.policies[policy_id].copy()

    def get_all_policies(self) -> List[Dict[str, Any]]:
        """
        Get all ethical policies
        
        Returns:
            List of all policy data
        """
        with self.policy_lock:
            return [policy.copy() for policy in self.policies.values()]

    def get_policies_by_category(self, category: EthicalCategory) -> List[Dict[str, Any]]:
        """
        Get policies by category
        
        Args:
            category: Category to filter by
            
        Returns:
            List of matching policy data
        """
        with self.policy_lock:
            return [
                policy.copy() for policy in self.policies.values()
                if "categories" in policy and category in policy["categories"]
            ]

    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an action against ethical policies
        
        Args:
            action: Action to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            action_type = action.get("type", "")
            action_params = action.get("parameters", {})
            action_context = action.get("context", {})
            
            results = {
                "action": action,
                "compliant": True,
                "policy_evaluations": [],
                "overall_score": 1.0,
                "concerns": []
            }
            
            with self.policy_lock:
                if not self.policies:
                    return results
                
                # Evaluate each policy
                for policy_id, policy in self.policies.items():
                    policy_result = self._evaluate_policy(policy, action)
                    results["policy_evaluations"].append(policy_result)
                    
                    # If any policy is violated, the action is non-compliant
                    if not policy_result["compliant"]:
                        results["compliant"] = False
                        results["concerns"].extend(policy_result["concerns"])
                
                # Calculate overall score (average of policy scores)
                if results["policy_evaluations"]:
                    total_score = sum(eval_result["score"] for eval_result in results["policy_evaluations"])
                    results["overall_score"] = total_score / len(results["policy_evaluations"])
                
                # Remove duplicates from concerns
                results["concerns"] = list({concern["description"]: concern for concern in results["concerns"]}.values())
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error evaluating action against ethical policies: {str(e)}")
            return {
                "action": action,
                "compliant": False,
                "error": str(e),
                "policy_evaluations": [],
                "overall_score": 0.0,
                "concerns": [{"description": f"Evaluation error: {str(e)}", "severity": "high"}]
            }

    def _evaluate_policy(self, policy: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an action against a specific policy
        
        Args:
            policy: Policy to evaluate against
            action: Action to evaluate
            
        Returns:
            Evaluation results for this policy
        """
        policy_name = policy.get("name", "Unnamed Policy")
        policy_id = policy.get("id", "unknown")
        rules = policy.get("rules", [])
        
        result = {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "compliant": True,
            "score": 1.0,
            "concerns": []
        }
        
        if not rules:
            return result
        
        # Evaluate each rule
        rule_scores = []
        for rule in rules:
            rule_result = self._evaluate_rule(rule, action)
            rule_scores.append(rule_result["score"])
            
            if not rule_result["compliant"]:
                result["compliant"] = False
                result["concerns"].append({
                    "rule": rule.get("name", "Unnamed Rule"),
                    "description": rule_result.get("reason", "Rule violated"),
                    "severity": rule.get("severity", "medium")
                })
        
        # Calculate policy score (average of rule scores)
        if rule_scores:
            result["score"] = sum(rule_scores) / len(rule_scores)
        
        return result

    def _evaluate_rule(self, rule: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an action against a specific rule
        
        Args:
            rule: Rule to evaluate against
            action: Action to evaluate
            
        Returns:
            Evaluation results for this rule
        """
        rule_type = rule.get("type", "")
        rule_condition = rule.get("condition", {})
        rule_name = rule.get("name", "Unnamed Rule")
        
        result = {
            "rule_name": rule_name,
            "compliant": True,
            "score": 1.0,
            "reason": ""
        }
        
        try:
            # Different rule types have different evaluation logic
            if rule_type == "parameter_constraint":
                # Check if action parameters meet constraints
                param_name = rule_condition.get("parameter", "")
                constraint = rule_condition.get("constraint", "")
                value = rule_condition.get("value")
                
                if param_name and constraint and action.get("parameters"):
                    param_value = action["parameters"].get(param_name)
                    
                    if constraint == "equals" and param_value != value:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must equal '{value}'"
                    
                    elif constraint == "not_equals" and param_value == value:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must not equal '{value}'"
                    
                    elif constraint == "contains" and (not param_value or value not in str(param_value)):
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must contain '{value}'"
                    
                    elif constraint == "not_contains" and param_value and value in str(param_value):
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must not contain '{value}'"
                    
                    elif constraint == "greater_than" and (param_value is None or param_value <= value):
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must be greater than {value}"
                    
                    elif constraint == "less_than" and (param_value is None or param_value >= value):
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Parameter '{param_name}' must be less than {value}"
            
            elif rule_type == "action_type_constraint":
                # Check if action type is allowed/disallowed
                allowed_types = rule_condition.get("allowed_types", [])
                disallowed_types = rule_condition.get("disallowed_types", [])
                action_type = action.get("type", "")
                
                if allowed_types and action_type not in allowed_types:
                    result["compliant"] = False
                    result["score"] = 0.0
                    result["reason"] = f"Action type '{action_type}' is not in allowed types: {', '.join(allowed_types)}"
                
                if disallowed_types and action_type in disallowed_types:
                    result["compliant"] = False
                    result["score"] = 0.0
                    result["reason"] = f"Action type '{action_type}' is in disallowed types: {', '.join(disallowed_types)}"
            
            elif rule_type == "context_constraint":
                # Check if action context meets constraints
                context_key = rule_condition.get("key", "")
                constraint = rule_condition.get("constraint", "")
                value = rule_condition.get("value")
                
                if context_key and constraint and action.get("context"):
                    context_value = action["context"].get(context_key)
                    
                    if constraint == "equals" and context_value != value:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Context '{context_key}' must equal '{value}'"
                    
                    elif constraint == "not_equals" and context_value == value:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Context '{context_key}' must not equal '{value}'"
                    
                    elif constraint == "exists" and context_value is None:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Context must include '{context_key}'"
                    
                    elif constraint == "not_exists" and context_value is not None:
                        result["compliant"] = False
                        result["score"] = 0.0
                        result["reason"] = f"Context must not include '{context_key}'"
            
            # Add more rule types as needed
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule_name}: {str(e)}")
            return {
                "rule_name": rule_name,
                "compliant": False,
                "score": 0.0,
                "reason": f"Rule evaluation error: {str(e)}"
            }


class MemoryVault:
    """
    Manages persistent storage of system experiences and knowledge.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the memory vault
        
        Args:
            storage_path: Path to store memory data
        """
        self.logger = logging.getLogger("grace.cortex.memory")
        self.experiences: List[Dict[str, Any]] = []
        self.memory_lock = Lock()
        
        # Set up storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "memory"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Memory vault initialized with storage at {self.storage_path}")
        
        # Load existing memories
        self._load_memories()

    def _load_memories(self) -> None:
        """Load memories from storage"""
        try:
            memory_files = list(self.storage_path.glob("*.json"))
            self.logger.info(f"Loading memories from {len(memory_files)} files")
            
            for memory_file in memory_files:
                try:
                    with open(memory_file, "r") as f:
                        memories = json.load(f)
                        if isinstance(memories, list):
                            with self.memory_lock:
                                self.experiences.extend(memories)
                        else:
                            self.logger.warning(f"Skipping memory file {memory_file} - not a list")
                            
                except Exception as e:
                    self.logger.error(f"Error loading memories from {memory_file}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading memories: {str(e)}")

    def _save_memories(self) -> bool:
        """
        Save memories to storage
        
        Returns:
            Success status
        """
        try:
            with self.memory_lock:
                # Group experiences by month for better organization
                experiences_by_month = {}
                
                for exp in self.experiences:
                    timestamp = exp.get("timestamp", "")
                    if not timestamp:
                        continue
                    
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        month_key = f"{dt.year}-{dt.month:02d}"
                    except ValueError:
                        month_key = "unknown"
                    
                    if month_key not in experiences_by_month:
                        experiences_by_month[month_key] = []
                    
                    experiences_by_month[month_key].append(exp)
                
                # Save each month's experiences to a separate file
                for month_key, month_experiences in experiences_by_month.items():
                    file_path = self.storage_path / f"memories-{month_key}.json"
                    with open(file_path, "w") as f:
                        json.dump(month_experiences, f, indent=2)
                
                return True
        except Exception as e:
            self.logger.error(f"Error saving memories: {str(e)}")
            return False

    def store_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new experience
        
        Args:
            experience: Experience data to store
            
        Returns:
            Stored experience with metadata
        """
        try:
            # Add metadata
            timestamp = datetime.utcnow().isoformat()
            experience_id = str(uuid.uuid4())
            
            experience_record = {
                "id": experience_id,
                "timestamp": timestamp,
                "data": experience,
                "metadata": {
                    "storage_version": "1.0"
                }
            }
            
            with self.memory_lock:
                self.experiences.append(experience_record)
                
                # Save periodically (every 10 experiences)
                if len(self.experiences) % 10 == 0:
                    self._save_memories()
            
            self.logger.info(f"Stored experience {experience_id}")
            return experience_record
            
        except Exception as e:
            self.logger.error(f"Error storing experience: {str(e)}")
            raise ValueError(f"Failed to store experience: {str(e)}")

    def get_experience(self, experience_id: str) -> Dict[str, Any]:
        """
        Get an experience by ID
        
        Args:
            experience_id: ID of the experience to retrieve
            
        Returns:
            Experience data
        """
        with self.memory_lock:
            for exp in self.experiences:
                if exp.get("id") == experience_id:
                    return exp.copy
                    for exp in self.experiences:
                if exp.get("id") == experience_id:
                    return exp.copy()
            
            raise ValueError(f"Experience {experience_id} not found")

    def search_experiences(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search experiences based on query criteria
        
        Args:
            query: Search criteria
            limit: Maximum number of results
            
        Returns:
            List of matching experiences
        """
        try:
            results = []
            
            with self.memory_lock:
                for exp in self.experiences:
                    match = True
                    
                    # Check each query criterion
                    for key, value in query.items():
                        if key == "time_range":
                            # Time range query
                            start_time = value.get("start")
                            end_time = value.get("end")
                            exp_time = exp.get("timestamp")
                            
                            if not exp_time:
                                match = False
                                break
                            
                            if start_time and exp_time < start_time:
                                match = False
                                break
                            
                            if end_time and exp_time > end_time:
                                match = False
                                break
                        
                        elif key == "data":
                            # Match against experience data
                            exp_data = exp.get("data", {})
                            for data_key, data_value in value.items():
                                if data_key not in exp_data or exp_data[data_key] != data_value:
                                    match = False
                                    break
                        
                        elif key == "text_search":
                            # Simple text search in serialized experience
                            exp_str = json.dumps(exp).lower()
                            if value.lower() not in exp_str:
                                match = False
                                break
                    
                    if match:
                        results.append(exp.copy())
                        if len(results) >= limit:
                            break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching experiences: {str(e)}")
            raise ValueError(f"Failed to search experiences: {str(e)}")

    def get_recent_experiences(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent experiences
        
        Args:
            count: Number of experiences to retrieve
            
        Returns:
            List of recent experiences
        """
        with self.memory_lock:
            # Sort by timestamp (newest first)
            sorted_experiences = sorted(
                self.experiences, 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )
            
            return [exp.copy() for exp in sorted_experiences[:count]]

    def delete_experience(self, experience_id: str) -> bool:
        """
        Delete an experience
        
        Args:
            experience_id: ID of the experience to delete
            
        Returns:
            Success status
        """
        try:
            with self.memory_lock:
                for i, exp in enumerate(self.experiences):
                    if exp.get("id") == experience_id:
                        del self.experiences[i]
                        self._save_memories()
                        self.logger.info(f"Deleted experience {experience_id}")
                        return True
            
            self.logger.warning(f"Experience {experience_id} not found for deletion")
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting experience {experience_id}: {str(e)}")
            return False

    def summarize_experiences(self, time_range: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate a summary of experiences
        
        Args:
            time_range: Optional time range to summarize
            
        Returns:
            Summary data
        """
        try:
            with self.memory_lock:
                if not self.experiences:
                    return {
                        "count": 0,
                        "time_range": {"start": None, "end": None},
                        "categories": {}
                    }
                
                filtered_experiences = self.experiences
                
                # Apply time range filter if provided
                if time_range:
                    start_time = time_range.get("start")
                    end_time = time_range.get("end")
                    
                    if start_time or end_time:
                        filtered_experiences = []
                        for exp in self.experiences:
                            exp_time = exp.get("timestamp", "")
                            if not exp_time:
                                continue
                            
                            if start_time and exp_time < start_time:
                                continue
                            
                            if end_time and exp_time > end_time:
                                continue
                            
                            filtered_experiences.append(exp)
                
                # Get time range of filtered experiences
                timestamps = [exp.get("timestamp", "") for exp in filtered_experiences if exp.get("timestamp")]
                start = min(timestamps) if timestamps else None
                end = max(timestamps) if timestamps else None
                
                # Count experiences by category
                categories = {}
                for exp in filtered_experiences:
                    exp_data = exp.get("data", {})
                    category = exp_data.get("category", "uncategorized")
                    
                    if category not in categories:
                        categories[category] = 0
                    
                    categories[category] += 1
                
                return {
                    "count": len(filtered_experiences),
                    "time_range": {"start": start, "end": end},
                    "categories": categories
                }
                
        except Exception as e:
            self.logger.error(f"Error summarizing experiences: {str(e)}")
            return {
                "error": str(e),
                "count": 0,
                "time_range": {"start": None, "end": None},
                "categories": {}
            }


class CentralCortex:
    """
    Central coordination module for system-wide decision making and orchestration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the central cortex
        
        Args:
            config: Configuration options
        """
        self.logger = logging.getLogger("grace.cortex")
        self.config = config or {}
        
        # Initialize components
        self.trust_orchestrator = TrustOrchestrator(
            storage_path=self.config.get("trust_storage_path")
        )
        
        self.ethical_framework = EthicalFramework(
            policies_path=self.config.get("policies_path")
        )
        
        self.memory_vault = MemoryVault(
            storage_path=self.config.get("memory_storage_path")
        )
        
        # Set up event bus
        self.event_bus = EventBus()
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info("Central Cortex initialized")

    def _register_event_handlers(self):
        """Register handlers for system events"""
        self.event_bus.subscribe("pod.registered", self._handle_pod_registered)
        self.event_bus.subscribe("pod.unregistered", self._handle_pod_unregistered)
        self.event_bus.subscribe("pod.action_requested", self._handle_action_requested)
        self.event_bus.subscribe("pod.action_completed", self._handle_action_completed)
        self.event_bus.subscribe("pod.error", self._handle_pod_error)

    def _handle_pod_registered(self, event_data: Dict[str, Any]):
        """
        Handle pod registration event
        
        Args:
            event_data: Event data
        """
        try:
            pod_id = event_data.get("pod_id")
            pod_metadata = event_data.get("metadata", {})
            
            if not pod_id:
                self.logger.warning("Received pod registration event without pod ID")
                return
            
            # Initialize trust score for the pod
            self.trust_orchestrator.initialize_trust_score(pod_id, pod_metadata)
            
            # Store the registration as an experience
            self.memory_vault.store_experience({
                "type": "pod_registration",
                "pod_id": pod_id,
                "metadata": pod_metadata,
                "category": "system_events"
            })
            
            self.logger.info(f"Handled registration for pod {pod_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling pod registration: {str(e)}")

    def _handle_pod_unregistered(self, event_data: Dict[str, Any]):
        """
        Handle pod unregistration event
        
        Args:
            event_data: Event data
        """
        try:
            pod_id = event_data.get("pod_id")
            reason = event_data.get("reason", "Unknown")
            
            if not pod_id:
                self.logger.warning("Received pod unregistration event without pod ID")
                return
            
            # Store the unregistration as an experience
            self.memory_vault.store_experience({
                "type": "pod_unregistration",
                "pod_id": pod_id,
                "reason": reason,
                "category": "system_events"
            })
            
            self.logger.info(f"Handled unregistration for pod {pod_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling pod unregistration: {str(e)}")

    def _handle_action_requested(self, event_data: Dict[str, Any]):
        """
        Handle action request event
        
        Args:
            event_data: Event data
        """
        try:
            pod_id = event_data.get("pod_id")
            action = event_data.get("action", {})
            request_id = event_data.get("request_id")
            
            if not pod_id or not action:
                self.logger.warning("Received incomplete action request event")
                return
            
            # Evaluate the action against ethical policies
            evaluation = self.ethical_framework.evaluate_action(action)
            
            # Check if the pod has sufficient trust
            trust_threshold = 0.4  # Configurable threshold
            meets_threshold, trust_result = self.trust_orchestrator.evaluate_trust_threshold(
                pod_id, trust_threshold
            )
            
            # Determine if the action is approved
            approved = evaluation["compliant"] and meets_threshold
            
            # Store the request and decision as an experience
            self.memory_vault.store_experience({
                "type": "action_request",
                "pod_id": pod_id,
                "action": action,
                "request_id": request_id,
                "evaluation": evaluation,
                "trust_result": trust_result,
                "approved": approved,
                "category": "action_requests"
            })
            
            # Publish decision event
            self.event_bus.publish("cortex.action_decision", {
                "pod_id": pod_id,
                "request_id": request_id,
                "approved": approved,
                "reasons": {
                    "ethical": evaluation["concerns"] if not evaluation["compliant"] else [],
                    "trust": [] if meets_threshold else ["Insufficient trust level"]
                }
            })
            
            self.logger.info(f"Processed action request from pod {pod_id}: approved={approved}")
            
        except Exception as e:
            self.logger.error(f"Error handling action request: {str(e)}")

    def _handle_action_completed(self, event_data: Dict[str, Any]):
        """
        Handle action completion event
        
        Args:
            event_data: Event data
        """
        try:
            pod_id = event_data.get("pod_id")
            action = event_data.get("action", {})
            result = event_data.get("result", {})
            request_id = event_data.get("request_id")
            
            if not pod_id or not action:
                self.logger.warning("Received incomplete action completion event")
                return
            
            # Update trust score based on action result
            success = result.get("success", False)
            
            # Simple trust adjustment based on success/failure
            if success:
                # Increase trust for successful actions
                self.trust_orchestrator.update_trust_score(
                    pod_id,
                    {"history": 0.6, "consistency": 0.6},
                    "Successful action completion",
                    {"action": action, "result": result}
                )
            else:
                # Decrease trust for failed actions
                self.trust_orchestrator.update_trust_score(
                    pod_id,
                    {"history": 0.4, "consistency": 0.4},
                    "Failed action completion",
                    {"action": action, "result": result}
                )
            
            # Store the completion as an experience
            self.memory_vault.store_experience({
                "type": "action_completion",
                "pod_id": pod_id,
                "action": action,
                "result": result,
                "request_id": request_id,
                "category": "action_results"
            })
            
            self.logger.info(f"Handled action completion from pod {pod_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling action completion: {str(e)}")

    def _handle_pod_error(self, event_data: Dict[str, Any]):
        """
        Handle pod error event
        
        Args:
            event_data: Event data
        """
        try:
            pod_id = event_data.get("pod_id")
            error = event_data.get("error", {})
            severity = event_data.get("severity", "medium")
            
            if not pod_id or not error:
                self.logger.warning("Received incomplete pod error event")
                return
            
            # Update trust score based on error severity
            if severity == "high":
                # Significant trust decrease for severe errors
                self.trust_orchestrator.update_trust_score(
                    pod_id,
                    {"history": 0.3, "consistency": 0.3},
                    "High severity error",
                    {"error": error}
                )
            elif severity == "medium":
                # Moderate trust decrease for medium errors
                self.trust_orchestrator.update_trust_score(
                    pod_id,
                    {"history": 0.4, "consistency": 0.4},
                    "Medium severity error",
                    {"error": error}
                )
            else:
                # Minor trust decrease for low severity errors
                self.trust_orchestrator.update_trust_score(
                    pod_id,
                    {"history": 0.45, "consistency": 0.45},
                    "Low severity error",
                    {"error": error}
                )
            
            # Store the error as an experience
            self.memory_vault.store_experience({
                "type": "pod_error",
                "pod_id": pod_id,
                "error": error,
                "severity": severity,
                "category": "system_errors"
            })
            
            self.logger.info(f"Handled error from pod {pod_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling pod error: {str(e)}")

    def evaluate_action(self, pod_id: PodID, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an action for compliance and approval
        
        Args:
            pod_id: ID of the pod requesting the action
            action: Action to evaluate
            
                Returns:
            Evaluation results
        """
        try:
            # Evaluate the action against ethical policies
            ethical_evaluation = self.ethical_framework.evaluate_action(action)
            
            # Check if the pod has sufficient trust
            trust_threshold = 0.4  # Configurable threshold
            meets_threshold, trust_result = self.trust_orchestrator.evaluate_trust_threshold(
                pod_id, trust_threshold
            )
            
            # Determine if the action is approved
            approved = ethical_evaluation["compliant"] and meets_threshold
            
            return {
                "approved": approved,
                "ethical_evaluation": ethical_evaluation,
                "trust_evaluation": trust_result,
                "reasons": {
                    "ethical": ethical_evaluation["concerns"] if not ethical_evaluation["compliant"] else [],
                    "trust": [] if meets_threshold else ["Insufficient trust level"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating action: {str(e)}")
            return {
                "approved": False,
                "error": str(e),
                "reasons": {
                    "system": [f"Evaluation error: {str(e)}"]
                }
            }

    def get_pod_trust(self, pod_id: PodID) -> Dict[str, Any]:
        """
        Get trust information for a pod
        
        Args:
            pod_id: ID of the pod
            
        Returns:
            Trust information
        """
        try:
            return self.trust_orchestrator.get_trust_score(pod_id)
        except ValueError:
            # If no trust record exists, initialize one
            return self.trust_orchestrator.initialize_trust_score(pod_id)

    def get_system_trust_metrics(self) -> Dict[str, Any]:
        """
        Get system-wide trust metrics
        
        Returns:
            System trust metrics
        """
        return self.trust_orchestrator.calculate_system_trust()

    def get_ethical_policies(self, category: Optional[EthicalCategory] = None) -> List[Dict[str, Any]]:
        """
        Get ethical policies
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of policies
        """
        if category:
            return self.ethical_framework.get_policies_by_category(category)
        else:
            return self.ethical_framework.get_all_policies()

    def add_ethical_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new ethical policy
        
        Args:
            policy_data: Policy definition
            
        Returns:
            Added policy
        """
        return self.ethical_framework.add_policy(policy_data)

    def search_memories(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search system memories
        
        Args:
            query: Search criteria
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        return self.memory_vault.search_experiences(query, limit)

    def get_recent_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent system memories
        
        Args:
            count: Number of memories to retrieve
            
        Returns:
            List of recent memories
        """
        return self.memory_vault.get_recent_experiences(count)

    def store_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory
        
        Args:
            memory_data: Memory data to store
            
        Returns:
            Stored memory with metadata
        """
        return self.memory_vault.store_experience(memory_data)

    def summarize_system_state(self) -> Dict[str, Any]:
        """
        Generate a summary of the current system state
        
        Returns:
            System state summary
        """
        try:
            # Get trust metrics
            trust_metrics = self.trust_orchestrator.calculate_system_trust()
            
            # Get recent memories summary
            # Use last 24 hours
            end_time = datetime.utcnow().isoformat()
            start_time = (datetime.utcnow() - timedelta(days=1)).isoformat()
            
            memory_summary = self.memory_vault.summarize_experiences({
                "start": start_time,
                "end": end_time
            })
            
            # Count policies by category
            policies = self.ethical_framework.get_all_policies()
            policy_categories = {}
            
            for policy in policies:
                categories = policy.get("categories", [])
                for category in categories:
                    category_name = category.name if isinstance(category, EthicalCategory) else str(category)
                    if category_name not in policy_categories:
                        policy_categories[category_name] = 0
                    
                    policy_categories[category_name] += 1
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "trust_metrics": trust_metrics,
                "memory_metrics": memory_summary,
                "policy_metrics": {
                    "total_policies": len(policies),
                    "categories": policy_categories
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating system state summary: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Initialize logging
def setup_logging():
    """Configure logging for the cortex module"""
    logger = logging.getLogger("grace.cortex")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Main function for standalone operation
def main():
    """Main function when running as a standalone module"""
    logger = setup_logging()
    logger.info("Starting Central Cortex")
    
    # Create and initialize the cortex
    cortex = CentralCortex()
    
    # Add some default ethical policies
    privacy_policy = {
        "name": "Data Privacy Policy",
        "description": "Ensures proper handling of sensitive user data",
        "categories": ["PRIVACY", "SECURITY"],
        "rules": [
            {
                "name": "No PII in Logs",
                "type": "parameter_constraint",
                "condition": {
                    "parameter": "log_content",
                    "constraint": "not_contains",
                    "value": "PII"
                },
                "severity": "high"
            },
            {
                "name": "Encryption Required",
                "type": "context_constraint",
                "condition": {
                    "key": "encryption_enabled",
                    "constraint": "equals",
                    "value": True
                },
                "severity": "high"
            }
        ],
        "version": "1.0.0"
    }
    
    safety_policy = {
        "name": "System Safety Policy",
        "description": "Prevents actions that could harm system integrity",
        "categories": ["SAFETY", "SECURITY"],
        "rules": [
            {
                "name": "No Critical Resource Deletion",
                "type": "action_type_constraint",
                "condition": {
                    "disallowed_types": ["delete_critical_resource", "system_shutdown"]
                },
                "severity": "high"
            }
        ],
        "version": "1.0.0"
    }
    
    cortex.add_ethical_policy(privacy_policy)
    cortex.add_ethical_policy(safety_policy)
    
    logger.info("Added default ethical policies")
    
    # Print system state
    state = cortex.summarize_system_state()
    logger.info(f"System state: {json.dumps(state, indent=2)}")
    
    logger.info("Central Cortex is running")
    
    # In a real application, we would start a service here
    # For this example, we'll just keep the process running
    try:
        while True:
            time.sleep(60)
            state = cortex.summarize_system_state()
            logger.info(f"System state update: {json.dumps(state, indent=2)}")
    except KeyboardInterrupt:
        logger.info("Shutting down Central Cortex")


if __name__ == "__main__":
    main()
"""
Central Cortex API Interface

Provides a REST API for interacting with the Central Cortex.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Body, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.cortex_intent import CentralCortex, EthicalCategory, PodID


# Initialize logging
logger = logging.getLogger("grace.cortex.api")
logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Grace Central Cortex API",
    description="API for interacting with the Grace Central Cortex",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Central Cortex
cortex = CentralCortex()


# Pydantic models for API
class ActionRequest(BaseModel):
    pod_id: str
    action: Dict[str, Any]


class PolicyData(BaseModel):
    name: str
    description: str
    categories: List[str]
    rules: List[Dict[str, Any]]
    version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryData(BaseModel):
    data: Dict[str, Any]
    category: str = "general"


class SearchQuery(BaseModel):
    criteria: Dict[str, Any]
    limit: int = 10


# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Grace Central Cortex API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    state = cortex.summarize_system_state()
    return {
        "status": "healthy",
        "system_state": state
    }


# Trust endpoints
@app.get("/trust/{pod_id}")
async def get_pod_trust(pod_id: str):
    """Get trust information for a pod"""
    try:
        trust_info = cortex.get_pod_trust(pod_id)
        return trust_info
    except Exception as e:
        logger.error(f"Error getting trust for pod {pod_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trust")
async def get_system_trust():
    """Get system-wide trust metrics"""
    try:
        trust_metrics = cortex.get_system_trust_metrics()
        return trust_metrics
    except Exception as e:
        logger.error(f"Error getting system trust metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Action evaluation endpoints
@app.post("/actions/evaluate")
async def evaluate_action(request: ActionRequest):
    """Evaluate an action for compliance and approval"""
    try:
        evaluation = cortex.evaluate_action(request.pod_id, request.action)
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Ethical policy endpoints
@app.get("/policies")
async def get_policies(category: Optional[str] = None):
    """Get ethical policies"""
    try:
        if category:
            try:
                category_enum = EthicalCategory[category.upper()]
                policies = cortex.get_ethical_policies(category_enum)
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        else:
            policies = cortex.get_ethical_policies()
        
        return policies
    except Exception as e:
        logger.error(f"Error getting policies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/policies")
async def add_policy(policy: PolicyData):
    """Add a new ethical policy"""
    try:
        # Convert category strings to enum values
        categories = []
        for category in policy.categories:
            try:
                categories.append(EthicalCategory[category.upper()])
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        policy_dict = policy.dict()
        policy_dict["categories"] = categories
        
        added_policy = cortex.add_ethical_policy(policy_dict)
        return added_policy
    except Exception as e:
        logger.error(f"Error adding policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory endpoints
@app.get("/memories")
async def get_memories(count: int = 10):
    """Get recent memories"""
    try:
        memories = cortex.get_recent_memories(count)
        return memories
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search")
async def search_memories(query: SearchQuery):
    """Search memories"""
    try:
        results = cortex.search_memories(query.criteria, query.limit)
        return results
    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories")
async def store_memory(memory: MemoryData):
    """Store a new memory"""
    try:
        stored_memory = cortex.store_memory(memory.dict())
        return stored_memory
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# System state endpoint
@app.get("/system/state")
async def get_system_state():
    """Get system state summary"""
    try:
        state = cortex.summarize_system_state()
        return state
    except Exception as e:
        logger.error(f"Error getting system state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Main function to run the API server
def main():
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
"""
Central Cortex Tests

Simple tests for the Central Cortex functionality.
"""

import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from core.cortex_intent import (
    CentralCortex, TrustOrchestrator, EthicalFramework, 
    MemoryVault, EthicalCategory, PodID
)


class TestTrustOrchestrator(unittest.TestCase):
    """Tests for the TrustOrchestrator class"""
    
    def setUp(self):
        # Use in-memory storage for tests
        self.trust_orchestrator = TrustOrchestrator(storage_path="test_data/trust")
    
    def test_initialize_trust_score(self):
        """Test initializing a trust score"""
        pod_id = "test-pod-1"
        trust_record = self.trust_orchestrator.initialize_trust_score(pod_id)
        
        self.assertEqual(trust_record["pod_id"], pod_id)
        self.assertEqual(trust_record["trust_score"], 0.5)  # Default initial score
        self.assertEqual(trust_record["confidence"], 0.3)   # Default initial confidence
        
        # Verify components
        self.assertIn("components", trust_record)
        self.assertIn("history", trust_record["components"])
        self.assertIn("verification", trust_record["components"])
    
    def test_update_trust_score(self):
        """Test updating a trust score"""
        pod_id = "test-pod-2"
        
        # Initialize trust score
        self.trust_orchestrator.initialize_trust_score(pod_id)
        
        # Update trust score
        updated_record = self.trust_orchestrator.update_trust_score(
            pod_id,
            {"history": 0.8, "verification": 0.7},
            "Test update",
            {"test": "evidence"}
        )
        
        # Verify update
        self.assertGreater(updated_record["trust_score"], 0.5)  # Score should increase
        self.assertEqual(updated_record["components"]["history"], 0.8)
        self.assertEqual(updated_record["components"]["verification"], 0.7)
        
        # Verify history
        self.assertEqual(len(updated_record["history"]), 1)
        self.assertEqual(updated_record["history"][0]["reason"], "Test update")
    
    def test_evaluate_trust_threshold(self):
        """Test evaluating trust threshold"""
        pod_id = "test-pod-3"
        
        # Initialize trust score
        self.trust_orchestrator.initialize_trust_score(pod_id)
        
        # Test threshold evaluation
        meets_threshold, result = self.trust_orchestrator.evaluate_trust_threshold(pod_id, 0.4)
        self.assertTrue(meets_threshold)  # Default score 0.5 > threshold 0.4
        
        meets_threshold, result = self.trust_orchestrator.evaluate_trust_threshold(pod_id, 0.6)
        self.assertFalse(meets_threshold)  # Default score 0.5 < threshold 0.6


class TestEthicalFramework(unittest.TestCase):
    """Tests for the EthicalFramework class"""
    
    def setUp(self):
        # Use in-memory storage for tests
        self.ethical_framework = EthicalFramework(policies_path="test_data/policies")
    
    def test_add_policy(self):
        """Test adding a policy"""
        policy_data = {
            "name": "Test Policy",
            "description": "A test policy",
            "categories": ["PRIVACY", "SECURITY"],
            "rules": [
                {
                    "name": "Test Rule",
                    "type": "parameter_constraint",
                    "condition": {
                        "parameter": "test_param",
                        "constraint": "equals",
                        "value": "test_value"
                    },
                                        "severity": "medium"
                }
            ],
            "version": "1.0.0"
        }
        
        # Convert string categories to enum
        policy_data["categories"] = [EthicalCategory[cat] for cat in policy_data["categories"]]
        
        # Add policy
        added_policy = self.ethical_framework.add_policy(policy_data)
        
        # Verify policy was added
        self.assertIsNotNone(added_policy["id"])
        self.assertEqual(added_policy["name"], "Test Policy")
        
        # Get policy by ID
        policy_id = added_policy["id"]
        retrieved_policy = self.ethical_framework.get_policy(policy_id)
        
        self.assertEqual(retrieved_policy["name"], "Test Policy")
        self.assertEqual(len(retrieved_policy["rules"]), 1)
    
    def test_evaluate_action(self):
        """Test evaluating an action against policies"""
        # Add a test policy
        policy_data = {
            "name": "Parameter Test Policy",
            "description": "Tests parameter constraints",
            "categories": [EthicalCategory.SAFETY],
            "rules": [
                {
                    "name": "Value Must Match",
                    "type": "parameter_constraint",
                    "condition": {
                        "parameter": "target",
                        "constraint": "equals",
                        "value": "allowed_value"
                    },
                    "severity": "high"
                }
            ],
            "version": "1.0.0"
        }
        
        self.ethical_framework.add_policy(policy_data)
        
        # Test compliant action
        compliant_action = {
            "type": "test_action",
            "parameters": {
                "target": "allowed_value",
                "other_param": "something"
            }
        }
        
        result = self.ethical_framework.evaluate_action(compliant_action)
        self.assertTrue(result["compliant"])
        self.assertGreaterEqual(result["overall_score"], 0.9)
        
        # Test non-compliant action
        non_compliant_action = {
            "type": "test_action",
            "parameters": {
                "target": "disallowed_value",
                "other_param": "something"
            }
        }
        
        result = self.ethical_framework.evaluate_action(non_compliant_action)
        self.assertFalse(result["compliant"])
        self.assertLess(result["overall_score"], 0.5)
        self.assertGreaterEqual(len(result["concerns"]), 1)


class TestMemoryVault(unittest.TestCase):
    """Tests for the MemoryVault class"""
    
    def setUp(self):
        # Use in-memory storage for tests
        self.memory_vault = MemoryVault(storage_path="test_data/memory")
    
    def test_store_experience(self):
        """Test storing an experience"""
        experience_data = {
            "type": "test_experience",
            "details": {
                "source": "test",
                "importance": "medium"
            },
            "category": "test_category"
        }
        
        # Store experience
        stored_exp = self.memory_vault.store_experience(experience_data)
        
        # Verify stored experience
        self.assertIsNotNone(stored_exp["id"])
        self.assertIsNotNone(stored_exp["timestamp"])
        self.assertEqual(stored_exp["data"], experience_data)
    
    def test_search_experiences(self):
        """Test searching experiences"""
        # Store some test experiences
        for i in range(5):
            self.memory_vault.store_experience({
                "type": "test_search",
                "index": i,
                "category": "search_test"
            })
        
        # Search by type
        results = self.memory_vault.search_experiences({"data": {"type": "test_search"}})
        self.assertEqual(len(results), 5)
        
        # Search by text
        results = self.memory_vault.search_experiences({"text_search": "search_test"})
        self.assertGreaterEqual(len(results), 1)
    
    def test_get_recent_experiences(self):
        """Test getting recent experiences"""
        # Store some test experiences
        for i in range(3):
            self.memory_vault.store_experience({
                "type": "test_recent",
                "index": i,
                "category": "recent_test"
            })
        
        # Get recent experiences
        results = self.memory_vault.get_recent_experiences(2)
        self.assertEqual(len(results), 2)
        
        # Verify order (newest first)
        self.assertEqual(results[0]["data"]["index"], 2)
        self.assertEqual(results[1]["data"]["index"], 1)


class TestCentralCortex(unittest.TestCase):
    """Tests for the CentralCortex class"""
    
    def setUp(self):
        # Mock the components for testing
        self.mock_trust = MagicMock(spec=TrustOrchestrator)
        self.mock_ethics = MagicMock(spec=EthicalFramework)
        self.mock_memory = MagicMock(spec=MemoryVault)
        
        # Create cortex with mocked components
        self.cortex = CentralCortex()
        self.cortex.trust_orchestrator = self.mock_trust
        self.cortex.ethical_framework = self.mock_ethics
        self.cortex.memory_vault = self.mock_memory
    
    def test_evaluate_action(self):
        """Test evaluating an action"""
        pod_id = "test-pod"
        action = {"type": "test_action", "parameters": {"param1": "value1"}}
        
        # Set up mock returns
        self.mock_ethics.evaluate_action.return_value = {
            "compliant": True,
            "overall_score": 0.9,
            "concerns": []
        }
        
        self.mock_trust.evaluate_trust_threshold.return_value = (
            True,  # Meets threshold
            {"trust_score": 0.7, "confidence": 0.6}
        )
        
        # Evaluate action
        result = self.cortex.evaluate_action(pod_id, action)
        
        # Verify result
        self.assertTrue(result["approved"])
        self.assertIn("ethical_evaluation", result)
        self.assertIn("trust_evaluation", result)
        
        # Verify mocks were called
        self.mock_ethics.evaluate_action.assert_called_once_with(action)
        self.mock_trust.evaluate_trust_threshold.assert_called_once()
    
    def test_get_pod_trust(self):
        """Test getting pod trust"""
        pod_id = "test-pod"
        
        # Set up mock return
        self.mock_trust.get_trust_score.return_value = {
            "pod_id": pod_id,
            "trust_score": 0.7,
            "confidence": 0.6
        }
        
        # Get pod trust
        result = self.cortex.get_pod_trust(pod_id)
        
        # Verify result
        self.assertEqual(result["pod_id"], pod_id)
        self.assertEqual(result["trust_score"], 0.7)
        
        # Verify mock was called
        self.mock_trust.get_trust_score.assert_called_once_with(pod_id)
    
    def test_get_ethical_policies(self):
        """Test getting ethical policies"""
        # Set up mock return
        self.mock_ethics.get_all_policies.return_value = [
            {"id": "policy1", "name": "Policy 1"},
            {"id": "policy2", "name": "Policy 2"}
        ]
        
        # Get policies
        result = self.cortex.get_ethical_policies()
        
        # Verify result
        self.assertEqual(len(result), 2)
        
        # Verify mock was called
        self.mock_ethics.get_all_policies.assert_called_once()
    
    def test_store_memory(self):
        """Test storing a memory"""
        memory_data = {"type": "test_memory", "details": "test"}
        
        # Set up mock return
        self.mock_memory.store_experience.return_value = {
            "id": "memory1",
            "timestamp": "2023-01-01T00:00:00",
            "data": memory_data
        }
        
        # Store memory
        result = self.cortex.store_memory(memory_data)
        
        # Verify result
        self.assertEqual(result["id"], "memory1")
        self.assertEqual(result["data"], memory_data)
        
        # Verify mock was called
        self.mock_memory.store_experience.assert_called_once_with(memory_data)


def main():
    """Run the tests"""
    unittest.main()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.ERROR)
    
    # Run tests
    main()
"""
Central Cortex Example

Demonstrates how to use the Central Cortex.
"""

import json
import logging
import time
from typing import Dict, Any

from core.cortex_intent import (
    CentralCortex, EthicalCategory, PodID
)


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_policies(cortex: CentralCortex):
    """Create sample ethical policies"""
    # Privacy policy
    privacy_policy = {
        "name": "Data Privacy Policy",
        "description": "Ensures proper handling of sensitive user data",
        "categories": [EthicalCategory.PRIVACY, EthicalCategory.SECURITY],
        "rules": [
            {
                "name": "No PII in Logs",
                "type": "parameter_constraint",
                "condition": {
                    "parameter": "log_content",
                    "constraint": "not_contains",
                    "value": "PII"
                },
                "severity": "high"
            },
            {
                "name": "Encryption Required",
                "type": "context_constraint",
                "condition": {
                    "key": "encryption_enabled",
                    "constraint": "equals",
                    "value": True
                },
                "severity": "high"
            }
        ],
        "version": "1.0.0"
    }
    
    # Safety policy
    safety_policy = {
        "name": "System Safety Policy",
        "description": "Prevents actions that could harm system integrity",
        "categories": [EthicalCategory.SAFETY, EthicalCategory.SECURITY],
        "rules": [
            {
                "name": "No Critical Resource Deletion",
                "type": "action_type_constraint",
                "condition": {
                    "disallowed_types": ["delete_critical_resource", "system_shutdown"]
                },
                "severity": "high"
            }
        ],
        "version": "1.0.0"
    }
    
    # Add policies
    cortex.add_ethical_policy(privacy_policy)
    cortex.add_ethical_policy(safety_policy)
    
    print("Added sample ethical policies")


def simulate_pod_actions(cortex: CentralCortex):
    """Simulate pod actions and evaluate them"""
    # Create a pod
    pod_id = "example-pod-1"
    
    # Initialize trust score
    trust_info = cortex.get_pod_trust(pod_id)
    print(f"Initial pod trust: {json.dumps(trust_info, indent=2)}")
    
    # Define some actions
    actions = [
        # Compliant action
        {
            "type": "read_data",
            "parameters": {
                "resource": "public_data",
                "log_content": "Reading public data"
            },
            "context": {
                "encryption_enabled": True,
                "user_authorized": True
            }
        },
        # Non-compliant action (contains PII in logs)
        {
            "type": "read_data",
            "parameters": {
                "resource": "user_profile",
                "log_content": "Reading PII data for user"
            },
            "context": {
                "encryption_enabled": True,
                "user_authorized": True
            }
        },
        # Non-compliant action (disallowed action type)
        {
            "type": "delete_critical_resource",
            "parameters": {
                "resource": "system_config",
                "log_content": "Deleting system configuration"
            },
            "context": {
                "encryption_enabled": True,
                "user_authorized": True
            }
        }
    ]
    
    # Evaluate each action
    for i, action in enumerate(actions):
        print(f"\nEvaluating action {i+1}: {action['type']}")
        
        # Evaluate action
        result = cortex.evaluate_action(pod_id, action)
        
        print(f"Action approved: {result['approved']}")
        
        if not result['approved']:
            print("Reasons for denial:")
            for category, reasons in result['reasons'].items():
                if reasons:
                    print(f"  {category.upper()}:")
                    for reason in reasons:
                        print(f"    - {reason}")
        
        # Simulate action execution and update trust
        if result['approved']:
            # Successful action increases trust
            cortex.trust_orchestrator.update_trust_score(
                pod_id,
                {"history": 0.6, "consistency": 0.6},
                "Successful action execution",
                {"action": action}
            )
            
            # Store memory of successful action
            cortex.store_memory({
                "type": "action_execution",
                "pod_id": pod_id,
                "action": action,
                "result": {"success": True},
                "category": "pod_actions"
            })
            
            print("Action executed successfully, trust increased")
        else:
            # Failed action decreases trust
            cortex.trust_orchestrator.update_trust_score(
                pod_id,
                {"history": 0.4, "consistency": 0.4},
                "Denied action attempt",
                {"action": action, "evaluation": result}
            )
            
            # Store memory of denied action
            cortex.store_memory({
                "type": "action_denial",
                "pod_id": pod_id,
                "action": action,
                "evaluation": result,
                "category": "pod_actions"
            })
            
            print("Action denied, trust decreased")
    
    # Get updated trust score
    trust_info = cortex.get_pod_trust(pod_id)
    print(f"\nUpdated pod trust: {json.dumps(trust_info, indent=2)}")


def explore_system_state(cortex: CentralCortex):
    """Explore the system state"""
    # Get system state
    state = cortex.summarize_system_state()
    print(f"\nSystem state: {json.dumps(state, indent=2)}")
    
    # Get recent memories
    memories = cortex.get_recent_memories(5)
    print(f"\nRecent memories ({len(memories)}):")
    for memory in memories:
        print(f"  - {memory['id']}: {memory['data'].get('
        print(f"  - {memory['id']}: {memory['data'].get('type')} ({memory['timestamp']})")
    
    # Search memories
    search_results = cortex.search_memories({"data": {"category": "pod_actions"}})
    print(f"\nPod action memories ({len(search_results)}):")
    for memory in search_results:
        print(f"  - {memory['id']}: {memory['data'].get('type')} - {memory['data'].get('pod_id')}")
    
    # Get ethical policies
    policies = cortex.get_ethical_policies()
    print(f"\nEthical policies ({len(policies)}):")
    for policy in policies:
        print(f"  - {policy['id']}: {policy['name']} (Categories: {[c.name for c in policy['categories']]})")


def main():
    """Main function"""
    setup_logging()
    print("Central Cortex Example")
    
    # Create Central Cortex
    cortex = CentralCortex()
    
    # Create sample policies
    create_sample_policies(cortex)
    
    # Simulate pod actions
    simulate_pod_actions(cortex)
    
    # Explore system state
    explore_system_state(cortex)
    
    print("\nExample completed")


if __name__ == "__main__":
    main()
# Grace Central Cortex

The Central Cortex is the core decision-making and coordination module for the Grace AI system. It integrates ethical reasoning, trust management, and memory to enable responsible and trustworthy AI behavior.

## Components

The Central Cortex consists of several key components:

### 1. Trust Orchestrator

Manages trust scores for system components (pods) based on their behavior and interactions. Trust scores influence decision-making and permissions within the system.

Features:
- Dynamic trust scoring based on component behavior
- Trust thresholds for action approval
- Trust history tracking
- System-wide trust metrics

### 2. Ethical Framework

Evaluates actions against ethical policies to ensure system behavior aligns with defined ethical standards.

Features:
- Policy-based ethical evaluation
- Multiple policy categories (privacy, safety, fairness, etc.)
- Rule-based constraints on actions
- Ethical compliance scoring

### 3. Memory Vault

Stores and retrieves system experiences to inform decision-making and enable learning from past interactions.

Features:
- Persistent storage of experiences
- Experience categorization
- Search and retrieval capabilities
- Experience summarization

## API

The Central Cortex provides both a programmatic API and a REST API for integration with other system components.

### REST API Endpoints

- `/health` - Health check endpoint
- `/trust/{pod_id}` - Get trust information for a pod
- `/trust` - Get system-wide trust metrics
- `/actions/evaluate` - Evaluate an action for compliance and approval
- `/policies` - Get or add ethical policies
- `/memories` - Get, search, or store memories
- `/system/state` - Get system state summary

## Usage

### Basic Usage

```python
from core.cortex_intent import CentralCortex

# Create Central Cortex
cortex = CentralCortex()

# Evaluate an action
pod_id = "example-pod"
action = {
    "type": "read_data",
    "parameters": {
        "resource": "user_profile",
        "log_content": "Reading user data"
    },
    "context": {
        "encryption_enabled": True,
        "user_authorized": True
    }
}

evaluation = cortex.evaluate_action(pod_id, action)
print(f"Action approved: {evaluation['approved']}")

# Get pod trust information
trust_info = cortex.get_pod_trust(pod_id)
print(f"Pod trust score: {trust_info['trust_score']}")

# Store a memory
cortex.store_memory({
    "type": "user_interaction",
    "details": "User requested information",
    "category": "user_events"
})

# Get system state
state = cortex.summarize_system_state()
print(f"System state: {state}")
```

### Using the Client

```python
from tools.cortex_client import CortexClient

# Create client
client = CortexClient("http://localhost:8000")

# Check health
health = client.get_health()
print(f"API health: {health['status']}")

# Evaluate an action
evaluation = client.evaluate_action("example-pod", {
    "type": "read_data",
    "parameters": {"resource": "public_data"}
})
print(f"Action approved: {evaluation['approved']}")
```

## Running the API Server

```bash
python -m core.cortex_api
```

## Running the Example

```bash
python -m examples.cortex_example
```

## Running Tests

```bash
python -m tests.test_cortex
```

## Configuration

The Central Cortex can be configured with the following options:

```python
config = {
    "trust_storage_path": "data/trust",
    "policies_path": "data/policies",
    "memory_storage_path": "data/memory"
}

cortex = CentralCortex(config)
```

## License

[MIT License](LICENSE)
```

Finally, let's create a simple setup script to make the package installable:

```python:setup.py
from setuptools import setup, find_packages

setup(
    name="grace-cortex",
    version="0.1.0",
    description="Central Cortex for the Grace AI system",
    author="Grace AI Team",
    author_email="info@grace-ai.org",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "requests>=2.26.0",
        "pydantic>=1.8.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)

