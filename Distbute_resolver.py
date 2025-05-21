import os
import json
import logging
import time
import uuid
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field

# Mock imports for external dependencies
# Flask mocks
class Flask:
    def __init__(self, name):
        self.name = name
        self.routes = {}
        
    def route(self, path, methods=None):
        if methods is None:
            methods = ['GET']
        
        def decorator(f):
            self.routes[path] = (f, methods)
            return f
        return decorator

class Response:
    def __init__(self, data, status=200, mimetype=None):
        self.data = data
        self.status = status
        self.mimetype = mimetype

class Request:
    def __init__(self, json=None, args=None):
        self.json = json if json else {}
        self.args = args if args else {}

# Werkzeug mocks
class HTTPException(Exception):
    def __init__(self, description=None, response=None):
        self.description = description
        self.response = response

# Cryptography mocks
class Fernet:
    def __init__(self, key):
        self.key = key
    
    @staticmethod
    def generate_key():
        return b'mock_key_for_testing_purposes_only'
    
    def encrypt(self, data):
        if isinstance(data, str):
            data = data.encode()
        return b'encrypted_' + data
    
    def decrypt(self, token):
        if token.startswith(b'encrypted_'):
            return token[len(b'encrypted_'):]
        return token

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30  # seconds
PARLIAMENT_ENDPOINT = os.environ.get('PARLIAMENT_ENDPOINT', 'http://localhost:8080/parliament')
DISPUTE_LOG_PATH = os.environ.get('DISPUTE_LOG_PATH', '/var/log/grace/disputes')

# Ensure log directory exists
os.makedirs(DISPUTE_LOG_PATH, exist_ok=True)

class DisputeStatus(Enum):
    """Status of a dispute resolution process."""
    PENDING = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    FAILED = auto()
    DEFERRED = auto()

class ResolutionStrategy(Enum):
    """Available strategies for resolving disputes."""
    MAJORITY_VOTE = auto()
    WEIGHTED_VOTE = auto()
    ETHICAL_PRIORITY = auto()
    HUMAN_INTERVENTION = auto()
    CONSENSUS_BUILDING = auto()
    SANDBOX_TEST = auto()

@dataclass
class Dispute:
    """Represents a dispute that needs resolution."""
    id: str
    title: str
    description: str
    options: List[Dict[str, Any]]
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: DisputeStatus = DisputeStatus.PENDING
    resolution: Optional[Dict[str, Any]] = None
    strategy: ResolutionStrategy = ResolutionStrategy.MAJORITY_VOTE
    priority: int = 1  # 1-5, with 5 being highest
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dispute to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "options": self.options,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "status": self.status.name,
            "resolution": self.resolution,
            "strategy": self.strategy.name,
            "priority": self.priority,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dispute':
        """Create a Dispute instance from a dictionary."""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            options=data["options"],
            context=data["context"],
            created_at=created_at,
            status=DisputeStatus[data["status"]],
            resolution=data.get("resolution"),
            strategy=ResolutionStrategy[data["strategy"]],
            priority=data["priority"],
            tags=data.get("tags", [])
        )

class ParliamentClient:
    """Client for interacting with Grace's Parliament ethical decision engine."""
    
    def __init__(self, endpoint: str = PARLIAMENT_ENDPOINT, timeout: int = DEFAULT_TIMEOUT):
        self.endpoint = endpoint
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialized Parliament client with session {self.session_id}")
    
    async def submit_dispute(self, dispute: Dispute) -> Dict[str, Any]:
        """Submit a dispute to Parliament for resolution."""
        try:
            # In a real implementation, this would make an HTTP request
            logger.info(f"Submitting dispute {dispute.id} to Parliament")
            
            # Simulate network delay
            await asyncio.sleep(0.5)
            
            # Mock response
            response = {
                "dispute_id": dispute.id,
                "status": "ACCEPTED",
                "estimated_time": 5,  # seconds
                "tracking_id": str(uuid.uuid4())
            }
            
            logger.info(f"Parliament accepted dispute {dispute.id} with tracking {response['tracking_id']}")
            return response
        except Exception as e:
            logger.error(f"Failed to submit dispute to Parliament: {str(e)}")
            raise
    
    async def get_resolution(self, dispute_id: str, tracking_id: str) -> Dict[str, Any]:
        """Get the resolution for a previously submitted dispute."""
        try:
            # In a real implementation, this would make an HTTP request
            logger.info(f"Checking resolution for dispute {dispute_id}")
            
            # Simulate network delay
            await asyncio.sleep(1.0)
            
            # Mock response
            resolution = {
                "dispute_id": dispute_id,
                "status": "RESOLVED",
                "resolution": {
                    "selected_option": 0,  # Index of the selected option
                    "confidence": 0.85,
                    "reasoning": "This option aligns best with system values and ethical guidelines.",
                    "value_alignment": {
                        "system": 0.9,
                        "contributors": 0.8,
                        "businesses": 0.7,
                        "operator": 0.85,
                        "humanity": 0.95
                    }
                },
                "tracking_id": tracking_id
            }
            
            logger.info(f"Received resolution for dispute {dispute_id}")
            return resolution
        except Exception as e:
            logger.error(f"Failed to get resolution from Parliament: {str(e)}")
            raise

class DisputeLogger:
    """Handles logging of disputes and their resolutions."""
    
    def __init__(self, log_dir: str = DISPUTE_LOG_PATH):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Dispute logger initialized with log directory: {self.log_dir}")
    
    def log_dispute(self, dispute: Dispute) -> None:
        """Log a dispute to the filesystem."""
        try:
            dispute_path = os.path.join(self.log_dir, f"{dispute.id}.json")
            with open(dispute_path, 'w') as f:
                json.dump(dispute.to_dict(), f, indent=2)
            logger.info(f"Logged dispute {dispute.id} to {dispute_path}")
        except Exception as e:
            logger.error(f"Failed to log dispute {dispute.id}: {str(e)}")
    
    def get_dispute(self, dispute_id: str) -> Optional[Dispute]:
        """Retrieve a dispute from the log."""
        try:
            dispute_path = os.path.join(self.log_dir, f"{dispute_id}.json")
            if not os.path.exists(dispute_path):
                logger.warning(f"Dispute {dispute_id} not found in logs")
                return None
                
            with open(dispute_path, 'r') as f:
                dispute_data = json.load(f)
            
            logger.info(f"Retrieved dispute {dispute_id} from logs")
            return Dispute.from_dict(dispute_data)
        except Exception as e:
            logger.error(f"Failed to retrieve dispute {dispute_id}: {str(e)}")
            return None
    
    def list_disputes(self, status: Optional[DisputeStatus] = None) -> List[str]:
        """List all dispute IDs, optionally filtered by status."""
        try:
            dispute_ids = []
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.json'):
                    dispute_id = filename[:-5]  # Remove .json extension
                    
                    if status is not None:
                        dispute = self.get_dispute(dispute_id)
                        if dispute and dispute.status == status:
                            dispute_ids.append(dispute_id)
                    else:
                        dispute_ids.append(dispute_id)
            
            logger.info(f"Listed {len(dispute_ids)} disputes" + 
                        (f" with status {status.name}" if status else ""))
            return dispute_ids
        except Exception as e:
            logger.error(f"Failed to list disputes: {str(e)}")
            return []

class DisputeResolver:
    """Main class for resolving disputes between competing logic or models."""
    
    def __init__(self):
        self.parliament_client = ParliamentClient()
        self.dispute_logger = DisputeLogger()
        logger.info("DisputeResolver initialized")
    
    def create_dispute(
        self,
        title: str,
        description: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any],
        strategy: ResolutionStrategy = ResolutionStrategy.MAJORITY_VOTE,
        priority: int = 1,
        tags: List[str] = None
    ) -> Dispute:
        """Create a new dispute for resolution."""
        if tags is None:
            tags = []
            
        if not 1 <= priority <= 5:
            raise ValueError("Priority must be between 1 and 5")
            
        if not options:
            raise ValueError("At least one option must be provided")
            
        dispute_id = str(uuid.uuid4())
        dispute = Dispute(
            id=dispute_id,
            title=title,
            description=description,
            options=options,
            context=context,
            status=DisputeStatus.PENDING,
            strategy=strategy,
            priority=priority,
            tags=tags
        )
        
        self.dispute_logger.log_dispute(dispute)
        logger.info(f"Created new dispute: {dispute_id} - {title}")
        return dispute
    
    async def resolve_dispute(self, dispute: Dispute) -> Dispute:
        """Resolve a dispute using the specified strategy."""
        if dispute.status != DisputeStatus.PENDING:
            logger.warning(f"Dispute {dispute.id} is not in PENDING state, current state: {dispute.status.name}")
            return dispute
        
        logger.info(f"Starting resolution for dispute {dispute.id} using {dispute.strategy.name} strategy")
        
        # Update status to in progress
        dispute.status = DisputeStatus.IN_PROGRESS
        self.dispute_logger.log_dispute(dispute)
        
        try:
            if dispute.strategy == ResolutionStrategy.MAJORITY_VOTE:
                resolution = await self._resolve_by_majority_vote(dispute)
            elif dispute.strategy == ResolutionStrategy.WEIGHTED_VOTE:
                resolution = await self._resolve_by_weighted_vote(dispute)
            elif dispute.strategy == ResolutionStrategy.ETHICAL_PRIORITY:
                resolution = await self._resolve_by_ethical_priority(dispute)
            elif dispute.strategy == ResolutionStrategy.HUMAN_INTERVENTION:
                resolution = await self._resolve_by_human_intervention(dispute)
            elif dispute.strategy == ResolutionStrategy.CONSENSUS_BUILDING:
                resolution = await self._resolve_by_consensus_building(dispute)
            elif dispute.strategy == ResolutionStrategy.SANDBOX_TEST:
                resolution = await self._resolve_by_sandbox_test(dispute)
            else:
                raise ValueError(f"Unsupported resolution strategy: {dispute.strategy.name}")
            
            dispute.resolution = resolution
            dispute.status = DisputeStatus.RESOLVED
            logger.info(f"Dispute {dispute.id} resolved successfully")
        except Exception as e:
            logger.error(f"Failed to resolve dispute {dispute.id}: {str(e)}")
            dispute.status = DisputeStatus.FAILED
            dispute.resolution = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        # Log the updated dispute
        self.dispute_logger.log_dispute(dispute)
        return dispute
    
    async def _resolve_by_majority_vote(self, dispute: Dispute) -> Dict[str, Any]:
        """Resolve dispute by simple majority vote."""
        # Submit to Parliament for ethical evaluation
        submission = await self.parliament_client.submit_dispute(dispute)
        tracking_id = submission["tracking_id"]
        
        # Wait for resolution
        for _ in range(MAX_RETRY_ATTEMPTS):
            await asyncio.sleep(2)  # Wait before checking
            resolution = await self.parliament_client.get_resolution(dispute.id, tracking_id)
            if resolution["status"] == "RESOLVED":
                selected_option = dispute.options[resolution["resolution"]["selected_option"]]
                return {
                    "selected_option": resolution["resolution"]["selected_option"],
                    "option_details": selected_option,
                    "confidence": resolution["resolution"]["confidence"],
                    "reasoning": resolution["resolution"]["reasoning"],
                    "value_alignment": resolution["resolution"]["value_alignment"],
                    "method": "majority_vote",
                    "timestamp": datetime.now().isoformat()
                }
        
        raise TimeoutError(f"Resolution timed out for dispute {dispute.id}")
    
    async def _resolve_by_weighted_vote(self, dispute: Dispute) -> Dict[str, Any]:
        """Resolve dispute by weighted voting based on confidence scores."""
        # For demonstration, we'll simulate this with a mock implementation
        await asyncio.sleep(1)  # Simulate processing time
        
        # Mock weighted voting calculation
        weights = [0.7, 0.2, 0.1]  # Example weights for different voters
        scores = []
        
        for i, option in enumerate(dispute.options):
            # Calculate weighted score for each option
            score = sum((i + 1) * weight for i, weight in enumerate(weights))
            scores.append((i, score))
        
        # Select option with highest score
        selected_idx, highest_score = max(scores, key=lambda x: x[1])
        selected_option = dispute.options[selected_idx]
        
        return {
            "selected_option": selected_idx,
            "option_details": selected_option,
            "confidence": highest_score / sum(weight for weight in weights) / len(dispute.options),
            "reasoning": "Selected based on weighted voting across multiple evaluators",
            "method": "weighted_vote",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _resolve_by_ethical_priority(self, dispute: Dispute) -> Dict[str, Any]:
        """Resolve dispute by prioritizing ethical considerations."""
        # Submit to Parliament for ethical evaluation
        submission = await self.parliament_client.submit_dispute(dispute)
        tracking_id = submission["tracking_id"]
        
        # Wait for resolution
        for _ in range(MAX_RETRY_ATTEMPTS):
            await asyncio.sleep(2)  # Wait before checking
            resolution = await self.parliament_client.get_resolution(dispute.id, tracking_id)
            if resolution["status"] == "RESOLVED":
                # Find option with highest humanity score
                value_alignment = resolution["resolution"]["value_alignment"]
                humanity_score = value_alignment["humanity"]
                
                if humanity_score < 0.7:
                    raise ValueError(
                        f"No option meets minimum ethical standards (humanity score: {humanity_score})"
                    )
                
                selected_option = dispute.options[resolution["resolution"]["selected_option"]]
                return {
                    "selected_option": resolution["resolution"]["selected_option"],
                    "option_details": selected_option,
                    "confidence": resolution["resolution"]["confidence"],
                    "reasoning": f"Selected based on ethical priority. Humanity score: {humanity_score}",
                    "value_alignment": value_alignment,
                    "method": "ethical_priority",
                    "timestamp": datetime.now().isoformat()
                }
        
        raise TimeoutError(f"Resolution timed out for dispute {dispute.id}")
    
    async def _resolve_by_human_intervention(self, dispute: Dispute) -> Dict[str, Any]:
        """Defer resolution to human intervention."""
        # In a real implementation, this would notify a human operator
        # and wait for their input. For this example, we'll simulate a response.
        
        logger.info(f"Dispute {dispute.id} requires human intervention")
        
        # Simulate waiting for human input
        await asyncio.sleep(2)
        
        # Mock human decision
        selected_idx = 0  # Human selected the first option
        selected_option = dispute.options[selected_idx]
        
        return {
            "selected_option": selected_idx,
            "option_details": selected_option,
            "confidence": 1.0,  # Human decisions get full confidence
            "reasoning": "Decision made by human operator",
            "human_operator": "system_admin",  # In real system, this would be the actual user
            "method": "human_intervention",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _resolve_by_consensus_building(self, dispute: Dispute) -> Dict[str, Any]:
        """Resolve by building consensus among multiple evaluators."""
        # This would typically involve multiple rounds of evaluation
        # For this example, we'll simulate a simplified consensus process
        
        logger.info(f"Building consensus for dispute {dispute.id}")
        
        # Simulate multiple rounds of evaluation
        rounds = 3
        consensus_reached = False
        selected_idx = None
        confidence = 0.0
        
        for round_num in range(rounds):
            logger.info(f"Consensus round {round_num + 1} for dispute {dispute.id}")
            await asyncio.sleep(1)  # Simulate processing time
            
            # Mock consensus evaluation
            if round_num == rounds - 1:  # Last round
                consensus_reached = True
                selected_idx = 1  # Consensus on second option
                confidence = 0.85
        
        if not consensus_reached:
            raise ValueError(f"Failed to reach consensus after {rounds} rounds")
        
        selected_option = dispute.options[selected_idx]
        
        return {
            "selected_option": selected_idx,
            "option_details": selected_option,
            "confidence": confidence,
            "reasoning": f"Consensus reached after {rounds} rounds of evaluation",
            "method": "consensus_building",
            "rounds_required": rounds,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _resolve_by_sandbox_test(self, dispute: Dispute) -> Dict[str, Any]:
        """Resolve by testing options in a sandbox environment."""
        # In a real implementation, this would create isolated environments
        # to test each option and evaluate the results
        
        logger.info(f"Running sandbox tests for dispute {dispute.id}")
        
        results = []
        for i, option in enumerate(dispute.options):
            logger.info(f"Testing option {i} in sandbox")
            await asyncio.sleep(1)  # Simulate test execution
            
            # Mock test results
            success = i != 2  # Simulate that option 2 fails
            error = None if success else "Option failed security validation"
            performance_score = 0.9 - (i * 0.2) if success else 0
            
            results.append({
                "option_idx": i,
                "success": success,
                "error": error,
                "performance": performance_score
            })
        
        # Find best performing successful option
        successful_results = [r for r in results if r["success"]]
        if not successful_results:
            raise ValueError("No options passed sandbox testing")
        
        best_result = max(successful_results, key=lambda r: r["performance"])
        selected_idx = best_result["option_idx"]
        selected_option = dispute.options[selected_idx]
        
        return {
            "selected_option": selected_idx,
            "option_details": selected_option,
            "confidence": best_result["performance"],
            "reasoning": f"Selected based on sandbox test results. Performance: {best_result['performance']}",
            "test_results": results,
            "method": "sandbox_test",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_dispute(self, dispute_id: str) -> Optional[Dispute]:
        """Retrieve a dispute by ID."""
        return self.dispute_logger.get_dispute(dispute_id)
    
    async def list_disputes(
        self, status: Optional[DisputeStatus] = None
    ) -> List[str]:
        """List all dispute IDs, optionally filtered by status."""
        return self.dispute_logger.list_disputes(status)

class CapabilityRegistry:
    """Registry for system capabilities and their ethical boundaries."""
    
    def __init__(self):
        self.capabilities = {}
        self.boundaries = {}
        logger.info("CapabilityRegistry initialized")
    
    def register_capability(
        self, name: str, description: str, boundaries: Dict[str, Any]
    ) -> None:
        """Register a new capability with its ethical boundaries."""
        self.capabilities[name] = {
            "description": description,
            "registered_at": datetime.now().isoformat()
        }
        self.boundaries[name] = boundaries
        logger.info(f"Registered capability: {name}")
    
    def get_capability_boundaries(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the ethical boundaries for a capability."""
        return self.boundaries.get(name)
    
    def is_action_permitted(self, capability: str, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if an action is permitted within capability boundaries."""
        if capability not in self.boundaries:
            return False, f"Capability '{capability}' not registered"
        
        boundaries = self.boundaries[capability]
        
        # Check each boundary condition
        for key, boundary in boundaries.items():
            if key in action:
                action_value = action[key]
                
                if isinstance(boundary, dict) and "min" in boundary and "max" in boundary:
                    # Range boundary
                    if not (boundary["min"] <= action_value <= boundary["max"]):
                        return False, f"Action exceeds {key} boundary: {action_value} not in range [{boundary['min']}, {boundary['max']}]"
                elif isinstance(boundary, list):
                    # Enumeration boundary
                    if action_value not in boundary:
                        return False, f"Action value for {key} not in allowed values: {action_value} not in {boundary}"
                else:
                    # Direct comparison
                    if action_value != boundary:
                        return False, f"Action value for {key} does not match required value: {action_value} != {boundary}"
        
        return True, "Action permitted"

class DisputeAPI:
    """API for interacting with the dispute resolution system."""
    
    def __init__(self, resolver: DisputeResolver):
        self.resolver = resolver
        self.app = Flask(__name__)
        self.setup_routes()
        logger.info("DisputeAPI initialized")
    
    def setup_routes(self):
        """Set up API routes."""
        @self.app.route('/disputes', methods=['POST'])
        async def create_dispute():
            try:
                data = Request().json
                required_fields = ['title', 'description', 'options', 'context']
                for field in required_fields:
                    if field not in data:
                        return Response(
                            json.dumps({"error": f"Missing required field: {field}"}),
                            status=400,
                            mimetype='application/json'
                        )
                
                strategy = ResolutionStrategy[data.get('strategy', 'MAJORITY_VOTE')]
                priority = int(data.get('priority', 1))
                tags = data.get('tags', [])
                
                dispute = self.resolver.create_dispute(
                    title=data['title'],
                    description=data['description'],
                    options=data['options'],
                    context=data['context'],
                    strategy=strategy,
                    priority=priority,
                    tags=tags
                )
                
                return Response(
                    json.dumps({"dispute_id": dispute.id}),
                    status=201,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error creating dispute: {str(e)}")
                return Response(
                    json.dumps({"error": str(e)}),
                    status=500,
                    mimetype='application/json'
                )
        
        @self.app.route('/disputes/<dispute_id>/resolve', methods=['POST'])
        async def resolve_dispute(dispute_id):
            try:
                dispute = await self.resolver.get_dispute(dispute_id)
                if not dispute:
                    return Response(
                        json.dumps({"error": f"Dispute not found: {dispute_id}"}),
                        status=404,
                        mimetype='application/json'
                    )
                
                resolved_dispute = await self.resolver.resolve_dispute(dispute)
                return Response(
                    json.dumps(resolved_dispute.to_dict()),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error resolving dispute: {str(e)}")
                return Response(
                    json.dumps({"error": str(e)}),
                    status=500,
                    mimetype='application/json'
                )
        
        @self.app.route('/disputes/<dispute_id>', methods=['GET'])
        async def get_dispute(dispute_id):
            try:
                dispute = await self.resolver.get_dispute(dispute_id)
                if not dispute:
                    return Response(
                        json.dumps({"error": f"Dispute not found: {dispute_id}"}),
                        status=404,
                        mimetype='application/json'
                    )
                
                return Response(
                    json.dumps(dispute.to_dict()),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error getting dispute: {str(e)}")
                return Response(
                    json.dumps({"error": str(e)}),
                    status=500,
                    mimetype='application/json'
                )
        
        @self.app.route('/disputes', methods=['GET'])
        async def list_disputes():
            try:
                status_param = Request().args.get('status')
                status = DisputeStatus[status_param] if status_param else None
                
                dispute_ids = await self.resolver.list_disputes(status)
                return Response(
                    json.dumps({"disputes": dispute_ids}),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error listing disputes: {str(e)}")
                return Response(
                    json.dumps({"error": str(e)}),
                    status=500,
                    mimetype='application/json'
                )
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the API server."""
        logger.info(f"Starting DisputeAPI server on {host}:{port}")
        # In a real implementation, this would start the Flask server
        print(f"DisputeAPI server would start on {host}:{port}")

# Example usage
async def main():
    # Initialize the dispute resolver
    resolver = DisputeResolver()
    
    # Create a sample dispute
    dispute = resolver.create_dispute(
        title="Model Selection Dispute",
        description="Determine which model to use for the next processing step",
        options=[
            {
                "name": "Model A",
                "description": "Faster but less accurate",
                "accuracy": 0.85,
                "latency_ms": 50
            },
            {
                "name": "Model B",
                "description": "More accurate but slower",
                "accuracy": 0.95,
                "latency_ms": 150
            },
            {
                "name": "Model C",
                "description": "Balanced performance",
                "accuracy": 0.90,
                "latency_ms": 100
            }
        ],
        context={
            "task": "image_classification",
            "priority": "accuracy",
            "deadline_ms": 200
        },
        strategy=ResolutionStrategy.ETHICAL_PRIORITY,
        priority=3,
        tags=["model_selection", "performance", "accuracy"]
    )
    
    # Resolve the dispute
    resolved_dispute = await resolver.resolve_dispute(dispute)
    
    # Print the resolution
    if resolved_dispute.status == DisputeStatus.RESOLVED:
        print(f"Dispute resolved: {resolved_dispute.resolution}")
    else:
        print(f"Dispute resolution failed: {resolved_dispute.status.name}")
    
    # Initialize and run the API
    api = DisputeAPI(resolver)
    api.run()

if __name__ == "__main__":
    asyncio.run(main())
