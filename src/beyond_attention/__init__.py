"""A selective state-space model, implemented from scratch and measured honestly."""

from .agent import (
    AGENT_DIM,
    FAMILIES,
    RECALL_FAMILIES,
    REGISTER_NAMES,
    SELECTIVE_DISTRACTORS,
    SELECTIVE_KEYS,
    SELECTIVE_STORES,
    SELECTIVE_WIDTH,
    STEP_COUNTS,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    AgentRun,
    Event,
    Instruction,
    Registers,
    Step,
    Task,
    ToolCall,
    ToolResult,
    ToolSchema,
    bridge_to_selective_scan,
    carried_value,
    choose_action,
    evaluate_plan,
    example_task,
    execute,
    instruction,
    new_state,
    capacity_suite,
    capacity_task,
    example_recall_task,
    example_stale_task,
    read_registers,
    recall_suite,
    recall_task,
    replay,
    required_state_width,
    run_agent,
    scan_states,
    selective_example_task,
    selective_suite,
    slot_value,
    stale_suite,
    stale_task,
    state_bytes,
    task_suite,
    validate_call,
)
from .memory import (
    DEFAULT_CAPACITY,
    TAGGED_SLOT_BYTES,
    UNTAGGED_SLOT_BYTES,
    MemoryStore,
    store_bytes,
)
from .model import (
    AttentionBlock,
    LanguageModel,
    RMSNorm,
    SelectiveSSMBlock,
    build_pair,
    count_parameters,
)
from .streaming import (
    BlockState,
    StreamState,
    init_stream,
    kv_cache_bytes,
    ssm_state_bytes,
    stream_sequence,
    stream_step,
)
from .ssm import (
    selective_scan,
    selective_scan_associative,
    selective_scan_chunked,
    selective_scan_reference,
)
from .tasks import (
    Batch,
    accuracy,
    mqar_batch,
    register_batch,
    register_chance,
    vocabulary_for,
)
from .train import Result, train
from .voice import (
    FEATURE_NAMES,
    F0_MAX_HZ,
    F0_MIN_HZ,
    VoiceEncoder,
    VoiceFeatures,
    affect_descriptors,
    autocorrelation_f0,
    feature_matrix,
    frame_count,
    frame_features,
    frame_signal,
    hann_window,
    waveform_to_model_input,
)
# The one-call surface sits on top of both: it is imported last because it is the
# layer that depends on the others, and it adds no measurement of its own.
from .api import (
    AFFECT_FRAMING,
    AffectSummary,
    affect_summary,
    prosody_reading,
    run_task,
    trace_lines,
)

__version__ = "0.1.0"

__all__ = [
    # The agent loop. `stream_step` is deliberately not re-exported: the
    # streaming block above already owns that name for the model's own step,
    # and two functions with one name in one namespace is how a caller ends up
    # driving the wrong recurrence. `scan_states` is the sequence-level entry
    # point here, and `bridge_to_selective_scan` is the way from the agent's
    # memory into the model's scan.
    "run_agent", "replay", "AgentRun", "Step", "Task", "task_suite",
    "example_task", "evaluate_plan",
    # The selective family: a stream of keyed events and distractors, and a
    # query that only a memory wide enough to hold several keys can answer.
    # ``selective_suite`` exposes the task shape so the width and distractor
    # sweeps measure the same tasks the main table does.
    "selective_suite", "selective_example_task", "required_state_width",
    "slot_value", "SELECTIVE_STORES", "SELECTIVE_DISTRACTORS", "SELECTIVE_KEYS",
    "SELECTIVE_WIDTH",
    "Instruction", "instruction", "ToolCall", "ToolResult", "ToolSchema",
    "validate_call", "execute", "TOOL_SCHEMAS", "TOOL_NAMES",
    "new_state", "scan_states", "read_registers", "Registers", "Event",
    "choose_action", "carried_value", "bridge_to_selective_scan",
    "AGENT_DIM", "REGISTER_NAMES", "FAMILIES", "STEP_COUNTS", "state_bytes",
    # The long-context store: an external, keyed, in-process memory the loop
    # writes with ``remember`` and reads with ``fetch``, plus the recall
    # families that exercise it over a controllable distance. It is separate
    # from the selective family's *internal* slots, and named apart from it.
    "MemoryStore", "store_bytes", "DEFAULT_CAPACITY",
    "TAGGED_SLOT_BYTES", "UNTAGGED_SLOT_BYTES",
    "RECALL_FAMILIES", "recall_suite", "recall_task", "capacity_suite",
    "capacity_task", "stale_suite", "stale_task",
    "example_recall_task", "example_stale_task",
    "AttentionBlock", "LanguageModel", "RMSNorm", "SelectiveSSMBlock",
    "build_pair", "count_parameters",
    "selective_scan", "selective_scan_associative", "selective_scan_chunked",
    "selective_scan_reference",
    "BlockState", "StreamState", "init_stream", "stream_step", "stream_sequence",
    "kv_cache_bytes", "ssm_state_bytes",
    "Batch", "accuracy", "mqar_batch", "register_batch",
    "register_chance", "vocabulary_for",
    "Result", "train",
    "VoiceEncoder", "VoiceFeatures", "affect_descriptors",
    "autocorrelation_f0", "feature_matrix",
    "frame_count", "frame_features", "frame_signal", "hann_window",
    "waveform_to_model_input",
    "FEATURE_NAMES", "F0_MAX_HZ", "F0_MIN_HZ",
    # The two one-call entry points: a waveform to a prosody summary, and a task
    # to its whole trace. Both return the measured object rather than a rendering
    # of it, so a caller can compute on exactly what the dashboard draws.
    "affect_summary", "prosody_reading", "AffectSummary", "AFFECT_FRAMING",
    "run_task", "trace_lines",
    "__version__",
]
