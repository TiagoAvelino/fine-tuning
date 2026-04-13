#!/usr/bin/env python3
"""Transform legacy OCP troubleshooting dataset into MCP-agent training format.

Reads ocp-instructions.jsonl and produces ocp-agent-instructions.jsonl with
rich tool-trace examples for training a Platform Engineering Solver agent.
"""

import json
import re
import hashlib
import argparse
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Tool-family selection rules per (category, sub_cause)
# ---------------------------------------------------------------------------

TOOL_FAMILIES_BY_CATEGORY = {
    "quota_exceeded": {
        "_default": ["get_quota", "get_workload_status"],
        "cpu":          ["get_quota", "get_workload_status"],
        "memory":       ["get_quota", "get_workload_status"],
        "storage":      ["get_quota", "get_pvc"],
        "gpu":          ["get_quota", "get_nodes"],
        "object_count": ["get_quota", "get_workload_status"],
        "limitrange":   ["get_quota", "get_config"],
    },
    "scheduling_constraint": {
        "_default":        ["get_pod_events", "get_nodes"],
        "taint":           ["get_pod_events", "get_nodes"],
        "nodeselector":    ["get_pod_events", "get_nodes"],
        "resource_fit":    ["get_pod_events", "get_nodes"],
        "anti_affinity":   ["get_pod_events", "get_nodes"],
        "affinity":        ["get_pod_events", "get_nodes"],
        "gpu_resource":    ["get_pod_events", "get_nodes"],
        "custom_resource": ["get_pod_events", "get_nodes"],
        "topology":        ["get_pod_events", "get_nodes"],
        "priority":        ["get_pod_events", "get_workload_status"],
    },
    "image_pull_error": {
        "_default":       ["get_pod_events", "get_workload_status"],
        "tag_not_found":  ["get_pod_events", "get_workload_status"],
        "creds_invalid":  ["get_pod_events", "get_secret_metadata"],
        "auth_required":  ["get_pod_events", "get_secret_metadata"],
        "network":        ["get_pod_events", "get_workload_status"],
        "arch_mismatch":  ["get_pod_events", "get_nodes"],
        "registry_down":  ["get_pod_events", "get_workload_status"],
        "rate_limit":     ["get_pod_events", "get_workload_status"],
    },
    "crashloop_backoff": {
        "_default":      ["get_logs", "get_pod_events"],
        "missing_env":   ["get_logs", "get_config"],
        "config_error":  ["get_logs", "get_config"],
        "db_connection": ["get_logs", "get_endpoints"],
        "scc":           ["get_logs", "get_config"],
        "oom":           ["get_pod_events", "get_workload_status"],
        "migration":     ["get_logs", "get_workload_status"],
        "dependency":    ["get_logs", "get_endpoints"],
        "bad_rollout":   ["get_logs", "get_rollout"],
        "liveness":      ["get_pod_events", "get_logs"],
        "network":       ["get_logs", "get_endpoints"],
    },
    "route_503": {
        "_default":     ["get_route", "get_endpoints"],
        "selector":     ["get_route", "get_endpoints", "get_workload_status"],
        "timeout":      ["get_route", "get_endpoints"],
        "readiness":    ["get_endpoints", "get_workload_status"],
        "tls_backend":  ["get_route", "get_endpoints"],
        "network":      ["get_route", "get_endpoints", "get_service"],
    },
    "pvc_pending": {
        "_default":   ["get_pvc", "get_storage_class"],
        "no_sc":      ["get_pvc", "get_storage_class"],
        "csi_driver": ["get_pvc", "get_storage_class"],
        "vsphere":    ["get_pvc", "get_storage_class"],
        "snapshot":   ["get_pvc", "get_storage_class"],
        "capacity":   ["get_pvc", "get_nodes"],
        "access_mode":["get_pvc", "get_storage_class"],
    },
    "unknown": {
        "_default":       ["get_pod_events", "get_logs", "get_nodes"],
        "app_logic":      ["get_pod_events", "get_logs"],
        "node_flapping":  ["get_nodes", "get_pod_events"],
        "pending_unknown": ["get_pod_events", "get_workload_status"],
        "eviction":       ["get_pod_events", "get_nodes"],
    },
}

# ---------------------------------------------------------------------------
# Resource-type inference from instruction text
# ---------------------------------------------------------------------------

RESOURCE_TYPE_PATTERNS = [
    (r"\bDaemonSet:\s*(\S+)", "DaemonSet"),
    (r"\bStatefulSet:\s*(\S+)", "StatefulSet"),
    (r"\bDeployment:\s*(\S+)", "Deployment"),
    (r"\bCronJob:\s*(\S+)", "CronJob"),
    (r"\bJob:\s*(\S+)", "Job"),
    (r"\bRoute:\s*(\S+)", "Route"),
    (r"\bPVC:\s*(\S+)", "PVC"),
    (r"\bService:\s*(\S+)", "Service"),
    (r"\bPod\s+(\S+)", "Pod"),
]

NAMESPACE_PATTERNS = [
    r"-n\s+(\S+)",
    r"namespace[:\s]+(\S+)",
    r"in\s+(\S+)\s+namespace",
]

# ---------------------------------------------------------------------------
# Goal templates by category
# ---------------------------------------------------------------------------

GOAL_TEMPLATES = {
    "quota_exceeded":         "Determine which resource quota is blocking the workload and recommend how to resolve it",
    "scheduling_constraint":  "Identify why the pod cannot be scheduled and determine the scheduling constraint",
    "image_pull_error":       "Diagnose why the container image cannot be pulled and identify the root cause",
    "crashloop_backoff":      "Investigate why the container is crash-looping and identify the root cause from logs and events",
    "route_503":              "Determine why the route is returning HTTP 503 errors and identify the backend issue",
    "pvc_pending":            "Investigate why the PVC is stuck in Pending state and identify the storage provisioning issue",
    "unknown":                "Gather evidence to diagnose the issue since the root cause is not immediately clear",
}

EVIDENCE_MAP = {
    "quota_exceeded":        ["Current quota usage and limits", "Pod resource requests"],
    "scheduling_constraint": ["Pod scheduling events", "Node labels, taints, and available resources"],
    "image_pull_error":      ["Pod events showing pull errors", "Image reference and pull secret configuration"],
    "crashloop_backoff":     ["Container logs from previous crash", "Pod events and restart history"],
    "route_503":             ["Route configuration and backend service", "Endpoint list and pod readiness"],
    "pvc_pending":           ["PVC events and provisioner status", "StorageClass configuration"],
    "unknown":               ["Pod events and status", "Container logs", "Node conditions"],
}


def _stable_id(instruction: str) -> str:
    return "ocp-" + hashlib.sha256(instruction.encode()).hexdigest()[:12]


def _extract_resource(instruction: str) -> tuple[str, str]:
    for pattern, rtype in RESOURCE_TYPE_PATTERNS:
        m = re.search(pattern, instruction)
        if m:
            name = m.group(1).rstrip(";,.: ")
            return rtype, name
    return "unknown", "unknown"


def _extract_namespace(instruction: str) -> str:
    for pat in NAMESPACE_PATTERNS:
        m = re.search(pat, instruction, re.IGNORECASE)
        if m:
            return m.group(1)
    return "unknown"


def _normalize_user_request(instruction: str) -> str:
    """Strip meta-framing and return a clean user request."""
    prefixes = [
        r"^(SRE escalation\.?\s*What'?s the fix\?\s*\n*)",
        r"^(Production alert\s*[—–-]+\s*need help:?\s*\n*)",
        r"^(Help me troubleshoot this OpenShift issue:?\s*\n*)",
        r"^(I'm new to OpenShift\.?\s*Can you explain what'?s happening and how to fix it\?\s*\n*)",
        r"^(I ran oc describe and got this:?\s*\n*)",
        r"^(What would cause this\?\s*\n*)",
        r"^(Can you diagnose this and provide oc commands to fix it\?\s*\n*)",
        r"^(One of our pods has a problem:?\s*\n*)",
        r"^(Started seeing this after a cluster upgrade:?\s*\n*)",
        r"^(My application team reported this\.?\s*I need to explain and fix it:?\s*\n*)",
        r"^(Noticed during on-call:?\s*\n*)",
        r"^(After patching the cluster:?\s*\n*)",
        r"^(Developer team escalated this:?\s*\n*)",
    ]
    text = instruction.strip()
    for pfx in prefixes:
        text = re.sub(pfx, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if not text.endswith("?") and not text.endswith("."):
        text += ". What is the root cause and how do I fix it?"
    return text


# ---------------------------------------------------------------------------
# Synthetic tool-result generators
# ---------------------------------------------------------------------------

def _synth_get_quota(instruction: str, resp: dict) -> dict:
    sub = resp.get("sub_cause", "cpu")
    resource_map = {
        "cpu": ("requests.cpu", "8", "7500m", "500m"),
        "memory": ("requests.memory", "64Gi", "56Gi", "16Gi"),
        "storage": ("requests.storage", "500Gi", "480Gi", "50Gi"),
        "gpu": ("nvidia.com/gpu", "4", "4", "2"),
        "object_count": ("count/pods", "50", "50", "1"),
        "limitrange": ("requests.cpu", "4", "3800m", "500m"),
    }
    res, limit, used, req = resource_map.get(sub, resource_map["cpu"])
    m = re.search(r"requested:\s*(\S+=\S+)", instruction)
    if m:
        req = m.group(1).split("=")[1]
    m = re.search(r"used:\s*(\S+)", instruction)
    if m:
        used = m.group(1).rstrip(";,")
    m = re.search(r"limited:\s*(\S+)", instruction)
    if m:
        limit = m.group(1).rstrip(";,")
    return {
        "quota_name": "resource-quota",
        "resource": res,
        "hard_limit": limit,
        "used": used,
        "requested": req,
        "exceeded": True,
    }


def _synth_get_workload_status(instruction: str, resp: dict) -> dict:
    _, name = _extract_resource(instruction)
    rtype, _ = _extract_resource(instruction)
    m = re.search(r"Pod state:\s*(\S+)", instruction)
    pod_state = m.group(1).rstrip(";,.") if m else "Unknown"
    m = re.search(r"Restarts:\s*(\d+)", instruction)
    restarts = int(m.group(1)) if m else 0
    non_ready = ("Pending", "CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff")
    return {
        "resource": name,
        "type": rtype,
        "replicas_desired": 1,
        "replicas_ready": 0 if pod_state in non_ready else 1,
        "pod_state": pod_state,
        "restart_count": restarts,
    }


def _synth_get_pod_events(instruction: str, resp: dict) -> dict:
    _, name = _extract_resource(instruction)
    m = re.search(r"Events?:\s*(.+?)(?:\n|$)", instruction)
    event_msg = m.group(1).strip() if m else "Normal event"
    return {
        "resource": name,
        "events": [
            {"type": "Warning", "reason": "FailedScheduling" if "scheduling" in resp.get("category", "") else "BackOff",
             "message": event_msg[:200]}
        ],
    }


def _synth_get_logs(instruction: str, resp: dict) -> dict:
    _, name = _extract_resource(instruction)
    m = re.search(r"Logs?:\s*(.+?)(?:\n|$)", instruction)
    log_line = m.group(1).strip() if m else resp.get("explanation", "error occurred")
    return {
        "resource": name,
        "container": "main",
        "tail_lines": 20,
        "log_snippet": log_line[:300],
    }


def _synth_get_nodes(instruction: str, resp: dict) -> dict:
    m = re.search(r"(\d+)/(\d+) nodes are available", instruction)
    total = int(m.group(2)) if m else 6
    workers = max(total - 3, 1)
    nodes = []
    for i in range(workers):
        nodes.append({
            "name": f"worker-{i+1}",
            "status": "Ready",
            "cpu_alloc": "4",
            "mem_alloc": "16Gi",
            "gpu": "0",
        })
    if "gpu" in instruction.lower():
        nodes[0]["gpu"] = "1"
    return {"node_count": total, "worker_nodes": nodes}


def _synth_get_route(instruction: str, resp: dict) -> dict:
    m = re.search(r"Route:\s*(\S+)", instruction)
    route_name = m.group(1).rstrip(";,.") if m else "unknown-route"
    tls = "edge"
    if "re-encrypt" in instruction.lower():
        tls = "reencrypt"
    elif "passthrough" in instruction.lower():
        tls = "passthrough"
    return {
        "route_name": route_name,
        "host": f"{route_name}.apps.cluster.example.com",
        "tls_termination": tls,
        "service_target": route_name.replace("-route", ""),
        "status": "Admitted",
    }


def _synth_get_endpoints(instruction: str, resp: dict) -> dict:
    m = re.search(r"(?:Service|Route):\s*(\S+)", instruction)
    svc = m.group(1).rstrip(";,.") if m else "unknown"
    has_endpoints = "selector" not in resp.get("sub_cause", "")
    m2 = re.search(r"endpoints:\s*(\d+)", instruction, re.IGNORECASE)
    if m2:
        has_endpoints = int(m2.group(1)) > 0
    return {
        "service": svc,
        "endpoints": [{"ip": "10.130.2.15", "port": 8080}] if has_endpoints else [],
        "ready_count": 1 if has_endpoints else 0,
    }


def _synth_get_pvc(instruction: str, resp: dict) -> dict:
    m = re.search(r"PVC:\s*(\S+)", instruction)
    pvc_name = m.group(1).rstrip(";,.") if m else "unknown-pvc"
    m2 = re.search(r"Status:\s*(\S+)", instruction)
    status = m2.group(1) if m2 else "Pending"
    m3 = re.search(r"Events?:\s*(.+?)(?:\n|$)", instruction)
    event = m3.group(1).strip() if m3 else "provisioning failed"
    return {
        "pvc_name": pvc_name,
        "status": status,
        "storage_class": "gp3-csi",
        "requested_size": "50Gi",
        "bound_pv": None,
        "event_message": event[:200],
    }


def _synth_get_storage_class(instruction: str, resp: dict) -> dict:
    m = re.search(r"StorageClass:\s*(\S+)", instruction)
    sc_name = m.group(1).rstrip(";,.") if m else "gp3-csi"
    return {
        "name": sc_name,
        "provisioner": "ebs.csi.aws.com",
        "reclaim_policy": "Delete",
        "volume_binding_mode": "WaitForFirstConsumer",
        "allow_expansion": True,
    }


def _synth_get_config(instruction: str, resp: dict) -> dict:
    _, name = _extract_resource(instruction)
    return {
        "resource": name,
        "config_type": "deployment_spec",
        "relevant_fields": {
            "replicas": 1,
            "strategy": "RollingUpdate",
        },
    }


def _synth_get_secret_metadata(instruction: str, resp: dict) -> dict:
    return {
        "secret_name": "pull-secret",
        "type": "kubernetes.io/dockerconfigjson",
        "created": "2025-01-15T10:00:00Z",
        "keys": [".dockerconfigjson"],
        "data_size_bytes": 256,
    }


def _synth_get_rollout(instruction: str, resp: dict) -> dict:
    _, name = _extract_resource(instruction)
    return {
        "resource": name,
        "current_revision": 3,
        "desired_revision": 4,
        "status": "progressing",
        "conditions": [{"type": "Progressing", "status": "True"}],
    }


def _synth_get_service(instruction: str, resp: dict) -> dict:
    m = re.search(r"(?:Service|Route):\s*(\S+)", instruction)
    svc = m.group(1).rstrip(";,.") if m else "unknown"
    return {
        "service": svc,
        "type": "ClusterIP",
        "cluster_ip": "172.30.45.123",
        "ports": [{"port": 8080, "target_port": 8080, "protocol": "TCP"}],
        "selector": {"app": svc},
    }


TOOL_SYNTH = {
    "get_quota":           _synth_get_quota,
    "get_workload_status": _synth_get_workload_status,
    "get_pod_events":      _synth_get_pod_events,
    "get_logs":            _synth_get_logs,
    "get_nodes":           _synth_get_nodes,
    "get_route":           _synth_get_route,
    "get_endpoints":       _synth_get_endpoints,
    "get_pvc":             _synth_get_pvc,
    "get_storage_class":   _synth_get_storage_class,
    "get_config":          _synth_get_config,
    "get_secret_metadata": _synth_get_secret_metadata,
    "get_rollout":         _synth_get_rollout,
    "get_service":         _synth_get_service,
}


def _build_tool_args(tool_name: str, resource_type: str, resource_name: str, namespace: str) -> dict:
    """Build minimal tool arguments."""
    ns = {"namespace": namespace} if namespace != "unknown" else {}
    if tool_name in ("get_quota",):
        return {**ns}
    if tool_name in ("get_nodes",):
        return {}
    if tool_name in ("get_storage_class",):
        return {}
    if tool_name in ("get_secret_metadata",):
        return {"secret_name": "pull-secret", **ns}
    if tool_name in ("get_pvc",):
        return {"pvc_name": resource_name, **ns}
    if tool_name in ("get_route",):
        return {"route_name": resource_name, **ns}
    if tool_name in ("get_endpoints", "get_service"):
        return {"service_name": resource_name, **ns}
    return {"resource_type": resource_type, "resource_name": resource_name, **ns}


# ---------------------------------------------------------------------------
# Mutation safety rules
# ---------------------------------------------------------------------------

SAFE_MUTATIONS = {
    "oc annotate",
    "oc label",
    "oc scale",
}

MUTATING_COMMANDS = {
    "oc patch", "oc create", "oc delete", "oc apply",
    "oc adm", "oc set",
}


def _is_mutating(cmd: str) -> bool:
    for prefix in MUTATING_COMMANDS:
        if cmd.strip().startswith(prefix):
            return True
    return False


def _is_safe_mutation(cmd: str) -> bool:
    for prefix in SAFE_MUTATIONS:
        if cmd.strip().startswith(prefix):
            return True
    return False


def _assess_auto_apply(commands: list[str]) -> tuple[bool, bool]:
    """Return (safe_to_auto_apply, requires_approval)."""
    has_mutation = any(_is_mutating(c) for c in commands)
    if not has_mutation:
        return False, False
    all_safe = all(_is_safe_mutation(c) for c in commands if _is_mutating(c))
    return all_safe, not all_safe


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def transform_example(instruction: str, response_str: str, idx: int) -> dict:
    resp = json.loads(response_str)
    category = resp.get("category", "unknown")
    sub_cause = resp.get("sub_cause", "unknown")
    confidence = resp.get("confidence", "medium")
    explanation = resp.get("explanation", "")
    fix = resp.get("fix", [])
    commands = resp.get("commands", [])
    verification = resp.get("verification", "")

    resource_type, resource_name = _extract_resource(instruction)
    namespace = _extract_namespace(instruction)
    case_id = _stable_id(instruction)

    cat_tools = TOOL_FAMILIES_BY_CATEGORY.get(category, TOOL_FAMILIES_BY_CATEGORY["unknown"])
    selected_tools = cat_tools.get(sub_cause, cat_tools["_default"])

    goal = GOAL_TEMPLATES.get(category, GOAL_TEMPLATES["unknown"])
    evidence = EVIDENCE_MAP.get(category, EVIDENCE_MAP["unknown"])

    tool_trace = []
    for tool_name in selected_tools:
        synth_fn = TOOL_SYNTH.get(tool_name)
        if synth_fn:
            args = _build_tool_args(tool_name, resource_type, resource_name, namespace)
            result = synth_fn(instruction, resp)
            tool_trace.append({
                "tool_name": tool_name,
                "arguments": args,
                "tool_result": result,
            })

    safe_auto, needs_approval = _assess_auto_apply(commands)

    grounded_explanation = explanation
    if tool_trace:
        first_tool = tool_trace[0]["tool_name"]
        grounded_explanation = (
            f"Based on {first_tool} results: {explanation}"
        )

    return {
        "case_id": case_id,
        "source_case": {
            "instruction": instruction,
            "legacy_response": resp,
        },
        "agent_training_example": {
            "user_request": _normalize_user_request(instruction),
            "context": {
                "namespace": namespace,
                "resource_type": resource_type,
                "resource_name": resource_name,
            },
            "policy": {
                "must_ground_with_tools": True,
                "mutations_require_approval": True,
            },
            "assistant_plan": {
                "goal": goal,
                "needed_evidence": evidence,
                "selected_tool_family": selected_tools,
            },
            "tool_trace": tool_trace,
            "final_response": {
                "category": category,
                "sub_cause": sub_cause,
                "confidence": confidence,
                "explanation": grounded_explanation,
                "fix": fix,
                "commands": commands,
                "verification": verification,
                "used_tools": selected_tools,
                "safe_to_auto_apply": safe_auto,
                "requires_user_approval": needs_approval,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Transform OCP dataset to agent format")
    parser.add_argument("--input", default="data/ocp-instructions.jsonl")
    parser.add_argument("--output", default="data/ocp-agent-instructions.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    count = 0
    errors = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result = transform_example(obj["instruction"], obj["response"], idx)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] Line {idx+1}: {e}")

    print(f"Transformed {count} examples ({errors} errors) -> {output_path}")


if __name__ == "__main__":
    main()
