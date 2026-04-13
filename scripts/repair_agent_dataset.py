#!/usr/bin/env python3
"""Strict cleanup and consistency repair pass on the OCP agent dataset.

Repairs:
- Rule 4: Mutation approval policy (all mutations require approval)
- Rule 2: Endpoint service mismatches, crashloop db_connection missing dependency endpoints
- Rule 7: assistant_plan.goal and needed_evidence quality
- Rule 10: Commands with <namespace> placeholders
- Rule 4: safe_to_auto_apply alignment
- Rule 8: Explanation quality check
"""

import json
import re
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Goal templates by (category, sub_cause) — specific investigative goals
# ---------------------------------------------------------------------------

GOAL_TEMPLATES = {
    ("quota_exceeded", "cpu"): "Identify which CPU quota is blocking pod creation and determine if the limit or usage needs adjustment",
    ("quota_exceeded", "memory"): "Identify which memory quota is preventing scheduling and assess whether to increase the limit or reduce requests",
    ("quota_exceeded", "storage"): "Determine which storage quota is preventing PVC provisioning",
    ("quota_exceeded", "gpu"): "Verify GPU quota usage and determine whether GPU capacity can be freed or the quota increased",
    ("quota_exceeded", "object_count"): "Check which object-count quota is at its limit and identify candidates for cleanup",
    ("quota_exceeded", "limitrange"): "Determine why pod creation is rejected due to missing resource requests and whether a LimitRange default is needed",
    ("scheduling_constraint", "taint"): "Identify which node taints are preventing pod scheduling and determine the correct toleration",
    ("scheduling_constraint", "nodeselector"): "Verify which nodeSelector labels are missing from available nodes",
    ("scheduling_constraint", "resource_fit"): "Check node-level resource availability to confirm whether CPU, memory, or GPU shortage is blocking scheduling",
    ("scheduling_constraint", "anti_affinity"): "Determine if anti-affinity rules are preventing scheduling due to insufficient topology spread",
    ("scheduling_constraint", "gpu_resource"): "Verify GPU availability across nodes and determine if the request exceeds any single node's capacity",
    ("scheduling_constraint", "custom_resource"): "Check whether the requested extended resource is registered on any node",
    ("scheduling_constraint", "topology_spread"): "Verify topology spread constraints and determine if replica count or maxSkew needs adjustment",
    ("scheduling_constraint", "priority"): "Check if a higher-priority pod preempted this workload",
    ("image_pull_error", "tag_not_found"): "Verify the image tag exists in the registry and check for typos in the image reference",
    ("image_pull_error", "creds_invalid"): "Determine if the pull secret credentials are valid and scoped to the correct registry",
    ("image_pull_error", "auth_required"): "Check whether a pull secret exists and is linked to the service account for this namespace",
    ("image_pull_error", "network"): "Determine if the image pull failure is caused by network connectivity issues to the registry",
    ("image_pull_error", "arch_mismatch"): "Verify whether the image supports the node's CPU architecture",
    ("image_pull_error", "registry_down"): "Determine if the container registry is unreachable or experiencing an outage",
    ("image_pull_error", "rate_limited"): "Check if Docker Hub or the registry is rate-limiting pulls and whether authentication would help",
    ("crashloop_backoff", "missing_env"): "Identify the missing environment variable from container logs and check the deployment config for the required secret or configmap reference",
    ("crashloop_backoff", "config_error"): "Analyze container logs to find the configuration error and inspect the deployment spec for incorrect mounts or settings",
    ("crashloop_backoff", "db_connection"): "Check container logs for connection errors and verify the upstream database service endpoints are healthy",
    ("crashloop_backoff", "scc"): "Determine if the container is failing due to SecurityContextConstraints or read-only filesystem restrictions",
    ("crashloop_backoff", "oom"): "Check pod events for OOMKilled signals and compare the container's memory limit against actual usage",
    ("crashloop_backoff", "migration"): "Analyze logs for database migration conflicts and determine the safe remediation path",
    ("crashloop_backoff", "dependency"): "Verify the upstream dependency service is running by checking its endpoints",
    ("crashloop_backoff", "bad_rollout"): "Inspect the rollout status and determine if a rollback is needed",
    ("crashloop_backoff", "liveness"): "Check pod events for liveness probe failures and determine if the probe configuration needs tuning",
    ("crashloop_backoff", "network"): "Analyze logs for network connectivity errors and check the target service endpoints",
    ("route_503", "selector"): "Verify the service selector matches the pod labels and check endpoint readiness",
    ("route_503", "timeout"): "Check route timeout configuration and backend response times",
    ("route_503", "readiness"): "Verify pod readiness probe status and endpoint health",
    ("route_503", "tls_backend"): "Check TLS configuration between the route and backend service",
    ("route_503", "network"): "Verify network connectivity between the router and backend pods",
    ("pvc_pending", "no_sc"): "Check if the requested StorageClass exists in the cluster",
    ("pvc_pending", "csi_driver"): "Verify the CSI driver is healthy and the StorageClass provisioner is correctly configured",
    ("pvc_pending", "vsphere"): "Check vSphere CSI driver status and datastore configuration",
    ("pvc_pending", "snapshot"): "Determine if existing volume snapshots are blocking the PVC operation",
    ("pvc_pending", "capacity"): "Check available storage capacity on the backend storage system",
    ("pvc_pending", "access_mode"): "Verify the requested access mode is supported by the StorageClass and provisioner",
    ("unknown", "app_logic"): "Gather initial evidence to narrow down the issue category before concluding",
    ("unknown", "node_flapping"): "Check node conditions and kubelet health to identify the cause of intermittent NotReady status",
    ("unknown", "pending_unknown"): "Investigate why the pod is stuck in Pending with no events — check for webhook, scheduler, or API server issues",
    ("unknown", "eviction"): "Check node conditions and pod events for eviction signals",
}

# ---------------------------------------------------------------------------
# Evidence templates by (category, sub_cause)
# ---------------------------------------------------------------------------

EVIDENCE_TEMPLATES = {
    "quota_exceeded": ["Resource quota usage and limits", "Workload pod scheduling status"],
    "scheduling_constraint": ["Pod scheduling failure events with rejection reason", "Node inventory with labels, taints, and allocatable resources"],
    "image_pull_error": ["Pod events showing image pull error details", "Pull secret configuration and registry credentials"],
    "crashloop_backoff": ["Container logs from previous crash with exit code", "Deployment environment variables, mounts, and resource configuration"],
    "route_503": ["Route configuration including TLS termination and target service", "Service endpoint health and pod readiness"],
    "pvc_pending": ["PVC status, events, and provisioner messages", "StorageClass configuration and CSI driver health"],
    "unknown": ["Pod events and recent status transitions", "Container logs and node conditions"],
}

EVIDENCE_OVERRIDES = {
    ("crashloop_backoff", "db_connection"): ["Container logs showing connection error details", "Upstream database service endpoint readiness"],
    ("crashloop_backoff", "dependency"): ["Container logs showing dependency failure", "Upstream service endpoint availability"],
    ("crashloop_backoff", "network"): ["Container logs showing network error", "Target service endpoint status"],
    ("crashloop_backoff", "missing_env"): ["Container logs showing missing environment variable", "Deployment spec environment configuration"],
    ("crashloop_backoff", "scc"): ["Container logs showing permission denied or read-only error", "Pod security context and SCC assignment"],
    ("crashloop_backoff", "oom"): ["Pod events with OOMKilled termination reason", "Container memory limits vs actual usage"],
    ("image_pull_error", "creds_invalid"): ["Pod events showing authentication failure", "Pull secret metadata and configured registries"],
    ("image_pull_error", "auth_required"): ["Pod events showing authentication requirement", "Service account pull secret bindings"],
    ("route_503", "selector"): ["Route target service configuration", "Service selector vs pod label comparison and endpoint count"],
}

# ---------------------------------------------------------------------------
# Dependency service inference for crashloop db_connection
# ---------------------------------------------------------------------------

_DEP_MAP = {
    "kafka": ("kafka-bootstrap", 9092),
    "redis": ("redis-master", 6379),
    "postgres": ("postgres", 5432),
    "mysql": ("mysql", 3306),
    "mongodb": ("mongodb", 27017),
    "rabbitmq": ("rabbitmq", 5672),
    "elasticsearch": ("elasticsearch", 9200),
    "memcached": ("memcached", 11211),
    "cassandra": ("cassandra", 9042),
    "nats": ("nats", 4222),
}


def _infer_dependency(instruction: str, rname: str) -> tuple[str, int]:
    text = instruction.lower()
    for kw, (svc, port) in _DEP_MAP.items():
        if kw in text:
            return svc, port
    return f"{rname}-db", 5432


# ---------------------------------------------------------------------------
# ValuePool for varied IPs when needed
# ---------------------------------------------------------------------------

class _VP:
    def __init__(self, seed: str):
        self._rng = random.Random(seed)

    def ip(self) -> str:
        return f"10.{self._rng.choice([128,129,130,131])}.{self._rng.randint(0,15)}.{self._rng.randint(2,254)}"

    def port(self) -> int:
        return self._rng.choice([8080, 8443, 3000, 5000, 9090, 8081, 3306, 5432, 6379])


# ---------------------------------------------------------------------------
# Repair functions
# ---------------------------------------------------------------------------

def _repair_approval_policy(ex: dict) -> dict:
    """Rule 4: Strict mutation approval policy."""
    ra = ex["agent_training_example"]["recommended_action"]
    fr = ex["agent_training_example"]["final_response"]

    if ra["type"] == "mutation":
        ra["requires_approval"] = True
        if not ra.get("proposed_tool"):
            ra["proposed_tool"] = "patch_resource"
        ra["reason"] = "The proposed fix modifies cluster resources and requires operator approval before execution"
        fr["requires_user_approval"] = True
        fr["safe_to_auto_apply"] = False

    elif ra["type"] == "read_only":
        ra["requires_approval"] = False
        ra["proposed_tool"] = None
        ra["reason"] = "This is a diagnostic investigation; no cluster state changes are proposed"
        fr["requires_user_approval"] = False
        fr["safe_to_auto_apply"] = False

    return ex


def _repair_plan_goal(ex: dict) -> dict:
    """Rule 7: Replace legacy-explanation-as-goal with investigative goal."""
    cat = ex["agent_training_example"]["final_response"]["category"]
    sub = ex["agent_training_example"]["final_response"]["sub_cause"]
    key = (cat, sub)

    goal = GOAL_TEMPLATES.get(key)
    if not goal:
        goal = GOAL_TEMPLATES.get((cat, "_default"))
    if not goal:
        goal = f"Investigate the {cat.replace('_', ' ')} issue and determine root cause from tool evidence"

    ex["agent_training_example"]["assistant_plan"]["goal"] = goal
    return ex


def _repair_needed_evidence(ex: dict) -> dict:
    """Rule 7: Replace generic 'X output for Y' with meaningful evidence descriptions."""
    cat = ex["agent_training_example"]["final_response"]["category"]
    sub = ex["agent_training_example"]["final_response"]["sub_cause"]

    evidence = EVIDENCE_OVERRIDES.get((cat, sub))
    if not evidence:
        evidence = EVIDENCE_TEMPLATES.get(cat, ["Pod events and status", "Related resource configuration"])

    ex["agent_training_example"]["assistant_plan"]["needed_evidence"] = evidence
    return ex


def _repair_namespace_in_commands(ex: dict) -> dict:
    """Rule 10: Replace <namespace> placeholder with actual namespace."""
    ns = ex["agent_training_example"]["context"]["namespace"]
    if ns and ns != "unknown":
        cmds = ex["agent_training_example"]["final_response"]["commands"]
        ex["agent_training_example"]["final_response"]["commands"] = [
            cmd.replace("<namespace>", ns) for cmd in cmds
        ]
        verif = ex["agent_training_example"]["final_response"]["verification"]
        ex["agent_training_example"]["final_response"]["verification"] = verif.replace("<namespace>", ns)
    return ex


def _repair_endpoint_service(ex: dict) -> dict:
    """Rule 2: Fix unknown/empty endpoint service names."""
    cat = ex["agent_training_example"]["final_response"]["category"]
    rname = ex["agent_training_example"]["context"]["resource_name"]
    ns = ex["agent_training_example"]["context"]["namespace"]
    inst = ex["source_case"]["instruction"]

    # Try to extract a service/route name from the source instruction
    fallback_name = rname
    if fallback_name in ("unknown", ""):
        m = re.search(r"Route:\s*(\S+)", inst)
        if m:
            fallback_name = m.group(1).rstrip(";,.: ")
        else:
            m = re.search(r"Service:\s*(\S+)", inst)
            if m:
                fallback_name = m.group(1).rstrip(";,.: ")
            else:
                m = re.search(r"Deployment:\s*(\S+)", inst)
                if m:
                    fallback_name = m.group(1).rstrip(";,.: ")
    if fallback_name in ("unknown", ""):
        vp = _VP(ex["case_id"])
        fallback_name = vp._rng.choice([
            "web-frontend", "api-backend", "app-service", "main-app",
            "portal-svc", "gateway-svc", "catalog-api", "auth-service",
        ])
        ex["agent_training_example"]["context"]["resource_name"] = fallback_name

    for t in ex["agent_training_example"]["tool_trace"]:
        if t["tool_name"] == "get_endpoints":
            svc = t["tool_result"].get("service", "")
            if svc in ("unknown", ""):
                if cat == "route_503":
                    for t2 in ex["agent_training_example"]["tool_trace"]:
                        if t2["tool_name"] == "get_route":
                            route_target = t2["tool_result"].get("target_service", "")
                            if route_target and route_target != "unknown":
                                svc = route_target
                            else:
                                svc = fallback_name
                                t2["tool_result"]["target_service"] = fallback_name
                            break
                    else:
                        svc = fallback_name
                else:
                    svc = fallback_name
                t["tool_result"]["service"] = svc
                t["arguments"]["service_name"] = svc
                if ns and ns != "unknown":
                    t["tool_result"]["namespace"] = ns

        if t["tool_name"] == "get_route":
            target = t["tool_result"].get("target_service", "")
            if target in ("unknown", ""):
                t["tool_result"]["target_service"] = fallback_name
            name = t["tool_result"].get("name", "")
            if name in ("unknown", ""):
                t["tool_result"]["name"] = fallback_name
                t["arguments"]["route_name"] = fallback_name
                t["tool_result"]["host"] = f"{fallback_name}.apps.cluster.example.com"

    return ex


def _repair_crashloop_dependency_endpoints(ex: dict) -> dict:
    """Rule 2: Add dependency endpoint check for crashloop db_connection/dependency/network."""
    cat = ex["agent_training_example"]["final_response"]["category"]
    sub = ex["agent_training_example"]["final_response"]["sub_cause"]
    if cat != "crashloop_backoff" or sub not in ("db_connection", "dependency", "network"):
        return ex

    has_ep = any(t["tool_name"] == "get_endpoints" for t in ex["agent_training_example"]["tool_trace"])
    if has_ep:
        return ex

    inst = ex["source_case"]["instruction"]
    rname = ex["agent_training_example"]["context"]["resource_name"]
    ns = ex["agent_training_example"]["context"]["namespace"]

    dep_svc, dep_port = _infer_dependency(inst, rname)

    trace = ex["agent_training_example"]["tool_trace"]
    ep_tool = {
        "tool_name": "get_endpoints",
        "arguments": {"namespace": ns, "service_name": dep_svc},
        "tool_result": {
            "service": dep_svc,
            "namespace": ns,
            "endpoints": [],
            "ready_count": 0,
            "not_ready_count": random.randint(1, 3),
        },
    }

    if len(trace) >= 2:
        old_second = trace[1]
        if old_second["tool_name"] in ("get_pod_events", "get_config"):
            trace[1] = ep_tool
        else:
            trace.append(ep_tool)
    else:
        trace.append(ep_tool)

    tools_used = [t["tool_name"] for t in trace]
    ex["agent_training_example"]["final_response"]["used_tools"] = tools_used
    ex["agent_training_example"]["assistant_plan"]["selected_tool_family"] = tools_used
    return ex


def _repair_explanation_prefix(ex: dict) -> dict:
    """Rule 8: Remove 'Based on get_X results:' prefix if still present."""
    expl = ex["agent_training_example"]["final_response"]["explanation"]
    expl = re.sub(r"^Based on get_\w+ results:\s*", "", expl)
    ex["agent_training_example"]["final_response"]["explanation"] = expl
    return ex


def _repair_tools_used_alignment(ex: dict) -> dict:
    """Ensure used_tools and selected_tool_family match actual tool_trace."""
    trace = ex["agent_training_example"]["tool_trace"]
    tools = [t["tool_name"] for t in trace]
    ex["agent_training_example"]["final_response"]["used_tools"] = tools
    ex["agent_training_example"]["assistant_plan"]["selected_tool_family"] = tools
    return ex


# ---------------------------------------------------------------------------
# Main repair pipeline
# ---------------------------------------------------------------------------

def repair_example(ex: dict) -> dict:
    ex = _repair_approval_policy(ex)
    ex = _repair_plan_goal(ex)
    ex = _repair_needed_evidence(ex)
    ex = _repair_namespace_in_commands(ex)
    ex = _repair_endpoint_service(ex)
    ex = _repair_crashloop_dependency_endpoints(ex)
    ex = _repair_explanation_prefix(ex)
    ex = _repair_tools_used_alignment(ex)
    return ex


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ocp-agent-instructions-v2.jsonl")
    parser.add_argument("--output", default="data/ocp-agent-instructions-v3.jsonl")
    args = parser.parse_args()

    count = 0
    errors = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                repaired = repair_example(ex)
                fout.write(json.dumps(repaired, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] Line {idx+1}: {e}")

    print(f"Repaired {count} examples ({errors} errors) -> {args.output}")


if __name__ == "__main__":
    main()
