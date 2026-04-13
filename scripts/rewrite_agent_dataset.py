#!/usr/bin/env python3
"""Rewrite OCP agent dataset to gold quality.

Transforms shallow tool traces into evidence-grounded reasoning traces with:
- tool outputs that provably support the diagnosis
- natural explanations connecting evidence to conclusions
- multi-step disambiguation (25-35% of examples)
- uncertainty / insufficient-evidence examples (10-15%)
- recommended_action field with mutation safety
- varied synthetic values (no repeated placeholder data)
"""

import json
import re
import hashlib
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Varied synthetic value pools
# ---------------------------------------------------------------------------

_AZS = ["us-east-1a", "us-east-1b", "us-east-1c", "us-west-2a", "us-west-2b",
        "eu-west-1a", "eu-west-1b", "eu-central-1a"]
_INSTANCE_TYPES = ["m5.xlarge", "m5.2xlarge", "m6i.xlarge", "r5.xlarge",
                   "c5.2xlarge", "m5.4xlarge", "r6i.2xlarge"]
_GPU_TYPES = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "p3.2xlarge"]
_NAMESPACES = ["production", "staging", "development", "monitoring", "ci-cd",
               "platform-services", "data-pipeline", "ml-ops", "payments",
               "backend", "frontend", "infra", "observability", "security",
               "analytics", "messaging", "api-gateway"]
_SC_NAMES = ["gp3-csi", "gp2-csi", "standard-csi", "ocs-storagecluster-ceph-rbd",
             "thin-csi", "managed-premium", "ebs-sc", "nfs-client"]
_SC_PROVISIONERS = ["ebs.csi.aws.com", "kubernetes.io/aws-ebs",
                    "openshift-storage.rbd.csi.ceph.com", "csi.vsphere.vmware.com",
                    "pd.csi.storage.gke.io", "disk.csi.azure.com"]
_PORTS = [8080, 8443, 3000, 5000, 9090, 8081, 3306, 5432, 6379, 27017, 9200, 4317, 9411]


class ValuePool:
    """Generate varied but reproducible synthetic values per example."""

    def __init__(self, seed: str):
        self._rng = random.Random(seed)

    def ip(self) -> str:
        octet2 = self._rng.choice([128, 129, 130, 131, 132])
        return f"10.{octet2}.{self._rng.randint(0, 15)}.{self._rng.randint(2, 254)}"

    def port(self) -> int:
        return self._rng.choice(_PORTS)

    def namespace(self) -> str:
        return self._rng.choice(_NAMESPACES)

    def node_name(self) -> str:
        az = self._rng.choice(_AZS)
        suffix = "".join(self._rng.choices("abcdefghjkmnpqrstuvwxyz0123456789", k=5))
        return f"ip-10-0-{self._rng.randint(1, 99)}-{self._rng.randint(1, 254)}.{az.rsplit('-', 1)[0]}.compute.internal"

    def cpu_alloc(self) -> str:
        return self._rng.choice(["3500m", "4000m", "7500m", "8000m", "15500m", "31500m"])

    def mem_alloc(self) -> str:
        return self._rng.choice(["14Gi", "16Gi", "30Gi", "32Gi", "62Gi", "64Gi"])

    def instance_type(self) -> str:
        return self._rng.choice(_INSTANCE_TYPES)

    def gpu_instance(self) -> str:
        return self._rng.choice(_GPU_TYPES)

    def storage_class(self) -> str:
        return self._rng.choice(_SC_NAMES)

    def provisioner(self) -> str:
        return self._rng.choice(_SC_PROVISIONERS)

    def timestamp(self) -> str:
        m = self._rng.randint(1, 12)
        d = self._rng.randint(1, 28)
        h = self._rng.randint(0, 23)
        mi = self._rng.randint(0, 59)
        return f"2026-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:00Z"

    def workers(self, count: int, gpu_count: int = 0) -> list[dict]:
        nodes = []
        for i in range(count):
            n = {
                "name": self.node_name(),
                "status": "Ready",
                "instance_type": self.instance_type(),
                "cpu_allocatable": self.cpu_alloc(),
                "memory_allocatable": self.mem_alloc(),
                "gpu_allocatable": 0,
            }
            if i < gpu_count:
                n["instance_type"] = self.gpu_instance()
                n["gpu_allocatable"] = 1
            nodes.append(n)
        return nodes


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_RESOURCE_PAT = [
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


def _extract_resource(text: str) -> tuple[str, str]:
    for pat, rtype in _RESOURCE_PAT:
        m = re.search(pat, text)
        if m:
            return rtype, m.group(1).rstrip(";,.: ")
    return "unknown", "unknown"


def _extract_event_message(text: str) -> str:
    m = re.search(r"Events?:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip()[:300] if m else ""


def _extract_log_line(text: str) -> str:
    m = re.search(r"Logs?:\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip()[:300] if m else ""


def _extract_image_ref(text: str) -> str:
    m = re.search(r"image\s+([\w./-]+:\S+)", text)
    return m.group(1).rstrip(";,.") if m else "unknown:latest"


def _extract_pod_state(text: str) -> str:
    m = re.search(r"Pod state:\s*(\S+)", text)
    return m.group(1).rstrip(";,.") if m else "Unknown"


def _extract_restarts(text: str) -> int:
    m = re.search(r"Restarts:\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def _extract_exit_code(text: str) -> int | None:
    m = re.search(r"Exit code:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _extract_quota_values(text: str) -> dict:
    vals = {}
    m = re.search(r"requested:\s*(\S+=)?(\S+)", text, re.I)
    if m:
        vals["requested"] = m.group(2).rstrip(";,")
    m = re.search(r"used:\s*(\S+)", text, re.I)
    if m:
        vals["used"] = m.group(1).rstrip(";,")
    m = re.search(r"limited:\s*(\S+)", text, re.I)
    if m:
        vals["limited"] = m.group(1).rstrip(";,")
    m = re.search(r"remaining:\s*(\S+)", text, re.I)
    if m:
        vals["remaining"] = m.group(1).rstrip(";,")
    m = re.search(r"quota:\s*(\S+)", text)
    if m:
        vals["quota_name"] = m.group(1).rstrip(";,")
    return vals


def _extract_route_name(text: str) -> str:
    m = re.search(r"Route:\s*(\S+)", text)
    return m.group(1).rstrip(";,.") if m else "unknown-route"


def _extract_pvc_name(text: str) -> str:
    m = re.search(r"PVC:\s*(\S+)", text)
    return m.group(1).rstrip(";,.") if m else "unknown-pvc"


def _extract_sc_name(text: str) -> str:
    m = re.search(r"StorageClass:\s*(\S+)", text)
    return m.group(1).rstrip(";,.") if m else None


def _extract_node_count(text: str) -> int:
    m = re.search(r"(\d+)/(\d+) nodes are available", text)
    return int(m.group(2)) if m else 6


# ---------------------------------------------------------------------------
# Trace-type assignment (deterministic from case_id)
# ---------------------------------------------------------------------------

def _trace_type(case_id: str) -> str:
    h = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 100
    if h < 60:
        return "direct"
    if h < 90:
        return "disambig"
    return "uncertain"


# ---------------------------------------------------------------------------
# Namespace inference
# ---------------------------------------------------------------------------

def _infer_namespace(instruction: str, vp: ValuePool) -> str:
    for pat in [r"-n\s+(\S+)", r"namespace[:\s]+(\S+)", r"in\s+(\S+)\s+namespace"]:
        m = re.search(pat, instruction, re.I)
        if m:
            return m.group(1)
    return vp.namespace()


# ---------------------------------------------------------------------------
# User request normalizer
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = [
    r"^SRE escalation\.?\s*What'?s the fix\?\s*\n*",
    r"^Production alert\s*[—–-]+\s*need help:?\s*\n*",
    r"^Help me troubleshoot this OpenShift issue:?\s*\n*",
    r"^I'm new to OpenShift\.?\s*Can you explain what'?s happening and how to fix it\?\s*\n*",
    r"^I ran oc describe and got this:?\s*\n*",
    r"^What would cause this\?\s*\n*",
    r"^Can you diagnose this and provide oc commands to fix it\?\s*\n*",
    r"^One of our pods has a problem:?\s*\n*",
    r"^Started seeing this after a cluster upgrade:?\s*\n*",
    r"^My application team reported this\.?\s*I need to explain and fix it:?\s*\n*",
    r"^Noticed during on-call:?\s*\n*",
    r"^After patching the cluster:?\s*\n*",
    r"^Developer team escalated this:?\s*\n*",
]


def _normalize_request(instruction: str) -> str:
    t = instruction.strip()
    for pfx in _STRIP_PREFIXES:
        t = re.sub(pfx, "", t, flags=re.I | re.DOTALL).strip()
    if not t.endswith("?") and not t.endswith("."):
        t += ". What is the root cause and how do I fix it?"
    return t


# ---------------------------------------------------------------------------
# Recommended action builder
# ---------------------------------------------------------------------------

_MUTATING = {"oc patch", "oc create", "oc delete", "oc apply", "oc adm", "oc set",
             "oc scale", "oc rollout restart", "oc rollout undo"}
_SAFE_MUT = {"oc annotate", "oc label", "oc scale"}


def _recommended_action(commands: list[str], fix: list[str]) -> dict:
    has_mutation = any(
        any(cmd.strip().startswith(p) for p in _MUTATING)
        for cmd in commands
    )
    if not has_mutation:
        has_mutation = any(
            any(kw in f.lower() for kw in ["create", "delete", "patch", "add", "increase", "restart"])
            for f in fix
        )
    if not has_mutation:
        return {
            "type": "read_only",
            "requires_approval": False,
            "proposed_tool": None,
            "reason": "This is a diagnostic investigation; no cluster state changes are proposed",
        }

    all_safe = all(
        any(cmd.strip().startswith(p) for p in _SAFE_MUT)
        for cmd in commands
        if any(cmd.strip().startswith(p) for p in _MUTATING)
    )
    return {
        "type": "mutation",
        "requires_approval": not all_safe,
        "proposed_tool": "patch_resource",
        "reason": "The proposed fix modifies cluster resources and requires operator approval"
        if not all_safe
        else "The proposed change is a safe, reversible annotation or label update",
    }


# ---------------------------------------------------------------------------
# Category-specific evidence builders
# ---------------------------------------------------------------------------

def _build_quota_exceeded(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "cpu")
    ev = _extract_event_message(inst)
    qv = _extract_quota_values(inst)
    quota_name = qv.get("quota_name", f"{ns}-quota")

    def _clean_qval(val: str) -> str:
        return re.sub(r"^(cpu|memory|storage)=", "", val)

    res_map = {
        "cpu": ("requests.cpu", _clean_qval(qv.get("limited", "8")), _clean_qval(qv.get("used", f"{vp._rng.randint(6, 7)}500m")), _clean_qval(qv.get("requested", "500m"))),
        "memory": ("requests.memory", _clean_qval(qv.get("limited", "64Gi")), _clean_qval(qv.get("used", f"{vp._rng.randint(48, 60)}Gi")), _clean_qval(qv.get("requested", "16Gi"))),
        "storage": ("requests.storage", _clean_qval(qv.get("limited", "500Gi")), _clean_qval(qv.get("used", f"{vp._rng.randint(400, 490)}Gi")), _clean_qval(qv.get("requested", "50Gi"))),
        "gpu": ("nvidia.com/gpu", _clean_qval(qv.get("limited", "4")), _clean_qval(qv.get("used", "4")), _clean_qval(qv.get("requested", "2"))),
        "object_count": ("count/pods", _clean_qval(qv.get("limited", "50")), _clean_qval(qv.get("used", "50")), _clean_qval(qv.get("requested", "1"))),
        "limitrange": ("requests.cpu", _clean_qval(qv.get("limited", "4")), _clean_qval(qv.get("used", f"{vp._rng.randint(2, 3)}500m")), "unspecified"),
    }
    resource, hard, used, requested = res_map.get(sub, res_map["cpu"])

    if ttype == "direct":
        trace = [
            {"tool_name": "get_quota", "arguments": {"namespace": ns},
             "tool_result": {"quota_name": quota_name, "namespace": ns,
                             "resources": [{
                                 "resource": resource, "hard": hard, "used": used,
                                 "requested": requested, "exceeded": True
                             }]}},
            {"tool_name": "get_workload_status",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "kind": rtype, "namespace": ns,
                             "replicas": {"desired": 1, "ready": 0, "unavailable": 1},
                             "pod_phase": "Pending",
                             "conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]}},
        ]
        expl = (f"The resource quota '{quota_name}' in namespace {ns} shows {used} of {hard} "
                f"{resource} used. The {rtype} '{rname}' requests {requested}, which exceeds the remaining capacity. "
                f"{resp.get('explanation', '')}")
    elif ttype == "disambig":
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "FailedCreate",
                                         "message": f"exceeded quota: {quota_name}",
                                         "age": f"{vp._rng.randint(2, 30)}m"}]}},
            {"tool_name": "get_quota", "arguments": {"namespace": ns},
             "tool_result": {"quota_name": quota_name, "namespace": ns,
                             "resources": [{
                                 "resource": resource, "hard": hard, "used": used,
                                 "requested": requested, "exceeded": True
                             }]}},
        ]
        expl = (f"Pod events initially show a quota rejection from '{quota_name}' without specifying which resource. "
                f"Inspecting the quota reveals {resource} is at {used}/{hard} with a pending request of {requested}, "
                f"confirming the quota breach.")
    else:
        try:
            numeric = int(re.sub(r"[^\d]", "", hard))
            near_val = max(1, numeric - vp._rng.randint(1, 3))
        except (ValueError, TypeError):
            near_val = vp._rng.randint(5, 10)
        if "Gi" in hard:
            near_used = f"{near_val}Gi"
        elif "m" in hard:
            near_used = f"{near_val}m"
        else:
            near_used = str(near_val)
        trace = [
            {"tool_name": "get_quota", "arguments": {"namespace": ns},
             "tool_result": {"quota_name": quota_name, "namespace": ns,
                             "resources": [{
                                 "resource": resource, "hard": hard, "used": near_used,
                                 "requested": requested, "exceeded": False
                             }]}},
            {"tool_name": "get_workload_status",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "kind": rtype, "namespace": ns,
                             "replicas": {"desired": 1, "ready": 0, "unavailable": 1},
                             "pod_phase": "Pending",
                             "conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]}},
        ]
        expl = (f"The quota '{quota_name}' shows {near_used}/{hard} {resource} used, which appears close to the limit "
                f"but not exceeded at the time of inspection. The pod remains Pending. This may indicate a transient "
                f"spike or a request from another pod consumed the remaining capacity before this check. "
                f"Further investigation is needed to confirm the exact contention window.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "medium"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_scheduling(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "taint")
    ev = _extract_event_message(inst)
    nc = _extract_node_count(inst)
    workers_n = max(nc - 3, 2)

    if ttype == "direct":
        nodes = vp.workers(workers_n, gpu_count=1 if "gpu" in sub else 0)
        if sub == "taint":
            for n in nodes[:len(nodes)//2]:
                n["taints"] = [{"key": "node-role.kubernetes.io/infra", "effect": "NoSchedule"}]
        elif sub == "nodeselector":
            for n in nodes:
                n["labels"] = {"node-role.kubernetes.io/worker": ""}
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "FailedScheduling",
                                         "message": ev or f"0/{nc} nodes are available", "age": f"{vp._rng.randint(1, 15)}m"}]}},
            {"tool_name": "get_nodes", "arguments": {},
             "tool_result": {"total_nodes": nc, "master_nodes": 3, "worker_nodes": nodes}},
        ]
        expl = (f"Scheduling events report: {ev[:180]}. "
                f"Node inspection shows {workers_n} workers available. {resp.get('explanation', '')}")
    elif ttype == "disambig":
        nodes = vp.workers(workers_n, gpu_count=1 if "gpu" in sub else 0)
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "FailedScheduling",
                                         "message": f"0/{nc} nodes are available: multiple scheduling constraints",
                                         "age": f"{vp._rng.randint(1, 15)}m"}]}},
            {"tool_name": "get_nodes", "arguments": {},
             "tool_result": {"total_nodes": nc, "master_nodes": 3, "worker_nodes": nodes}},
        ]
        expl = (f"The scheduling event initially reports multiple constraints without specifying the primary blocker. "
                f"Inspecting nodes reveals the actual constraint: {resp.get('explanation', '')}.")
    else:
        nodes = vp.workers(workers_n)
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "FailedScheduling",
                                         "message": f"0/{nc} nodes are available",
                                         "age": f"{vp._rng.randint(5, 45)}m"}]}},
            {"tool_name": "get_nodes", "arguments": {},
             "tool_result": {"total_nodes": nc, "master_nodes": 3, "worker_nodes": nodes}},
        ]
        expl = (f"The scheduling event reports no nodes available but does not specify a single clear reason. "
                f"Node inventory shows {workers_n} workers that appear to have sufficient resources. "
                f"The issue may be a combination of taints, affinity rules, or resource pressure "
                f"not fully visible from node-level inspection alone. A deeper check of the pod spec's "
                f"tolerations, nodeSelector, and affinity rules is recommended.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "low"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_image_pull(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "tag_not_found")
    ev = _extract_event_message(inst)
    image = _extract_image_ref(inst)

    if ttype == "direct":
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "Failed",
                                         "message": f"Failed to pull image {image}: {ev[:200]}",
                                         "age": f"{vp._rng.randint(1, 20)}m"},
                                        {"type": "Warning", "reason": "BackOff",
                                         "message": "Back-off pulling image",
                                         "age": f"{vp._rng.randint(0, 5)}m"}]}},
        ]
        if sub in ("creds_invalid", "auth_required"):
            trace.append({"tool_name": "get_secret_metadata",
                          "arguments": {"namespace": ns, "secret_name": "default-dockercfg"},
                          "tool_result": {"secret_name": "default-dockercfg", "namespace": ns,
                                          "type": "kubernetes.io/dockercfg",
                                          "created": vp.timestamp(),
                                          "annotations": {},
                                          "keys": [".dockercfg"],
                                          "registries_configured": ["registry.redhat.io"],
                                          "note": f"No credentials configured for {image.split('/')[0]}"}})
        else:
            trace.append({"tool_name": "get_workload_status",
                          "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
                          "tool_result": {"name": rname, "kind": rtype, "namespace": ns,
                                          "replicas": {"desired": 1, "ready": 0, "unavailable": 1},
                                          "pod_phase": "Pending",
                                          "container_images": [image]}})
        expl = (f"Pod events confirm the image pull failure for '{image}': {ev[:150]}. "
                f"{resp.get('explanation', '')}")
    elif ttype == "disambig":
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "Failed",
                                         "message": f"Failed to pull image {image}",
                                         "age": f"{vp._rng.randint(1, 20)}m"}]}},
            {"tool_name": "get_secret_metadata",
             "arguments": {"namespace": ns, "secret_name": "pull-secret"},
             "tool_result": {"secret_name": "pull-secret", "namespace": ns,
                             "type": "kubernetes.io/dockerconfigjson",
                             "created": vp.timestamp(),
                             "keys": [".dockerconfigjson"],
                             "registries_configured": [vp._rng.choice(["docker.io", "quay.io", "gcr.io"])],
                             "note": f"Secret exists but may not cover {image.split('/')[0]}"}},
        ]
        expl = (f"The initial pod event shows a pull failure for '{image}' without a specific error code. "
                f"Checking pull secret metadata reveals credentials exist but are configured for a "
                f"different registry than the image source. {resp.get('explanation', '')}")
    else:
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [{"type": "Warning", "reason": "Failed",
                                         "message": f"Failed to pull image {image}: context deadline exceeded",
                                         "age": f"{vp._rng.randint(1, 20)}m"}]}},
            {"tool_name": "get_secret_metadata",
             "arguments": {"namespace": ns, "secret_name": "pull-secret"},
             "tool_result": {"secret_name": "pull-secret", "namespace": ns,
                             "type": "kubernetes.io/dockerconfigjson",
                             "created": vp.timestamp(),
                             "keys": [".dockerconfigjson"],
                             "registries_configured": [image.split("/")[0]]}},
        ]
        expl = (f"The pull failure for '{image}' timed out with 'context deadline exceeded', which could indicate "
                f"a network issue, DNS failure, or registry outage. The pull secret appears configured for the "
                f"correct registry. Cannot determine root cause without network-level diagnostics or retrying the pull. "
                f"Recommend checking node DNS resolution and egress network policies.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "low"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_crashloop(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "config_error")
    log_line = _extract_log_line(inst)
    ev = _extract_event_message(inst)
    restarts = _extract_restarts(inst)
    exit_code = _extract_exit_code(inst)

    if ttype == "direct":
        trace = [
            {"tool_name": "get_logs",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname, "previous": True, "tail": 30},
             "tool_result": {"name": rname, "namespace": ns, "container": "main",
                             "log_lines": [log_line or resp.get("explanation", "fatal error")],
                             "exit_code": exit_code, "restart_count": restarts}},
        ]
        if sub in ("missing_env", "config_error", "scc"):
            trace.append({"tool_name": "get_config",
                          "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
                          "tool_result": {"name": rname, "kind": rtype, "namespace": ns,
                                          "containers": [{"name": "main",
                                                          "image": f"registry.example.com/{rname}:latest",
                                                          "env_vars": [],
                                                          "volume_mounts": [],
                                                          "resources": {"requests": {"cpu": "250m", "memory": "256Mi"},
                                                                        "limits": {"cpu": "500m", "memory": "512Mi"}}}]}})
        elif sub in ("db_connection", "dependency", "network"):
            svc_name = vp._rng.choice(["postgres", "mysql", "redis", "kafka", "rabbitmq", "mongodb"])
            trace.append({"tool_name": "get_endpoints",
                          "arguments": {"namespace": ns, "service_name": svc_name},
                          "tool_result": {"service": svc_name, "namespace": ns,
                                          "endpoints": [], "ready_count": 0,
                                          "not_ready_count": vp._rng.randint(1, 3)}})
        else:
            trace.append({"tool_name": "get_pod_events",
                          "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
                          "tool_result": {"name": rname, "namespace": ns,
                                          "events": [
                                              {"type": "Warning", "reason": "BackOff",
                                               "message": "Back-off restarting failed container",
                                               "age": f"{vp._rng.randint(0, 5)}m"},
                                              {"type": "Normal", "reason": "Pulled",
                                               "message": f"Container image pulled successfully",
                                               "age": f"{vp._rng.randint(5, 15)}m"}]}})
        expl = (f"Container logs from the previous crash show: '{log_line[:150]}'. "
                f"{resp.get('explanation', '')}") if log_line else resp.get("explanation", "")

    elif ttype == "disambig":
        trace = [
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [
                                 {"type": "Warning", "reason": "BackOff",
                                  "message": f"Back-off restarting failed container (exit code: {exit_code or 1})",
                                  "count": restarts or vp._rng.randint(3, 20),
                                  "age": f"{vp._rng.randint(1, 30)}m"}]}},
            {"tool_name": "get_logs",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname, "previous": True, "tail": 30},
             "tool_result": {"name": rname, "namespace": ns, "container": "main",
                             "log_lines": [log_line or resp.get("explanation", "error")],
                             "exit_code": exit_code, "restart_count": restarts}},
        ]
        expl = (f"Pod events show {restarts or 'multiple'} restarts with exit code {exit_code or 1}, "
                f"but the event alone does not reveal the cause. Pulling the container logs from the "
                f"previous crash reveals: '{log_line[:150]}'. {resp.get('explanation', '')}")

    else:
        trace = [
            {"tool_name": "get_logs",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname, "previous": True, "tail": 30},
             "tool_result": {"name": rname, "namespace": ns, "container": "main",
                             "log_lines": ["Starting application...",
                                           f"WARN: retrying connection to upstream service",
                                           f"Process exited unexpectedly"],
                             "exit_code": exit_code or 1,
                             "restart_count": restarts or vp._rng.randint(2, 8)}},
            {"tool_name": "get_pod_events",
             "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
             "tool_result": {"name": rname, "namespace": ns,
                             "events": [
                                 {"type": "Warning", "reason": "BackOff",
                                  "message": "Back-off restarting failed container",
                                  "age": f"{vp._rng.randint(1, 30)}m"}]}},
        ]
        expl = (f"Container logs show a generic startup sequence followed by an unexpected exit "
                f"(code {exit_code or 1}), but no clear error message is present. Pod events confirm "
                f"repeated restarts. The root cause is ambiguous — it could be a timing-dependent "
                f"dependency failure, a configuration issue, or an intermittent resource constraint. "
                f"Recommend checking the full startup log sequence and inspecting the deployment's "
                f"environment variables and mounted secrets.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "low"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_route_503(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "timeout")
    route_name = _extract_route_name(inst)
    tls = "edge"
    if "re-encrypt" in inst.lower():
        tls = "reencrypt"
    elif "passthrough" in inst.lower():
        tls = "passthrough"

    svc_target = route_name.replace("-route", "").replace("-app", "-svc")
    has_endpoints = sub not in ("selector", "readiness")

    m = re.search(r"Service selector:\s*(\S+)", inst)
    svc_selector = m.group(1).rstrip(";,.") if m else f"app={rname}"
    m = re.search(r"Pod labels:\s*(\S+)", inst)
    pod_labels = m.group(1).rstrip(";,.") if m else svc_selector

    if ttype == "direct":
        eps = []
        if has_endpoints:
            for _ in range(vp._rng.randint(1, 3)):
                eps.append({"ip": vp.ip(), "port": vp.port(), "ready": True})
        trace = [
            {"tool_name": "get_route",
             "arguments": {"namespace": ns, "route_name": route_name},
             "tool_result": {"name": route_name, "namespace": ns,
                             "host": f"{route_name}.apps.cluster.example.com",
                             "tls_termination": tls,
                             "target_service": svc_target,
                             "target_port": vp.port(),
                             "admitted": True}},
            {"tool_name": "get_endpoints",
             "arguments": {"namespace": ns, "service_name": svc_target},
             "tool_result": {"service": svc_target, "namespace": ns,
                             "endpoints": eps, "ready_count": len(eps),
                             "not_ready_count": 0 if has_endpoints else vp._rng.randint(1, 3)}},
        ]
        if sub == "selector":
            trace[1]["tool_result"]["endpoints"] = []
            trace[1]["tool_result"]["ready_count"] = 0
            trace[1]["tool_result"]["selector"] = svc_selector
            trace[1]["tool_result"]["pod_labels_sample"] = pod_labels
        expl = (f"The route '{route_name}' (TLS: {tls}) targets service '{svc_target}'. "
                f"Endpoint inspection shows {len(eps)} ready backends. "
                f"{resp.get('explanation', '')}")

    elif ttype == "disambig":
        trace = [
            {"tool_name": "get_route",
             "arguments": {"namespace": ns, "route_name": route_name},
             "tool_result": {"name": route_name, "namespace": ns,
                             "host": f"{route_name}.apps.cluster.example.com",
                             "tls_termination": tls,
                             "target_service": svc_target,
                             "target_port": vp.port(),
                             "admitted": True}},
            {"tool_name": "get_endpoints",
             "arguments": {"namespace": ns, "service_name": svc_target},
             "tool_result": {"service": svc_target, "namespace": ns,
                             "endpoints": [], "ready_count": 0,
                             "not_ready_count": vp._rng.randint(1, 3),
                             "selector": svc_selector}},
        ]
        expl = (f"The route '{route_name}' appears correctly configured and admitted. However, "
                f"the target service '{svc_target}' has 0 ready endpoints. "
                f"{resp.get('explanation', '')}")

    else:
        eps = [{"ip": vp.ip(), "port": vp.port(), "ready": True}]
        trace = [
            {"tool_name": "get_route",
             "arguments": {"namespace": ns, "route_name": route_name},
             "tool_result": {"name": route_name, "namespace": ns,
                             "host": f"{route_name}.apps.cluster.example.com",
                             "tls_termination": tls,
                             "target_service": svc_target,
                             "target_port": vp.port(),
                             "admitted": True}},
            {"tool_name": "get_endpoints",
             "arguments": {"namespace": ns, "service_name": svc_target},
             "tool_result": {"service": svc_target, "namespace": ns,
                             "endpoints": eps, "ready_count": 1}},
        ]
        expl = (f"The route '{route_name}' is admitted and the service '{svc_target}' shows "
                f"{len(eps)} ready endpoint(s). Both appear healthy from a control-plane perspective. "
                f"The intermittent 503 may be caused by backend application latency, connection "
                f"exhaustion, or HAProxy timeout configuration. Further investigation into "
                f"application-level response times and router logs is needed.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "medium"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_pvc_pending(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "csi_driver")
    pvc_name = _extract_pvc_name(inst)
    ev = _extract_event_message(inst)
    sc_name = _extract_sc_name(inst) or vp.storage_class()

    provisioner_map = {
        "csi_driver": "ebs.csi.aws.com",
        "vsphere": "csi.vsphere.vmware.com",
        "no_sc": None,
        "snapshot": "ebs.csi.aws.com",
        "capacity": "ebs.csi.aws.com",
        "access_mode": "ebs.csi.aws.com",
    }
    prov = provisioner_map.get(sub, vp.provisioner())

    if ttype == "direct":
        trace = [
            {"tool_name": "get_pvc",
             "arguments": {"namespace": ns, "pvc_name": pvc_name},
             "tool_result": {"name": pvc_name, "namespace": ns, "status": "Pending",
                             "storage_class": sc_name if sub != "no_sc" else "<none>",
                             "requested_size": vp._rng.choice(["10Gi", "20Gi", "50Gi", "100Gi"]),
                             "access_modes": ["ReadWriteOnce"],
                             "events": [{"type": "Warning",
                                         "reason": "ProvisioningFailed",
                                         "message": ev[:250] or "provisioning failed"}]}},
            {"tool_name": "get_storage_class",
             "arguments": {"storage_class_name": sc_name},
             "tool_result": {"name": sc_name,
                             "provisioner": prov or "none",
                             "reclaim_policy": "Delete",
                             "volume_binding_mode": "WaitForFirstConsumer",
                             "allow_expansion": sub != "snapshot",
                             "exists": sub != "no_sc"}},
        ]
        expl = (f"PVC '{pvc_name}' is stuck in Pending. The provisioning event reports: "
                f"'{ev[:150]}'. StorageClass '{sc_name}' uses provisioner '{prov}'. "
                f"{resp.get('explanation', '')}")

    elif ttype == "disambig":
        trace = [
            {"tool_name": "get_pvc",
             "arguments": {"namespace": ns, "pvc_name": pvc_name},
             "tool_result": {"name": pvc_name, "namespace": ns, "status": "Pending",
                             "storage_class": sc_name,
                             "requested_size": vp._rng.choice(["10Gi", "20Gi", "50Gi"]),
                             "access_modes": ["ReadWriteOnce"],
                             "events": [{"type": "Warning",
                                         "reason": "ProvisioningFailed",
                                         "message": "waiting for first consumer to be created before binding"}]}},
            {"tool_name": "get_storage_class",
             "arguments": {"storage_class_name": sc_name},
             "tool_result": {"name": sc_name,
                             "provisioner": prov or "none",
                             "reclaim_policy": "Delete",
                             "volume_binding_mode": "WaitForFirstConsumer",
                             "allow_expansion": True,
                             "exists": True}},
        ]
        expl = (f"PVC '{pvc_name}' shows a generic 'waiting for first consumer' message. "
                f"Checking the StorageClass '{sc_name}' reveals WaitForFirstConsumer binding mode, "
                f"which means provisioning will not occur until a pod actually mounts the PVC. "
                f"{resp.get('explanation', '')}")

    else:
        trace = [
            {"tool_name": "get_pvc",
             "arguments": {"namespace": ns, "pvc_name": pvc_name},
             "tool_result": {"name": pvc_name, "namespace": ns, "status": "Pending",
                             "storage_class": sc_name,
                             "requested_size": "50Gi",
                             "access_modes": ["ReadWriteOnce"],
                             "events": []}},
            {"tool_name": "get_storage_class",
             "arguments": {"storage_class_name": sc_name},
             "tool_result": {"name": sc_name,
                             "provisioner": prov or "ebs.csi.aws.com",
                             "reclaim_policy": "Delete",
                             "volume_binding_mode": "Immediate",
                             "allow_expansion": True,
                             "exists": True}},
        ]
        expl = (f"PVC '{pvc_name}' is Pending with no events recorded, and the StorageClass "
                f"'{sc_name}' appears correctly configured with Immediate binding. The provisioner "
                f"may be unhealthy, or there could be a cloud provider API issue. Recommend checking "
                f"the CSI driver pods in the openshift-cluster-csi-drivers namespace and cloud provider quotas.")

    conf = resp.get("confidence", "high") if ttype != "uncertain" else "low"
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


def _build_unknown(inst: str, resp: dict, ttype: str, vp: ValuePool, ns: str, rtype: str, rname: str) -> dict:
    sub = resp.get("sub_cause", "app_logic")
    ev = _extract_event_message(inst)
    log_line = _extract_log_line(inst)

    trace = [
        {"tool_name": "get_pod_events",
         "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname},
         "tool_result": {"name": rname, "namespace": ns,
                         "events": [{"type": "Warning", "reason": "Unknown",
                                     "message": ev[:200] or "No clear event available",
                                     "age": f"{vp._rng.randint(5, 60)}m"}] if ev else []}},
    ]
    if sub in ("node_flapping",):
        trace.append({"tool_name": "get_nodes", "arguments": {},
                      "tool_result": {"total_nodes": 6, "master_nodes": 3,
                                      "worker_nodes": vp.workers(3)}})
    else:
        trace.append({"tool_name": "get_logs",
                      "arguments": {"namespace": ns, "resource_type": rtype, "resource_name": rname, "tail": 30},
                      "tool_result": {"name": rname, "namespace": ns, "container": "main",
                                      "log_lines": [log_line or "No obvious error in recent logs"],
                                      "exit_code": None}})

    expl = resp.get("explanation", "Root cause is unclear from available evidence.")
    if ttype == "uncertain" or resp.get("confidence") == "low":
        expl += " Further evidence gathering is needed before a confident diagnosis can be made."

    conf = "low" if ttype == "uncertain" else resp.get("confidence", "low")
    return {"trace": trace, "explanation": expl, "confidence": conf,
            "tools_used": [t["tool_name"] for t in trace]}


_BUILDERS = {
    "quota_exceeded": _build_quota_exceeded,
    "scheduling_constraint": _build_scheduling,
    "image_pull_error": _build_image_pull,
    "crashloop_backoff": _build_crashloop,
    "route_503": _build_route_503,
    "pvc_pending": _build_pvc_pending,
    "unknown": _build_unknown,
}


# ---------------------------------------------------------------------------
# Main rewriter
# ---------------------------------------------------------------------------

def rewrite_example(original: dict) -> dict:
    src = original["source_case"]
    inst = src["instruction"]
    resp = src["legacy_response"]
    case_id = original["case_id"]

    category = resp.get("category", "unknown")
    sub_cause = resp.get("sub_cause", "unknown")
    fix = resp.get("fix", [])
    commands = resp.get("commands", [])
    verification = resp.get("verification", "")

    rtype, rname = _extract_resource(inst)
    vp = ValuePool(case_id)
    ns = _infer_namespace(inst, vp)
    ttype = _trace_type(case_id)

    builder = _BUILDERS.get(category, _build_unknown)
    result = builder(inst, resp, ttype, vp, ns, rtype, rname)

    rec_action = _recommended_action(commands, fix)

    return {
        "case_id": case_id,
        "source_case": src,
        "agent_training_example": {
            "user_request": _normalize_request(inst),
            "context": {
                "namespace": ns,
                "resource_type": rtype,
                "resource_name": rname,
            },
            "policy": {
                "must_ground_with_tools": True,
                "mutations_require_approval": True,
            },
            "assistant_plan": {
                "goal": resp.get("explanation", "Investigate and diagnose the issue"),
                "needed_evidence": [
                    f"{t['tool_name']} output for {rname}" for t in result["trace"]
                ],
                "selected_tool_family": result["tools_used"],
            },
            "tool_trace": result["trace"],
            "recommended_action": rec_action,
            "final_response": {
                "category": category,
                "sub_cause": sub_cause,
                "confidence": result["confidence"],
                "explanation": result["explanation"],
                "fix": fix,
                "commands": commands,
                "verification": verification,
                "used_tools": result["tools_used"],
                "safe_to_auto_apply": rec_action["type"] == "read_only",
                "requires_user_approval": rec_action["requires_approval"],
            },
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ocp-agent-instructions.jsonl")
    parser.add_argument("--output", default="data/ocp-agent-instructions-v2.jsonl")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    count = 0
    errors = 0
    dist = {"direct": 0, "disambig": 0, "uncertain": 0}
    with open(inp) as fin, open(out, "w") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                orig = json.loads(line)
                rewritten = rewrite_example(orig)
                fout.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
                count += 1
                tt = _trace_type(orig["case_id"])
                dist[tt] += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] Line {idx+1}: {e}")

    print(f"Rewritten {count} examples ({errors} errors) -> {out}")
    total = sum(dist.values())
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")


if __name__ == "__main__":
    main()
