#!/usr/bin/env python3
"""Final evidence-alignment pass on the OCP agent dataset.

Fixes:
- PVC platform mismatches (vSphere/Ceph/hostPath with wrong provisioner)
- Crashloop dependency endpoint mismatches (e.g. Kafka issue checking MongoDB)
- Route endpoints with database ports (should use HTTP ports)
- Mutation approval re-verification
"""

import json
import re
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Platform-correct provisioner mapping
# ---------------------------------------------------------------------------

PLATFORM_PROVISIONERS = {
    "vsphere":  ("thin-csi", "csi.vsphere.vmware.com", "Delete", "Immediate"),
    "ceph":     ("ocs-storagecluster-ceph-rbd", "openshift-storage.rbd.csi.ceph.com", "Delete", "Immediate"),
    "odf":      ("ocs-storagecluster-ceph-rbd", "openshift-storage.rbd.csi.ceph.com", "Delete", "Immediate"),
    "hostpath": ("local-storage", "kubernetes.io/no-provisioner", "Retain", "WaitForFirstConsumer"),
    "local":    ("local-storage", "kubernetes.io/no-provisioner", "Retain", "WaitForFirstConsumer"),
    "nfs":      ("nfs-client", "cluster.local/nfs-subdir-external-provisioner", "Delete", "Immediate"),
    "aws":      ("gp3-csi", "ebs.csi.aws.com", "Delete", "WaitForFirstConsumer"),
    "ebs":      ("gp3-csi", "ebs.csi.aws.com", "Delete", "WaitForFirstConsumer"),
    "gcp":      ("standard-csi", "pd.csi.storage.gke.io", "Delete", "WaitForFirstConsumer"),
    "azure":    ("managed-premium", "disk.csi.azure.com", "Delete", "WaitForFirstConsumer"),
}

# ---------------------------------------------------------------------------
# Dependency keyword → service name mapping
# ---------------------------------------------------------------------------

DEP_SVC_MAP = {
    "kafka":         ("kafka-bootstrap", 9092),
    "redis":         ("redis-master", 6379),
    "postgres":      ("postgres", 5432),
    "mysql":         ("mysql", 3306),
    "mongo":         ("mongodb", 27017),
    "mongodb":       ("mongodb", 27017),
    "rabbitmq":      ("rabbitmq", 5672),
    "elasticsearch": ("elasticsearch", 9200),
    "memcached":     ("memcached", 11211),
    "cassandra":     ("cassandra", 9042),
    "nats":          ("nats", 4222),
    "mariadb":       ("mariadb", 3306),
    "etcd":          ("etcd", 2379),
}

# HTTP ports suitable for route endpoints
HTTP_PORTS = [8080, 8443, 3000, 5000, 9090, 8081, 8888, 4317, 9411, 8000, 443, 80]

DB_PORTS = {3306, 5432, 6379, 27017, 9042, 11211, 5672, 2379}


def _detect_platform(instruction: str) -> str | None:
    text = instruction.lower()
    for kw in ["vsphere", "vcenter", "vmware", "vsan"]:
        if kw in text:
            return "vsphere"
    for kw in ["ceph", "odf", "openshift data foundation", "ocs-storagecluster"]:
        if kw in text:
            return "ceph"
    for kw in ["hostpath", "host path", "local-storage", "local storage", "hostPath"]:
        if kw in text and "local-ssd" not in text:
            return "hostpath"
    for kw in ["nfs"]:
        if kw in text:
            return "nfs"
    for kw in ["ebs", "aws"]:
        if kw in text:
            return "aws"
    for kw in ["gcp", "gke", "pd.csi"]:
        if kw in text:
            return "gcp"
    for kw in ["azure", "aks"]:
        if kw in text:
            return "azure"
    return None


def _detect_dependencies(instruction: str) -> list[str]:
    text = instruction.lower()
    found = []
    for kw in DEP_SVC_MAP:
        if kw in text:
            found.append(kw)
    return found


def _rng_for(case_id: str) -> random.Random:
    return random.Random(case_id + "_final")


# ---------------------------------------------------------------------------
# Fix 1: PVC platform mismatches
# ---------------------------------------------------------------------------

def fix_pvc_platform(ex: dict) -> dict:
    cat = ex["agent_training_example"]["final_response"]["category"]
    if cat != "pvc_pending":
        return ex

    inst = ex["source_case"]["instruction"]
    platform = _detect_platform(inst)
    if not platform:
        return ex

    pinfo = PLATFORM_PROVISIONERS.get(platform)
    if not pinfo:
        return ex

    sc_name, provisioner, reclaim, binding = pinfo

    for t in ex["agent_training_example"]["tool_trace"]:
        if t["tool_name"] == "get_storage_class":
            current_prov = t["tool_result"].get("provisioner", "")
            if platform == "vsphere" and "vsphere" in current_prov:
                continue
            if platform in ("ceph", "odf") and "ceph" in current_prov:
                continue
            if platform in ("hostpath", "local") and "no-provisioner" in current_prov:
                continue

            sc_from_inst = None
            m = re.search(r"StorageClass:\s*(\S+)", inst)
            if m:
                sc_from_inst = m.group(1).rstrip(";,.: ")

            t["tool_result"]["name"] = sc_from_inst or sc_name
            t["tool_result"]["provisioner"] = provisioner
            t["tool_result"]["reclaim_policy"] = reclaim
            t["tool_result"]["volume_binding_mode"] = binding
            if sc_from_inst:
                t["arguments"]["storage_class_name"] = sc_from_inst

        if t["tool_name"] == "get_pvc":
            if "storage_class" in t["tool_result"]:
                sc_from_inst = None
                m = re.search(r"StorageClass:\s*(\S+)", inst)
                if m:
                    sc_from_inst = m.group(1).rstrip(";,.: ")
                t["tool_result"]["storage_class"] = sc_from_inst or sc_name

    return ex


# ---------------------------------------------------------------------------
# Fix 2: Crashloop dependency endpoint mismatches
# ---------------------------------------------------------------------------

def fix_crashloop_dependency_endpoints(ex: dict) -> dict:
    cat = ex["agent_training_example"]["final_response"]["category"]
    sub = ex["agent_training_example"]["final_response"]["sub_cause"]
    if cat != "crashloop_backoff" or sub not in ("db_connection", "dependency", "network"):
        return ex

    inst = ex["source_case"]["instruction"]
    deps = _detect_dependencies(inst)
    if not deps:
        return ex

    primary_dep = deps[0]
    correct_svc, correct_port = DEP_SVC_MAP[primary_dep]
    ns = ex["agent_training_example"]["context"]["namespace"]

    for t in ex["agent_training_example"]["tool_trace"]:
        if t["tool_name"] != "get_endpoints":
            continue
        current_svc = t["tool_result"].get("service", "")
        if any(d in current_svc.lower() for d in deps):
            continue

        t["tool_result"]["service"] = correct_svc
        t["arguments"]["service_name"] = correct_svc
        if ns and ns != "unknown":
            t["tool_result"]["namespace"] = ns
            t["arguments"]["namespace"] = ns

    return ex


# ---------------------------------------------------------------------------
# Fix 3: Route endpoints with database ports
# ---------------------------------------------------------------------------

def fix_route_endpoint_ports(ex: dict) -> dict:
    cat = ex["agent_training_example"]["final_response"]["category"]
    if cat != "route_503":
        return ex

    rng = _rng_for(ex["case_id"])

    for t in ex["agent_training_example"]["tool_trace"]:
        if t["tool_name"] != "get_endpoints":
            continue
        eps = t["tool_result"].get("endpoints", [])
        for ep in eps:
            if isinstance(ep, dict):
                port = ep.get("port", 0)
                if port in DB_PORTS:
                    ep["port"] = rng.choice(HTTP_PORTS)

    return ex


# ---------------------------------------------------------------------------
# Fix 4: Re-verify mutation approval
# ---------------------------------------------------------------------------

def fix_approval_policy(ex: dict) -> dict:
    ra = ex["agent_training_example"]["recommended_action"]
    fr = ex["agent_training_example"]["final_response"]

    if ra["type"] == "mutation":
        ra["requires_approval"] = True
        if not ra.get("proposed_tool"):
            ra["proposed_tool"] = "patch_resource"
        fr["requires_user_approval"] = True
        fr["safe_to_auto_apply"] = False
    elif ra["type"] == "read_only":
        ra["requires_approval"] = False
        ra["proposed_tool"] = None
        fr["requires_user_approval"] = False
        fr["safe_to_auto_apply"] = False

    return ex


# ---------------------------------------------------------------------------
# Fix 5: Realign used_tools / selected_tool_family after any trace changes
# ---------------------------------------------------------------------------

def fix_tools_alignment(ex: dict) -> dict:
    tools = [t["tool_name"] for t in ex["agent_training_example"]["tool_trace"]]
    ex["agent_training_example"]["final_response"]["used_tools"] = tools
    ex["agent_training_example"]["assistant_plan"]["selected_tool_family"] = tools
    return ex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def repair(ex: dict) -> dict:
    ex = fix_pvc_platform(ex)
    ex = fix_crashloop_dependency_endpoints(ex)
    ex = fix_route_endpoint_ports(ex)
    ex = fix_approval_policy(ex)
    ex = fix_tools_alignment(ex)
    return ex


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ocp-agent-instructions-v3.jsonl")
    parser.add_argument("--output", default="data/ocp-agent-instructions-final.jsonl")
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
                repaired = repair(ex)
                fout.write(json.dumps(repaired, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                errors += 1
                print(f"[WARN] Line {idx+1}: {e}")

    print(f"Final alignment: {count} examples ({errors} errors) -> {args.output}")


if __name__ == "__main__":
    main()
