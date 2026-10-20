import json
import re
import hashlib
import datetime
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field, ValidationError, model_validator

try:
    import git
    GIT_AVAILABLE = True
except Exception:
    git = None
    GIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Domain models / structured schemas
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AgentVerdict(str, Enum):
    APPROVE = "APPROVE"
    CONCERN = "CONCERN"
    REJECT = "REJECT"


class FinalVerdict(str, Enum):
    MERGE = "MERGE"
    MERGE_WITH_CONDITIONS = "MERGE_WITH_CONDITIONS"
    BLOCK = "BLOCK"


class Phase(str, Enum):
    INDEPENDENT = "independent"
    DELIBERATION = "deliberation"
    VERDICT = "verdict"
    RECORD = "record"


class ChangedFile(BaseModel):
    path: str
    additions: int = 0
    deletions: int = 0
    language: str = "text"
    risk_tag: Optional[str] = None


class TestResult(BaseModel):
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    coverage_changed_lines: float = 0.0
    flaky_history: List[str] = Field(default_factory=list)
    asserts_new_behavior: bool = False


class PRInput(BaseModel):
    id: str
    title: str
    author: str = "synthetic"
    category: str = "custom"
    base_branch: str = "main"
    head_branch: str = "feature"
    description: str = ""
    diff: str = ""
    changed_files: List[ChangedFile] = Field(default_factory=list)
    tests: TestResult = Field(default_factory=TestResult)

    critical_path: bool = False
    migration: bool = False
    dependency_change: bool = False
    secret_pattern: bool = False
    bug_pattern: bool = False
    missing_tests: bool = False

    expected_verdict: FinalVerdict = FinalVerdict.MERGE


class Finding(BaseModel):
    id: str
    agent: str
    severity: Severity
    file: Optional[str] = None
    line: Optional[int] = None
    title: str
    detail: str
    recommendation: Optional[str] = None


class Assessment(BaseModel):
    agent: str
    verdict: AgentVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    findings: List[Finding] = Field(default_factory=list)
    rationale: str
    round: int = 0
    viewed_evidence: List[str] = Field(default_factory=list)
    change_reason: Optional[str] = None
    token_usage: int = 0
    cost: float = 0.0


class PositionChange(BaseModel):
    agent: str
    from_verdict: AgentVerdict
    to_verdict: AgentVerdict
    reason: str
    evidence_cited: List[str] = Field(default_factory=list)
    round: int = 0

    @model_validator(mode="after")
    def require_reason_if_position_changed(self):
        if self.from_verdict != self.to_verdict:
            if not self.reason or not self.reason.strip():
                raise ValueError("Position change requires an explicit cited reason.")
        return self


class BoardMessage(BaseModel):
    id: str
    sender: str
    recipient: Optional[str] = None
    phase: Phase
    round: int = 0
    kind: str
    content: Dict[str, Any]
    timestamp: str
    tokens: int = 0
    cost: float = 0.0


class DecisionRecord(BaseModel):
    pr_id: str
    final_verdict: FinalVerdict
    independent_verdict: FinalVerdict
    rule: str
    conditions: List[str] = Field(default_factory=list)
    blockers: List[Finding] = Field(default_factory=list)
    rounds: List[int] = Field(default_factory=list)
    independent_assessments: List[Assessment] = Field(default_factory=list)
    deliberation_rounds: List[List[Assessment]] = Field(default_factory=list)
    final_assessments: List[Assessment] = Field(default_factory=list)
    position_changes: List[PositionChange] = Field(default_factory=list)
    reasoning_trail: List[str] = Field(default_factory=list)
    token_total: int = 0
    cost_total: float = 0.0
    budget_exceeded: bool = False
    created_at: str = ""


# ---------------------------------------------------------------------------
# Shared inspectable message board
# ---------------------------------------------------------------------------

class MessageBoard:
    """
    All agent communication goes through this board.
    Agents do not call each other directly.
    """

    def __init__(self):
        self.messages: List[BoardMessage] = []
        self.token_total: int = 0
        self.cost_total: float = 0.0

    def post(
        self,
        sender: str,
        phase: Phase,
        kind: str,
        content: Dict[str, Any],
        round_number: int = 0,
        recipient: Optional[str] = None,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> BoardMessage:
        msg = BoardMessage(
            id=f"msg-{len(self.messages) + 1:04d}",
            sender=sender,
            recipient=recipient,
            phase=phase,
            round=round_number,
            kind=kind,
            content=content,
            timestamp=now_iso(),
            tokens=tokens,
            cost=cost,
        )
        self.messages.append(msg)
        self.token_total += tokens
        self.cost_total += cost
        return msg

    def assessments(self, round_number: Optional[int] = None) -> List[Assessment]:
        out = []
        for m in self.messages:
            if m.kind == "assessment":
                a = Assessment.model_validate(m.content)
                if round_number is None or a.round == round_number:
                    out.append(a)
        return out

    def snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        msgs = self.messages if limit is None else self.messages[-limit:]
        return [m.model_dump(mode="json") for m in msgs]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def line_of(text: str, pattern: str) -> Optional[int]:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    for i, line in enumerate(text.splitlines(), start=1):
        if rx.search(line):
            return i
    return None


def estimate_tokens_cost(prompt: str, evidence: Dict[str, Any], response: Dict[str, Any], cost_per_1k: float) -> Tuple[int, float]:
    payload = json.dumps({"prompt": prompt, "evidence": evidence, "response": response}, sort_keys=True, default=str)
    tokens = max(64, len(payload) // 4)
    cost = round((tokens / 1000.0) * float(cost_per_1k), 8)
    return tokens, cost


def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


SECRET_RE = re.compile(
    r"""(
        AKIA[0-9A-Z]{16}
        |-----BEGIN\s+(RSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----
        |password\s*[:=]\s*['"].+['"]
        |passwd\s*[:=]\s*['"].+['"]
        |api[_-]?key\s*[:=]\s*['"].+['"]
        |secret\s*[:=]\s*['"].+['"]
        |token\s*[:=]\s*['"].+['"]
    )""",
    re.IGNORECASE | re.VERBOSE,
)

MIGRATION_RE = re.compile(r"(ALTER\s+TABLE|CREATE\s+INDEX|DROP\s+COLUMN|DROP\s+TABLE|SET\s+NOT\s+NULL|migrations/)", re.IGNORECASE)
ROLLBACK_RE = re.compile(r"(ROLLBACK|BACKFILL|PHASED|DUAL\s+WRITE|COMPATIBILITY)", re.IGNORECASE)
IRREVERSIBLE_RE = re.compile(r"(DROP\s+COLUMN|DROP\s+TABLE|RENAME\s+COLUMN|SET\s+NOT\s+NULL)", re.IGNORECASE)
UNSAFE_RE = re.compile(r"(eval\(|os\.system|subprocess\.(run|call).*shell=True|pickle\.loads|yaml\.load\()", re.IGNORECASE)
PERMISSION_RE = re.compile(r"(chmod\s+777|0o777|public-read|Action.*\*|Effect.*Allow.*\*)", re.IGNORECASE)
DEPENDENCY_FILES = ("package.json", "requirements.txt", "go.mod", "Gemfile", "pom.xml", "yarn.lock", "poetry.lock")
CRITICAL_KEYWORDS = ("payment", "billing", "auth", "login", "checkout", "order", "refund", "security", "token", "secret")


# ---------------------------------------------------------------------------
# Agent system prompts
# In a production repo these would live under prompts/*.md or prompts/*.txt.
# They are embedded here because the user requested all app code in app.py.
# ---------------------------------------------------------------------------

PROMPTS = {
    "CodeReviewer": """
You are CodeReviewer, a senior engineer on a release council.
Judge correctness, readability, and design consistency with the codebase.
Return only structured JSON matching the Assessment schema.
Do not speculate beyond the evidence. Every finding must include severity and file/line when possible.
""",
    "TestAnalyst": """
You are TestAnalyst, a test engineering specialist on a release council.
Judge whether changed lines are covered, whether tests assert the new behavior, and whether flaky tests increase release risk.
Return only structured JSON matching the Assessment schema.
""",
    "SecurityAuditor": """
You are SecurityAuditor, a product security engineer on a release council.
Look for injection risks, secret leakage, unsafe deserialization, dangerous dependency changes, and permission widening.
Return only structured JSON matching the Assessment schema.
""",
    "ReleaseManager": """
You are ReleaseManager, a release engineer responsible for blast radius and rollback safety.
Consider migration safety, critical-path exposure, dependency churn, rollback difficulty, and operational risk.
Return only structured JSON matching the Assessment schema.
""",
    "Coordinator": """
You are Coordinator. You do not provide opinions.
You run the protocol, enforce turn limits, track budget, collect findings, and write the final decision record.
""",
}


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class CouncilAgent:
    name = "Agent"

    def __init__(self, board: MessageBoard, cost_per_1k: float):
        self.board = board
        self.cost_per_1k = cost_per_1k
        self.system_prompt = PROMPTS.get(self.name, "")

    def evidence_view(self, pr: PRInput, peer_findings: List[Dict[str, Any]], peer_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def _independent_response(self, pr: PRInput) -> Dict[str, Any]:
        raise NotImplementedError

    def _deliberate_response(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _finalize(self, evidence: Dict[str, Any], response: Dict[str, Any], round_number: int) -> Assessment:
        assessment = Assessment(
            agent=self.name,
            verdict=response["verdict"],
            confidence=clamp01(response["confidence"]),
            findings=response.get("findings", []),
            rationale=response.get("rationale", ""),
            round=round_number,
            viewed_evidence=list(evidence.keys()),
            change_reason=response.get("change_reason"),
        )
        tokens, cost = estimate_tokens_cost(
            self.system_prompt,
            evidence,
            assessment.model_dump(mode="json"),
            self.cost_per_1k,
        )
        assessment.token_usage = tokens
        assessment.cost = cost

        phase = Phase.INDEPENDENT if round_number == 0 else Phase.DELIBERATION
        self.board.post(
            sender=self.name,
            phase=phase,
            kind="assessment",
            round_number=round_number,
            content=assessment.model_dump(mode="json"),
            tokens=tokens,
            cost=cost,
        )
        return assessment

    def independent(self, pr: PRInput) -> Assessment:
        evidence = self.evidence_view(pr, [], [])
        response = self._independent_response(pr)
        return self._finalize(evidence, response, 0)

    def deliberate(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Assessment:
        evidence = self.evidence_view(pr, peer_findings, peer_verdicts)
        response = self._deliberate_response(pr, previous, peer_findings, peer_verdicts, round_number)
        return self._finalize(evidence, response, round_number)

    # Shared deliberation primitives -----------------------------------------

    def _upgrade_from_peer_critical(self, previous: Assessment, peer_findings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if previous.verdict == AgentVerdict.REJECT:
            return None
        crit = next((f for f in peer_findings if str(f.get("severity", "")).lower() == Severity.CRITICAL.value), None)
        if not crit:
            return None

        reason = (
            f"Revised to REJECT after citing critical finding {crit.get('id')} "
            f"from {crit.get('agent')}: {crit.get('title')}."
        )
        adopted = Finding(
            id=stable_id(self.name, "adopted", crit.get("id")),
            agent=self.name,
            severity=Severity.CRITICAL,
            title=f"Adopted critical risk from {crit.get('agent')}",
            detail=crit.get("detail", ""),
            file=crit.get("file"),
            line=crit.get("line"),
            recommendation=crit.get("recommendation"),
        )
        findings = list(previous.findings) + [adopted]
        rationale = previous.rationale + " " + reason
        return {
            "verdict": AgentVerdict.REJECT,
            "confidence": max(previous.confidence, 0.88),
            "findings": findings,
            "rationale": rationale,
            "change_reason": reason,
        }

    def _downgrade_if_supported(
        self,
        previous: Assessment,
        peer_verdicts: List[Dict[str, Any]],
        argument: str,
    ) -> Optional[Dict[str, Any]]:
        if previous.verdict != AgentVerdict.CONCERN:
            return None
        if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in previous.findings):
            return None

        strong_approvals = [
            v for v in peer_verdicts
            if v.get("verdict") == AgentVerdict.APPROVE.value and float(v.get("confidence", 0)) >= 0.75
        ]
        if len(strong_approvals) < 2:
            return None

        names = ", ".join(v.get("agent", "?") for v in strong_approvals)
        reason = f"Revised from CONCERN to APPROVE because {argument} (citing high-confidence approvals from {names})."

        new_findings = []
        for f in previous.findings:
            if f.severity == Severity.MEDIUM:
                new_findings.append(
                    f.model_copy(
                        update={
                            "severity": Severity.LOW,
                            "detail": f.detail + " Demoted to informational after deliberation.",
                        }
                    )
                )
            else:
                new_findings.append(f)

        rationale = previous.rationale + " " + reason
        return {
            "verdict": AgentVerdict.APPROVE,
            "confidence": clamp01(previous.confidence + 0.14),
            "findings": new_findings,
            "rationale": rationale,
            "change_reason": reason,
        }

    def _no_change(self, previous: Assessment, round_number: int) -> Dict[str, Any]:
        conf = clamp01(previous.confidence + 0.07)
        rationale = (
            previous.rationale
            + f" In round {round_number}, I reviewed peer findings and found no new evidence that changes my position."
        )
        return {
            "verdict": previous.verdict,
            "confidence": conf,
            "findings": previous.findings,
            "rationale": rationale,
            "change_reason": None,
        }


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------

class CodeReviewer(CouncilAgent):
    name = "CodeReviewer"

    def evidence_view(self, pr: PRInput, peer_findings: List[Dict[str, Any]], peer_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "diff": pr.diff,
            "changed_files": [f.model_dump(mode="json") for f in pr.changed_files],
            "description": pr.description,
            "peer_findings": peer_findings,
            "peer_verdicts": peer_verdicts,
        }

    def _independent_response(self, pr: PRInput) -> Dict[str, Any]:
        findings: List[Finding] = []
        verdict = AgentVerdict.APPROVE
        confidence = 0.88
        rationale = "The change appears locally coherent, readable, and consistent with surrounding design."

        first_file = pr.changed_files[0].path if pr.changed_files else "src/module.py"

        if pr.bug_pattern:
            sev = Severity.CRITICAL if pr.critical_path else Severity.HIGH
            line = line_of(pr.diff, r"(if|for|return|amount|price|refund|cursor|timezone)") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "subtle-bug", first_file, line),
                    agent=self.name,
                    severity=sev,
                    file=first_file,
                    line=line,
                    title="Subtle correctness defect in changed logic",
                    detail="The diff changes control flow or arithmetic without a clear invariant guard, creating edge-case failure risk.",
                    recommendation="Add explicit boundary handling and targeted regression tests before release.",
                )
            )
            verdict = AgentVerdict.REJECT if sev == Severity.CRITICAL else AgentVerdict.CONCERN
            confidence = 0.93 if sev == Severity.CRITICAL else 0.76
            rationale = "I found a correctness concern in the changed logic. The defect is subtle but material."

        elif re.search(r"TODO|FIXME|XXX", pr.diff, re.IGNORECASE):
            line = line_of(pr.diff, r"TODO|FIXME|XXX") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "todo", first_file, line),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=line,
                    title="Non-blocking maintainability debt",
                    detail="The change leaves a TODO/FIXME marker in production code.",
                    recommendation="Acceptable if tracked, but prefer a follow-up issue link.",
                )
            )
            verdict = AgentVerdict.CONCERN
            confidence = 0.68
            rationale = "The code is mostly fine, but the TODO reduces design cleanliness and may hide follow-up work."

        elif pr.migration and not ROLLBACK_RE.search(pr.diff):
            findings.append(
                Finding(
                    id=stable_id(self.name, "migration-design", first_file, 1),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=1,
                    title="Migration lacks explicit compatibility notes",
                    detail="Schema or migration change does not document rollback/backfill compatibility.",
                    recommendation="Add rollback notes or phased migration comments.",
                )
            )
            verdict = AgentVerdict.CONCERN
            confidence = 0.70
            rationale = "Design consistency is weak around migration safety documentation."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "findings": findings,
            "rationale": rationale,
            "change_reason": None,
        }

    def _deliberate_response(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Dict[str, Any]:
        upgrade = self._upgrade_from_peer_critical(previous, peer_findings)
        if upgrade:
            return upgrade

        downgrade = self._downgrade_if_supported(
            previous,
            peer_verdicts,
            argument="TestAnalyst verified behavior coverage and SecurityAuditor found no unsafe pattern",
        )
        if downgrade:
            return downgrade

        return self._no_change(previous, round_number)


class TestAnalyst(CouncilAgent):
    name = "TestAnalyst"

    def evidence_view(self, pr: PRInput, peer_findings: List[Dict[str, Any]], peer_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "tests": pr.tests.model_dump(mode="json"),
            "changed_files": [f.model_dump(mode="json") for f in pr.changed_files],
            "critical_path": pr.critical_path,
            "missing_tests": pr.missing_tests,
            "peer_findings": peer_findings,
            "peer_verdicts": peer_verdicts,
        }

    def _independent_response(self, pr: PRInput) -> Dict[str, Any]:
        findings: List[Finding] = []
        verdict = AgentVerdict.APPROVE
        confidence = 0.90
        rationale = "Tests pass, changed-line coverage is adequate, and the new behavior is asserted."

        t = pr.tests
        first_file = pr.changed_files[0].path if pr.changed_files else "tests/test_change.py"

        if t.failed > 0:
            findings.append(
                Finding(
                    id=stable_id(self.name, "failing-tests", first_file, 1),
                    agent=self.name,
                    severity=Severity.CRITICAL,
                    file=first_file,
                    line=1,
                    title="Failing tests in the PR",
                    detail=f"{t.failed} test(s) failed. The change cannot be considered release-safe.",
                    recommendation="Fix failing tests before any merge discussion.",
                )
            )
            return {
                "verdict": AgentVerdict.REJECT,
                "confidence": 0.97,
                "findings": findings,
                "rationale": "There are failing tests. This is a hard release blocker.",
                "change_reason": None,
            }

        if pr.missing_tests or t.coverage_changed_lines < 30 or not t.asserts_new_behavior:
            if pr.critical_path and (t.coverage_changed_lines < 20 or not t.asserts_new_behavior):
                sev = Severity.CRITICAL
                verdict = AgentVerdict.REJECT
                confidence = 0.93
                rationale = "Critical-path behavior lacks meaningful test assertion. This is not safe to release."
            elif t.coverage_changed_lines < 50 or not t.asserts_new_behavior:
                sev = Severity.HIGH
                verdict = AgentVerdict.CONCERN
                confidence = 0.78
                rationale = "Changed-line coverage or behavioral assertion is insufficient for confident release."
            else:
                sev = Severity.MEDIUM
                verdict = AgentVerdict.CONCERN
                confidence = 0.68
                rationale = "Tests exist but do not strongly assert the new behavior."

            findings.append(
                Finding(
                    id=stable_id(self.name, "weak-tests", first_file, 1),
                    agent=self.name,
                    severity=sev,
                    file=first_file,
                    line=1,
                    title="Insufficient test coverage of changed behavior",
                    detail=(
                        f"Changed-line coverage={t.coverage_changed_lines:.0f}%, "
                        f"asserts_new_behavior={t.asserts_new_behavior}, missing_tests={pr.missing_tests}."
                    ),
                    recommendation="Add tests that assert the new behavior and boundary cases.",
                )
            )

        if t.flaky_history:
            findings.append(
                Finding(
                    id=stable_id(self.name, "flaky", first_file, 2),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=2,
                    title="Flaky test risk",
                    detail="Recent history includes flaky tests touching this area.",
                    recommendation="Quarantine or stabilize flaky tests before release.",
                )
            )
            if verdict == AgentVerdict.APPROVE:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.72)
                rationale += " Flaky history lowers confidence."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "findings": findings,
            "rationale": rationale,
            "change_reason": None,
        }

    def _deliberate_response(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Dict[str, Any]:
        upgrade = self._upgrade_from_peer_critical(previous, peer_findings)
        if upgrade:
            return upgrade

        downgrade = self._downgrade_if_supported(
            previous,
            peer_verdicts,
            argument="CodeReviewer found no high-risk correctness defect and ReleaseManager bounded operational exposure",
        )
        if downgrade:
            return downgrade

        return self._no_change(previous, round_number)


class SecurityAuditor(CouncilAgent):
    name = "SecurityAuditor"

    def evidence_view(self, pr: PRInput, peer_findings: List[Dict[str, Any]], peer_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "diff": pr.diff,
            "changed_files": [f.model_dump(mode="json") for f in pr.changed_files],
            "dependency_change": pr.dependency_change,
            "secret_pattern": pr.secret_pattern,
            "peer_findings": peer_findings,
            "peer_verdicts": peer_verdicts,
        }

    def _independent_response(self, pr: PRInput) -> Dict[str, Any]:
        findings: List[Finding] = []
        verdict = AgentVerdict.APPROVE
        confidence = 0.92
        rationale = "No secret leakage, injection, unsafe deserialization, or permission widening was found in the evidence."

        first_file = pr.changed_files[0].path if pr.changed_files else "src/config.py"

        secret_match = SECRET_RE.search(pr.diff)
        if secret_match or pr.secret_pattern:
            line = line_of(pr.diff, r"AKIA|password|api[_-]?key|secret|token|PRIVATE KEY") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "secret", first_file, line),
                    agent=self.name,
                    severity=Severity.CRITICAL,
                    file=first_file,
                    line=line,
                    title="Hardcoded secret or credential detected",
                    detail="The diff contains a likely secret, API key, token, password, or private key material.",
                    recommendation="Rotate exposed credentials, remove secret from diff, and use a secret manager.",
                )
            )
            return {
                "verdict": AgentVerdict.REJECT,
                "confidence": 0.98,
                "findings": findings,
                "rationale": "A critical secret-leakage risk is present. This must block release.",
                "change_reason": None,
            }

        if UNSAFE_RE.search(pr.diff):
            line = line_of(pr.diff, r"eval\(|os\.system|shell=True|pickle\.loads|yaml\.load\(") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "unsafe", first_file, line),
                    agent=self.name,
                    severity=Severity.HIGH,
                    file=first_file,
                    line=line,
                    title="Unsafe execution or deserialization pattern",
                    detail="The diff introduces a pattern commonly associated with injection or unsafe deserialization.",
                    recommendation="Replace with safe APIs, validated parsers, or explicit allowlists.",
                )
            )
            verdict = AgentVerdict.CONCERN
            confidence = 0.84
            rationale = "Unsafe pattern detected. It is high risk but not automatically critical without proof of attacker control."

        if PERMISSION_RE.search(pr.diff):
            line = line_of(pr.diff, r"chmod|0o777|public-read|Action|Effect") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "permission", first_file, line),
                    agent=self.name,
                    severity=Severity.HIGH,
                    file=first_file,
                    line=line,
                    title="Permission widening detected",
                    detail="The change appears to broaden filesystem or IAM permissions.",
                    recommendation="Reduce permissions to least privilege and document justification.",
                )
            )
            if verdict != AgentVerdict.REJECT:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.82)
                rationale += " Permission widening increases attack surface."

        if pr.dependency_change or any(p.path.endswith(DEPENDENCY_FILES) for p in pr.changed_files):
            findings.append(
                Finding(
                    id=stable_id(self.name, "dependency", first_file, 1),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=1,
                    title="Dependency surface changed",
                    detail="The PR modifies dependency manifests or lockfiles.",
                    recommendation="Review changelogs, license changes, and transitive risk.",
                )
            )
            if verdict == AgentVerdict.APPROVE:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.74)
                rationale += " Dependency changes require supply-chain review."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "findings": findings,
            "rationale": rationale,
            "change_reason": None,
        }

    def _deliberate_response(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Dict[str, Any]:
        upgrade = self._upgrade_from_peer_critical(previous, peer_findings)
        if upgrade:
            return upgrade

        downgrade = self._downgrade_if_supported(
            previous,
            peer_verdicts,
            argument="CodeReviewer showed input handling is constrained and no critical security finding remains",
        )
        if downgrade:
            return downgrade

        return self._no_change(previous, round_number)


class ReleaseManager(CouncilAgent):
    name = "ReleaseManager"

    def evidence_view(self, pr: PRInput, peer_findings: List[Dict[str, Any]], peer_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "critical_path": pr.critical_path,
            "migration": pr.migration,
            "dependency_change": pr.dependency_change,
            "changed_files": [f.model_dump(mode="json") for f in pr.changed_files],
            "tests": pr.tests.model_dump(mode="json"),
            "diff": pr.diff,
            "peer_findings": peer_findings,
            "peer_verdicts": peer_verdicts,
        }

    def _independent_response(self, pr: PRInput) -> Dict[str, Any]:
        findings: List[Finding] = []
        verdict = AgentVerdict.APPROVE
        confidence = 0.89
        rationale = "Blast radius is acceptable, rollback appears straightforward, and no critical-path release risk is evident."

        first_file = pr.changed_files[0].path if pr.changed_files else "ops/release.py"

        irreversible = bool(IRREVERSIBLE_RE.search(pr.diff)) and not bool(ROLLBACK_RE.search(pr.diff))

        if pr.migration and irreversible:
            sev = Severity.CRITICAL if (pr.critical_path or "DROP" in pr.diff.upper()) else Severity.HIGH
            line = line_of(pr.diff, r"DROP|NOT NULL|RENAME") or 1
            findings.append(
                Finding(
                    id=stable_id(self.name, "migration", first_file, line),
                    agent=self.name,
                    severity=sev,
                    file=first_file,
                    line=line,
                    title="Irreversible or poorly rollback-able migration",
                    detail="The migration drops or constrains data without a documented rollback/backfill path.",
                    recommendation="Use expand-contract migration, add rollback plan, and verify backup/restore.",
                )
            )
            verdict = AgentVerdict.REJECT if sev == Severity.CRITICAL else AgentVerdict.CONCERN
            confidence = 0.95 if sev == Severity.CRITICAL else 0.81
            rationale = "Migration safety is the dominant release risk here."

        elif pr.migration:
            findings.append(
                Finding(
                    id=stable_id(self.name, "migration-review", first_file, 1),
                    agent=self.name,
                    severity=Severity.HIGH,
                    file=first_file,
                    line=1,
                    title="Migration requires staged rollout controls",
                    detail="Even reversible migrations can create operational risk on critical paths.",
                    recommendation="Add feature flag, canary rollout, and rollback runbook.",
                )
            )
            verdict = AgentVerdict.CONCERN
            confidence = 0.78
            rationale = "Migration touched release-sensitive surface and needs operational guardrails."

        if pr.critical_path and (pr.missing_tests or pr.tests.coverage_changed_lines < 50):
            findings.append(
                Finding(
                    id=stable_id(self.name, "critical-path-tests", first_file, 2),
                    agent=self.name,
                    severity=Severity.HIGH,
                    file=first_file,
                    line=2,
                    title="Critical path lacks sufficient verification",
                    detail="The change touches a critical path while test evidence is weak.",
                    recommendation="Require targeted integration tests and staged rollout.",
                )
            )
            if verdict == AgentVerdict.APPROVE:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.76)
                rationale += " Critical-path exposure raises rollback concern."

        if pr.dependency_change:
            findings.append(
                Finding(
                    id=stable_id(self.name, "dependency-release", first_file, 3),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=3,
                    title="Dependency change increases release coordination cost",
                    detail="Dependency updates may require broader regression validation.",
                    recommendation="Pin versions, review release notes, and validate in staging.",
                )
            )
            if verdict == AgentVerdict.APPROVE:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.73)
                rationale += " Dependency churn adds release risk."

        if len(pr.changed_files) > 5:
            findings.append(
                Finding(
                    id=stable_id(self.name, "blast-radius", first_file, 4),
                    agent=self.name,
                    severity=Severity.MEDIUM,
                    file=first_file,
                    line=4,
                    title="Large blast radius",
                    detail="The PR touches many files, increasing rollback difficulty.",
                    recommendation="Consider splitting the change into smaller releasable units.",
                )
            )
            if verdict == AgentVerdict.APPROVE:
                verdict = AgentVerdict.CONCERN
                confidence = min(confidence, 0.72)
                rationale += " Broad file surface lowers rollback confidence."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "findings": findings,
            "rationale": rationale,
            "change_reason": None,
        }

    def _deliberate_response(
        self,
        pr: PRInput,
        previous: Assessment,
        peer_findings: List[Dict[str, Any]],
        peer_verdicts: List[Dict[str, Any]],
        round_number: int,
    ) -> Dict[str, Any]:
        upgrade = self._upgrade_from_peer_critical(previous, peer_findings)
        if upgrade:
            return upgrade

        downgrade = self._downgrade_if_supported(
            previous,
            peer_verdicts,
            argument="council found no critical correctness, security, or test blocker and rollback risk is bounded",
        )
        if downgrade:
            return downgrade

        return self._no_change(previous, round_number)


AGENT_CLASSES = {
    "CodeReviewer": CodeReviewer,
    "TestAnalyst": TestAnalyst,
    "SecurityAuditor": SecurityAuditor,
    "ReleaseManager": ReleaseManager,
}

DEFAULT_AGENT_NAMES = ["CodeReviewer", "TestAnalyst", "SecurityAuditor", "ReleaseManager"]


# ---------------------------------------------------------------------------
# Coordinator state machine
# ---------------------------------------------------------------------------

class Coordinator:
    """
    Runs the protocol:
      Phase 1: independent assessment
      Phase 2: bounded deliberation
      Phase 3: documented verdict rule
      Phase 4: decision record construction
    """

    def __init__(
        self,
        board: MessageBoard,
        agents: List[CouncilAgent],
        max_rounds: int = 3,
        budget_cap: float = 5.0,
    ):
        self.board = board
        self.agents = agents
        self.max_rounds = max(0, int(max_rounds))
        self.budget_cap = float(budget_cap)

    def run(self, pr: PRInput) -> DecisionRecord:
        self.board.post(
            sender="Coordinator",
            phase=Phase.INDEPENDENT,
            kind="session_start",
            content={"pr_id": pr.id, "title": pr.title, "agents": [a.name for a in self.agents]},
        )

        # Phase 1 -------------------------------------------------------------
        independent: List[Assessment] = []
        for agent in self.agents:
            independent.append(agent.independent(pr))

        current = {a.agent: a for a in independent}
        baseline = independent_baseline(independent)
        reasoning_trail: List[str] = [
            f"Phase 1 complete. Independent baseline by naive majority: {baseline.value}.",
            f"Independent verdicts: {', '.join(f'{a.agent}={a.verdict.value}@{a.confidence:.2f}' for a in independent)}.",
        ]

        budget_exceeded = self.board.cost_total >= self.budget_cap
        if budget_exceeded:
            reasoning_trail.append("Budget cap exceeded during independent phase; deliberation will not start.")

        # Phase 2 -------------------------------------------------------------
        need, trigger = self.should_deliberate(independent)
        rounds_used: List[int] = []
        deliberation_rounds: List[List[Assessment]] = []
        position_changes: List[PositionChange] = []

        if need and not budget_exceeded:
            reasoning_trail.append(f"Deliberation triggered: {trigger}.")
            self.board.post(
                sender="Coordinator",
                phase=Phase.DELIBERATION,
                kind="deliberation_open",
                content={"reason": trigger, "max_rounds": self.max_rounds},
            )

            for r in range(1, self.max_rounds + 1):
                if self.board.cost_total >= self.budget_cap:
                    budget_exceeded = True
                    reasoning_trail.append(f"Budget cap reached before round {r}; stopping deliberation.")
                    break

                prev_snapshot = dict(current)
                all_findings = [
                    f.model_dump(mode="json")
                    for a in current.values()
                    for f in a.findings
                ]
                all_verdicts = [
                    {"agent": a.agent, "verdict": a.verdict.value, "confidence": a.confidence}
                    for a in current.values()
                ]

                self.board.post(
                    sender="Coordinator",
                    phase=Phase.DELIBERATION,
                    kind="round_open",
                    round_number=r,
                    content={"round": r},
                )

                round_assessments: List[Assessment] = []

                for agent in self.agents:
                    prev = prev_snapshot[agent.name]
                    peer_findings = [f for f in all_findings if f.get("agent") != agent.name]
                    peer_verdicts = [v for v in all_verdicts if v.get("agent") != agent.name]

                    new = agent.deliberate(pr, prev, peer_findings, peer_verdicts, r)

                    if new.verdict != prev.verdict:
                        reason = new.change_reason or ""
                        try:
                            pc = PositionChange(
                                agent=agent.name,
                                from_verdict=prev.verdict,
                                to_verdict=new.verdict,
                                reason=reason,
                                evidence_cited=[
                                    f.get("id", "")
                                    for f in peer_findings
                                    if f.get("severity") == Severity.CRITICAL.value
                                ],
                                round=r,
                            )
                            position_changes.append(pc)
                            reasoning_trail.append(
                                f"Round {r}: {pc.agent} changed {pc.from_verdict.value} -> {pc.to_verdict.value}. "
                                f"Reason: {pc.reason}"
                            )
                            self.board.post(
                                sender="Coordinator",
                                phase=Phase.DELIBERATION,
                                kind="position_change",
                                round_number=r,
                                content=pc.model_dump(mode="json"),
                            )
                        except ValidationError:
                            new = prev
                            reasoning_trail.append(
                                f"Round {r}: {agent.name} attempted an uncited position change. Protocol reverted it."
                            )

                    round_assessments.append(new)
                    current[new.agent] = new

                deliberation_rounds.append(round_assessments)
                rounds_used.append(r)

                if self.converged(list(current.values())):
                    reasoning_trail.append(f"Round {r}: council converged.")
                    break

                if self.board.cost_total >= self.budget_cap:
                    budget_exceeded = True
                    reasoning_trail.append(f"Budget cap reached after round {r}; stopping deliberation.")
                    break
        else:
            if not need:
                reasoning_trail.append("No deliberation required: independent positions were sufficiently aligned and confident.")
            else:
                reasoning_trail.append("Deliberation suppressed because budget was already exceeded.")

        # Phase 3 -------------------------------------------------------------
        final_assessments = list(current.values())
        final_verdict, rule, blockers, conditions = self.compute_final_verdict(final_assessments, budget_exceeded)
        reasoning_trail.append(f"Final rule applied: {rule}")

        # Phase 4 -------------------------------------------------------------
        record = DecisionRecord(
            pr_id=pr.id,
            final_verdict=final_verdict,
            independent_verdict=baseline,
            rule=rule,
            conditions=conditions,
            blockers=blockers,
            rounds=rounds_used,
            independent_assessments=independent,
            deliberation_rounds=deliberation_rounds,
            final_assessments=final_assessments,
            position_changes=position_changes,
            reasoning_trail=reasoning_trail,
            token_total=self.board.token_total,
            cost_total=round(self.board.cost_total, 8),
            budget_exceeded=budget_exceeded,
            created_at=now_iso(),
        )

        self.board.post(
            sender="Coordinator",
            phase=Phase.VERDICT,
            kind="final_decision",
            content=record.model_dump(mode="json"),
        )

        return record

    def should_deliberate(self, assessments: List[Assessment]) -> Tuple[bool, str]:
        if not assessments:
            return False, "no assessments"
        verdicts = {a.verdict for a in assessments}
        min_conf = min(a.confidence for a in assessments)
        reasons = []
        if len(verdicts) > 1:
            reasons.append("verdict disagreement")
        if min_conf < 0.6:
            reasons.append(f"minimum confidence {min_conf:.2f} < 0.60")
        if reasons:
            return True, "; ".join(reasons)
        return False, "aligned and confident"

    def converged(self, assessments: List[Assessment]) -> bool:
        if not assessments:
            return True
        verdicts = {a.verdict for a in assessments}
        min_conf = min(a.confidence for a in assessments)
        return len(verdicts) == 1 and min_conf >= 0.6

    def compute_final_verdict(
        self,
        assessments: List[Assessment],
        budget_exceeded: bool,
    ) -> Tuple[FinalVerdict, str, List[Finding], List[str]]:
        if budget_exceeded:
            return (
                FinalVerdict.BLOCK,
                "Budget cap exceeded; safe default is BLOCK.",
                [],
                ["Human release manager review required after automated budget overrun."],
            )

        blockers: List[Finding] = []
        for a in assessments:
            if a.verdict == AgentVerdict.REJECT:
                for f in a.findings:
                    if f.severity == Severity.CRITICAL:
                        blockers.append(f)

        if blockers:
            conditions = [f.recommendation or f.title for f in blockers]
            for a in assessments:
                for f in a.findings:
                    if f.severity in (Severity.HIGH, Severity.MEDIUM):
                        conditions.append(f.recommendation or f.title)
            return (
                FinalVerdict.BLOCK,
                "Any REJECT with severity critical blocks the merge.",
                blockers,
                dedupe(conditions),
            )

        verdicts = {a.verdict for a in assessments}
        if verdicts == {AgentVerdict.APPROVE}:
            return (
                FinalVerdict.MERGE,
                "All agents approved and no critical blocker exists.",
                [],
                [],
            )

        conditions = []
        for a in assessments:
            for f in a.findings:
                if f.severity in (Severity.HIGH, Severity.MEDIUM, Severity.CRITICAL):
                    conditions.append(f.recommendation or f.title)
        if not conditions:
            conditions = ["Resolve remaining council concerns before release."]

        return (
            FinalVerdict.MERGE_WITH_CONDITIONS,
            "Unresolved disagreement or non-critical concerns; merge only after conditions are satisfied.",
            [],
            dedupe(conditions),
        )


def independent_baseline(assessments: List[Assessment]) -> FinalVerdict:
    """
    Naive independent baseline used for evaluation:
    majority of Phase-1 verdicts, without the council's final critical-blocker rule.
    """
    if not assessments:
        return FinalVerdict.MERGE_WITH_CONDITIONS
    counts = Counter(a.verdict for a in assessments)
    n = len(assessments)
    if counts[AgentVerdict.REJECT] > n / 2:
        return FinalVerdict.BLOCK
    if counts[AgentVerdict.APPROVE] > n / 2 and counts[AgentVerdict.REJECT] == 0:
        return FinalVerdict.MERGE
    return FinalVerdict.MERGE_WITH_CONDITIONS


def run_council(
    pr: PRInput,
    agent_names: Optional[List[str]] = None,
    max_rounds: int = 3,
    budget_cap: float = 5.0,
    cost_per_1k: float = 0.02,
) -> Tuple[DecisionRecord, MessageBoard]:
    board = MessageBoard()
    names = agent_names or DEFAULT_AGENT_NAMES
    agents = [AGENT_CLASSES[n](board, cost_per_1k) for n in names]
    coordinator = Coordinator(board, agents, max_rounds=max_rounds, budget_cap=budget_cap)
    record = coordinator.run(pr)
    return record, board


# ---------------------------------------------------------------------------
# Benchmark dataset generator
# ---------------------------------------------------------------------------

def cf(path: str, additions: int = 10, deletions: int = 2, language: str = "python", risk_tag: Optional[str] = None) -> ChangedFile:
    return ChangedFile(path=path, additions=additions, deletions=deletions, language=language, risk_tag=risk_tag)


def tr(
    passed: int = 10,
    failed: int = 0,
    skipped: int = 0,
    coverage: float = 90.0,
    asserts: bool = True,
    flaky: Optional[List[str]] = None,
) -> TestResult:
    return TestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        coverage_changed_lines=coverage,
        flaky_history=flaky or [],
        asserts_new_behavior=asserts,
    )


def generate_benchmark() -> List[PRInput]:
    prs: List[PRInput] = []

    # 6 clean changes, last two include non-blocking TODO to exercise deliberation
    modules = ["billing", "auth", "ui", "api", "worker", "notifications"]
    for i, module in enumerate(modules):
        todo = i >= 4
        if todo:
            diff = (
                f"+def format_{module}(value):\n"
                f"+    # TODO: extract constant after follow-up issue\n"
                f"+    return str(value).strip()\n"
            )
        else:
            diff = (
                f"+def format_{module}(value):\n"
                f"+    return str(value).strip()\n"
            )
        prs.append(
            PRInput(
                id=f"PR-CLEAN-{i+1}",
                category="clean",
                title=f"Clean {module} helper",
                description="Small, tested, low-risk helper change.",
                diff=diff,
                changed_files=[cf(f"src/{module}/utils.py", additions=6, deletions=1)],
                tests=tr(passed=12 + i, coverage=88 + i, asserts=True),
                expected_verdict=FinalVerdict.MERGE,
            )
        )

    # 6 subtle bugs: 3 critical/block, 3 non-critical/conditions
    bug_specs = [
        ("Off-by-one refund calculation", True),
        ("Empty cart null dereference", True),
        ("Cache invalidation race", True),
        ("Timezone conversion drift", False),
        ("Pagination cursor regression", False),
        ("Currency rounding mismatch", False),
    ]
    for i, (title, critical) in enumerate(bug_specs):
        diff = (
            "-    if amount > 0:\n"
            "+    if amount >= 0:\n"
            "+        process_refund(amount - 1)\n"
        )
        prs.append(
            PRInput(
                id=f"PR-BUG-{i+1}",
                category="subtle_bug",
                title=title,
                description="Subtle logic change with edge-case risk.",
                diff=diff,
                changed_files=[cf("src/payments/refund.py", additions=4, deletions=2, risk_tag="payments")],
                tests=tr(passed=8, failed=0, coverage=55 if critical else 62, asserts=not critical),
                critical_path=critical,
                bug_pattern=True,
                expected_verdict=FinalVerdict.BLOCK if critical else FinalVerdict.MERGE_WITH_CONDITIONS,
            )
        )

    # 6 missing-test PRs: 3 critical/block, 3 non-critical/conditions
    for i in range(6):
        critical = i < 3
        coverage = 12 if critical else 46
        prs.append(
            PRInput(
                id=f"PR-MISSING-{i+1}",
                category="missing_tests",
                title="Checkout behavior changed without assertions" if critical else "Minor behavior change with weak tests",
                description="Production behavior changed but tests do not assert the new contract.",
                diff=(
                    "+def charge_card(card, amount):\n"
                    "+    return gateway.charge(card, amount)\n"
                ),
                changed_files=[cf("src/checkout/service.py", additions=8, deletions=1, risk_tag="checkout")],
                tests=tr(passed=3, failed=0, coverage=coverage, asserts=False),
                critical_path=critical,
                missing_tests=True,
                expected_verdict=FinalVerdict.BLOCK if critical else FinalVerdict.MERGE_WITH_CONDITIONS,
            )
        )

    # 6 hardcoded secrets: all block
    secret_snippets = [
        'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"',
        'password = "SuperSecret123!"',
        'api_key: "sk-live-abcdef123456"',
        'token = "ghp_xxxxxxxxxxxxxxxxxxxx"',
        'SECRET_KEY = "rot13-secret-value"',
        'private_key = "-----BEGIN RSA PRIVATE KEY-----"',
    ]
    for i, snippet in enumerate(secret_snippets):
        prs.append(
            PRInput(
                id=f"PR-SECRET-{i+1}",
                category="hardcoded_secret",
                title="Configuration cleanup accidentally commits credential",
                description="A secret appears in the diff.",
                diff=f"+{snippet}\n",
                changed_files=[cf("config/prod.env.py", additions=2, deletions=0, risk_tag="secrets")],
                tests=tr(passed=5, failed=0, coverage=70, asserts=True),
                secret_pattern=True,
                expected_verdict=FinalVerdict.BLOCK,
            )
        )

    # 6 migration risks: 3 irreversible/block, 3 reversible/conditions
    for i in range(6):
        irreversible = i < 3
        if irreversible:
            diff = (
                "+ALTER TABLE orders DROP COLUMN legacy_tax;\n"
                "+ALTER TABLE orders ALTER COLUMN email SET NOT NULL;\n"
            )
        else:
            diff = (
                "+ALTER TABLE orders ADD COLUMN tax_v2 NUMERIC;\n"
                "+-- ROLLBACK: ALTER TABLE orders DROP COLUMN tax_v2;\n"
                "+-- BACKFILL: UPDATE orders SET tax_v2 = legacy_tax;\n"
            )
        prs.append(
            PRInput(
                id=f"PR-MIGRATION-{i+1}",
                category="migration_risk",
                title="Irreversible order schema migration" if irreversible else "Reversible tax schema migration",
                description="Database migration with operational release risk.",
                diff=diff,
                changed_files=[cf("migrations/0042_orders.sql", additions=6, deletions=1, risk_tag="migration")],
                tests=tr(passed=6, failed=0, coverage=58, asserts=True),
                migration=True,
                critical_path=(i % 2 == 0),
                expected_verdict=FinalVerdict.BLOCK if irreversible else FinalVerdict.MERGE_WITH_CONDITIONS,
            )
        )

    return prs


# ---------------------------------------------------------------------------
# Sandbox / GitPython helpers
# ---------------------------------------------------------------------------

def parse_changed_paths(diff: str) -> List[str]:
    paths = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:].strip())
        elif line.startswith("diff --git a/"):
            parts = line.split(" b/")
            if len(parts) == 2:
                paths.append(parts[1].strip())
    deduped = []
    seen = set()
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def build_pr_from_diff(diff: str, title: str = "Sandbox PR") -> PRInput:
    paths = parse_changed_paths(diff) or ["src/unknown.py"]
    files = []
    plus = len([l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")])
    minus = len([l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")])
    for p in paths:
        files.append(
            ChangedFile(
                path=p,
                additions=max(1, plus // max(1, len(paths))),
                deletions=max(0, minus // max(1, len(paths))),
            )
        )

    secret = bool(SECRET_RE.search(diff))
    migration = bool(MIGRATION_RE.search(diff)) or any("migration" in p.lower() for p in paths)
    dependency = any(p.endswith(DEPENDENCY_FILES) for p in paths)
    blob = (diff + " " + " ".join(paths)).lower()
    critical = any(k in blob for k in CRITICAL_KEYWORDS)
    has_test = any("test" in p.lower() for p in paths)
    missing = not has_test

    tests = TestResult(
        passed=5 if has_test else 0,
        failed=0,
        skipped=0,
        coverage_changed_lines=55 if has_test else 0,
        flaky_history=[],
        asserts_new_behavior=has_test,
    )

    return PRInput(
        id=f"PR-SANDBOX-{stable_id(diff)[:8]}",
        category="sandbox",
        title=title,
        description="User-supplied sandbox diff.",
        diff=diff,
        changed_files=files,
        tests=tests,
        critical_path=critical,
        migration=migration,
        dependency_change=dependency,
        secret_pattern=secret,
        bug_pattern=bool(re.search(r"FIXME|XXX", diff, re.IGNORECASE)),
        missing_tests=missing,
        expected_verdict=FinalVerdict.MERGE,
    )


def load_git_diff(repo_path: str, base: str, head: str) -> Tuple[Optional[str], Optional[str]]:
    if not GIT_AVAILABLE:
        return None, "GitPython is not available in this environment."
    try:
        repo = git.Repo(repo_path)
        diff = repo.git.diff(base, head, unified=3)
        return diff, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(prs: List[PRInput], records: List[DecisionRecord]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tp = fp = fn = tn = 0
    changed = 0
    correct = 0
    rounds_total = 0
    cost_total = 0.0
    rows = []

    for pr, rec in zip(prs, records):
        exp = pr.expected_verdict
        pred = rec.final_verdict
        exp_block = exp == FinalVerdict.BLOCK
        pred_block = pred == FinalVerdict.BLOCK

        if exp_block and pred_block:
            tp += 1
        elif (not exp_block) and pred_block:
            fp += 1
        elif exp_block and (not pred_block):
            fn += 1
        else:
            tn += 1

        if pred == exp:
            correct += 1
        if rec.independent_verdict != pred:
            changed += 1

        rounds_total += len(rec.deliberation_rounds)
        cost_total += rec.cost_total

        rows.append(
            {
                "PR": pr.id,
                "Category": pr.category,
                "Expected": exp.value,
                "Independent baseline": rec.independent_verdict.value,
                "Final": pred.value,
                "Changed by deliberation?": "yes" if rec.independent_verdict != pred else "no",
                "Rounds": len(rec.deliberation_rounds),
                "Cost USD": f"{rec.cost_total:.6f}",
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = correct / len(prs) if prs else 0.0
    deliberation_change_rate = changed / len(prs) if prs else 0.0

    metrics = {
        "count": len(prs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "changed": changed,
        "change_rate": deliberation_change_rate,
        "avg_rounds": rounds_total / len(prs) if prs else 0.0,
        "total_cost": cost_total,
    }
    return metrics, rows


# ---------------------------------------------------------------------------
# State-machine tests
# ---------------------------------------------------------------------------

def run_state_machine_tests() -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    def add(name: str, ok: bool, detail: str = ""):
        results.append({"Test": name, "Status": "PASS" if ok else "FAIL", "Detail": detail})

    prs = generate_benchmark()
    clean = next(p for p in prs if p.id == "PR-CLEAN-1")
    todo_clean = next(p for p in prs if p.id == "PR-CLEAN-5")
    secret = next(p for p in prs if p.id == "PR-SECRET-1")
    bug = next(p for p in prs if p.id == "PR-BUG-1")

    rec_clean, board_clean = run_council(clean, budget_cap=100.0)
    add("Clean PR merges", rec_clean.final_verdict == FinalVerdict.MERGE)
    add(
        "Message board records four independent assessments",
        len([m for m in board_clean.messages if m.kind == "assessment" and m.round == 0]) == 4,
    )

    rec_secret, board_secret = run_council(secret, budget_cap=100.0)
    add("Hardcoded secret blocks", rec_secret.final_verdict == FinalVerdict.BLOCK)
    add("Critical blocker captured", any(f.severity == Severity.CRITICAL for f in rec_secret.blockers))
    add("Secret disagreement triggers deliberation", len(rec_secret.deliberation_rounds) > 0)
    add("Secret deliberation messages are inspectable", len(board_secret.messages) > 8)

    rec_todo, _ = run_council(todo_clean, budget_cap=100.0)
    add("Non-blocking concern can converge to merge", rec_todo.final_verdict == FinalVerdict.MERGE)
    add("Position changes cite reasons", all(pc.reason.strip() for pc in rec_todo.position_changes))

    rec_budget, _ = run_council(secret, budget_cap=0.000001)
    add("Budget cap forces safe BLOCK", rec_budget.final_verdict == FinalVerdict.BLOCK and rec_budget.budget_exceeded)

    rec_bug, _ = run_council(bug, budget_cap=100.0)
    add("Critical subtle bug blocks", rec_bug.final_verdict == FinalVerdict.BLOCK)
    add(
        "Round count never exceeds three",
        all(len(r.deliberation_rounds) <= 3 for r in [rec_clean, rec_secret, rec_todo, rec_bug]),
    )

    raised = False
    try:
        PositionChange(
            agent="CodeReviewer",
            from_verdict=AgentVerdict.APPROVE,
            to_verdict=AgentVerdict.REJECT,
            reason="",
            round=1,
        )
    except ValidationError:
        raised = True
    add("Uncited position change is rejected by schema", raised)

    return results


# ---------------------------------------------------------------------------
# Markdown decision record renderer
# ---------------------------------------------------------------------------

def render_markdown(pr: PRInput, record: DecisionRecord, board: MessageBoard) -> str:
    lines = []
    lines.append(f"# ReleaseCouncil Decision Record — {pr.id}")
    lines.append("")
    lines.append(f"- **Title:** {pr.title}")
    lines.append(f"- **Category:** {pr.category}")
    lines.append(f"- **Created at:** {record.created_at}")
    lines.append(f"- **Final verdict:** `{record.final_verdict.value}`")
    lines.append(f"- **Independent baseline:** `{record.independent_verdict.value}`")
    lines.append(f"- **Rule:** {record.rule}")
    lines.append(f"- **Budget exceeded:** {record.budget_exceeded}")
    lines.append(f"- **Tokens:** {record.token_total}")
    lines.append(f"- **Cost USD:** {record.cost_total:.8f}")
    lines.append("")

    lines.append("## Final conditions")
    if record.conditions:
        for c in record.conditions:
            lines.append(f"- {c}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Critical blockers")
    if record.blockers:
        for b in record.blockers:
            loc = f"{b.file}:{b.line}" if b.file else "unknown"
            lines.append(f"- `{b.severity.value}` **{b.title}** ({b.agent}, {loc}) — {b.detail}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Phase 1 independent assessments")
    for a in record.independent_assessments:
        lines.append(f"### {a.agent}")
        lines.append(f"- Verdict: `{a.verdict.value}`")
        lines.append(f"- Confidence: {a.confidence:.2f}")
        lines.append(f"- Rationale: {a.rationale}")
        if a.findings:
            lines.append("- Findings:")
            for f in a.findings:
                loc = f"{f.file}:{f.line}" if f.file else "n/a"
                lines.append(f"  - `{f.severity.value}` {f.title} ({loc})")
        lines.append("")

    lines.append("## Deliberation rounds")
    if not record.deliberation_rounds:
        lines.append("- No deliberation rounds were required or allowed.")
    for i, round_assessments in enumerate(record.deliberation_rounds, start=1):
        lines.append(f"### Round {i}")
        for a in round_assessments:
            lines.append(f"- **{a.agent}**: `{a.verdict.value}` @ {a.confidence:.2f} — {a.rationale}")
        lines.append("")

    lines.append("## Position changes")
    if not record.position_changes:
        lines.append("- None")
    for pc in record.position_changes:
        lines.append(f"- Round {pc.round}: **{pc.agent}** changed `{pc.from_verdict.value}` -> `{pc.to_verdict.value}`")
        lines.append(f"  - Reason: {pc.reason}")
        if pc.evidence_cited:
            lines.append(f"  - Evidence cited: {', '.join(pc.evidence_cited)}")
    lines.append("")

    lines.append("## Reasoning trail")
    for step in record.reasoning_trail:
        lines.append(f"- {step}")
    lines.append("")

    lines.append("## Inspectable message board excerpt")
    for m in board.messages[-40:]:
        lines.append(f"- `{m.timestamp}` **{m.sender}** · {m.phase.value} · round {m.round} · {m.kind}")
    lines.append("")

    lines.append("```json")
    lines.append(json.dumps(board.snapshot(limit=25), indent=2, default=str)[:12000])
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo PR lookup
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = {
    "Clean refactor": "clean",
    "Subtle payment bug": "bug",
    "Missing tests on checkout": "missing",
    "Hardcoded AWS secret": "secret",
    "Irreversible migration": "migration",
}

DEMO_PR_IDS = {
    "clean": "PR-CLEAN-5",
    "bug": "PR-BUG-1",
    "missing": "PR-MISSING-1",
    "secret": "PR-SECRET-1",
    "migration": "PR-MIGRATION-1",
}


def get_demo_pr(scenario_label: str) -> PRInput:
    key = DEMO_SCENARIOS.get(scenario_label, "secret")
    target_id = DEMO_PR_IDS[key]
    for pr in generate_benchmark():
        if pr.id == target_id:
            return pr
    return generate_benchmark()[0]


def live_scenario_key_for_pr(pr: PRInput) -> str:
    if pr.secret_pattern:
        return "secret"
    if pr.migration:
        return "migration"
    if pr.missing_tests:
        return "missing"
    if pr.bug_pattern:
        return "bug"
    if pr.category == "clean":
        return "clean"
    return "clean"


# ---------------------------------------------------------------------------
# Live in-browser animated demo + WebM recorder
# ---------------------------------------------------------------------------

LIVE_WIDGET_TEMPLATE = r"""
<div class="rc-shell">
  <canvas id="rcCanvas" width="980" height="560"></canvas>
  <div class="rc-controls">
    <button id="rcPlay">Pause</button>
    <select id="rcScenario">
      <option value="clean">Clean refactor</option>
      <option value="bug">Subtle payment bug</option>
      <option value="missing">Missing tests</option>
      <option value="secret">Hardcoded secret</option>
      <option value="migration">Irreversible migration</option>
    </select>
    <label class="rc-speed">Speed
      <input id="rcSpeed" type="range" min="0.5" max="3" step="0.5" value="1" />
    </label>
    <button id="rcRecord">Generate Live Video (.webm)</button>
    <a id="rcDownload" style="display:none">Download video</a>
    <span id="rcStatus">Auto-playing live council demo…</span>
  </div>
</div>

<style>
.rc-shell {
  width: 100%;
  background: linear-gradient(135deg, #0b1220, #111827);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 14px;
  box-sizing: border-box;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}
.rc-shell canvas {
  width: 100%;
  height: auto;
  border-radius: 14px;
  background: #0b1220;
  display: block;
}
.rc-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-top: 12px;
}
.rc-controls button,
.rc-controls select {
  background: #111827;
  color: #e5e7eb;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 10px;
  padding: 8px 12px;
  cursor: pointer;
}
.rc-controls button:hover { border-color: #60a5fa; }
.rc-speed {
  color: #9ca3af;
  display: inline-flex;
  gap: 8px;
  align-items: center;
}
#rcStatus {
  color: #9ca3af;
  font-size: 13px;
}
#rcDownload {
  color: #34d399;
  font-size: 13px;
  text-decoration: underline;
}
</style>

<script>
(function () {
  const initialKey = "__SCENARIO__";
  const scenarios = {
    clean: {
      title: "Clean refactor with resolved TODO",
      expected: "MERGE",
      events: [
        { phase: "Phase 1", sender: "Coordinator", text: "Opening independent assessment.", pulse: false },
        { phase: "Phase 1", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "CONCERN", conf: 0.68, findings: 1, text: "TODO left in utility module.", pulse: true, tokens: 340 },
        { phase: "Phase 1", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "APPROVE", conf: 0.91, findings: 0, text: "92 percent changed-line coverage; behavior asserted.", pulse: true, tokens: 330 },
        { phase: "Phase 1", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "APPROVE", conf: 0.94, findings: 0, text: "No secrets or unsafe patterns.", pulse: true, tokens: 320 },
        { phase: "Phase 1", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "APPROVE", conf: 0.88, findings: 0, text: "Low blast radius; easy rollback.", pulse: true, tokens: 325 },
        { phase: "Phase 2", sender: "Coordinator", text: "Disagreement detected; opening round 1.", pulse: false },
        { phase: "Phase 2", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "APPROVE", conf: 0.83, findings: 1, text: "Revised: TestAnalyst confirmed coverage; TODO demoted to follow-up.", pulse: true, tokens: 390 },
        { phase: "Phase 3", sender: "Coordinator", text: "Final: MERGE. No critical blockers.", pulse: false, final: true }
      ]
    },
    bug: {
      title: "Subtle payment bug",
      expected: "BLOCK",
      events: [
        { phase: "Phase 1", sender: "Coordinator", text: "Opening independent assessment.", pulse: false },
        { phase: "Phase 1", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "REJECT", conf: 0.93, findings: 1, text: "Critical off-by-one refund logic.", pulse: true, tokens: 360 },
        { phase: "Phase 1", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "REJECT", conf: 0.91, findings: 1, text: "Critical path lacks boundary assertions.", pulse: true, tokens: 350 },
        { phase: "Phase 1", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "APPROVE", conf: 0.90, findings: 0, text: "No security issue in isolation.", pulse: true, tokens: 310 },
        { phase: "Phase 1", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "CONCERN", conf: 0.76, findings: 1, text: "Payment path blast radius is high.", pulse: true, tokens: 330 },
        { phase: "Phase 2", sender: "Coordinator", text: "Opening round 1 to reconcile severity.", pulse: false },
        { phase: "Phase 2", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "REJECT", conf: 0.88, findings: 1, text: "Adopted critical correctness risk from CodeReviewer.", pulse: true, tokens: 400 },
        { phase: "Phase 2", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "REJECT", conf: 0.90, findings: 2, text: "Adopted critical refund blocker; rollback unsafe without fix.", pulse: true, tokens: 410 },
        { phase: "Phase 3", sender: "Coordinator", text: "Final: BLOCK. Critical reject stands.", pulse: false, final: true }
      ]
    },
    missing: {
      title: "Missing tests on checkout",
      expected: "BLOCK",
      events: [
        { phase: "Phase 1", sender: "Coordinator", text: "Opening independent assessment.", pulse: false },
        { phase: "Phase 1", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "APPROVE", conf: 0.84, findings: 0, text: "Logic looks simple.", pulse: true, tokens: 320 },
        { phase: "Phase 1", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "REJECT", conf: 0.93, findings: 1, text: "Checkout behavior has no meaningful assertion.", pulse: true, tokens: 360 },
        { phase: "Phase 1", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "APPROVE", conf: 0.91, findings: 0, text: "No direct security flaw.", pulse: true, tokens: 310 },
        { phase: "Phase 1", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "CONCERN", conf: 0.77, findings: 1, text: "Critical path with weak verification.", pulse: true, tokens: 330 },
        { phase: "Phase 2", sender: "Coordinator", text: "Opening round 1.", pulse: false },
        { phase: "Phase 2", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "REJECT", conf: 0.88, findings: 1, text: "Adopted critical test gap from TestAnalyst.", pulse: true, tokens: 390 },
        { phase: "Phase 2", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "REJECT", conf: 0.89, findings: 2, text: "Adopted checkout blocker; release risk unacceptable.", pulse: true, tokens: 400 },
        { phase: "Phase 3", sender: "Coordinator", text: "Final: BLOCK.", pulse: false, final: true }
      ]
    },
    secret: {
      title: "Hardcoded AWS secret",
      expected: "BLOCK",
      events: [
        { phase: "Phase 1", sender: "Coordinator", text: "Opening independent assessment.", pulse: false },
        { phase: "Phase 1", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "APPROVE", conf: 0.86, findings: 0, text: "Config cleanup looks tidy.", pulse: true, tokens: 320 },
        { phase: "Phase 1", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "APPROVE", conf: 0.88, findings: 0, text: "Tests unaffected.", pulse: true, tokens: 310 },
        { phase: "Phase 1", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "REJECT", conf: 0.98, findings: 1, text: "AWS key committed in diff.", pulse: true, tokens: 370 },
        { phase: "Phase 1", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "APPROVE", conf: 0.85, findings: 0, text: "Rollback seems easy.", pulse: true, tokens: 315 },
        { phase: "Phase 2", sender: "Coordinator", text: "Opening round 1; minority critical finding must be examined.", pulse: false },
        { phase: "Phase 2", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "REJECT", conf: 0.90, findings: 1, text: "Adopted secret leakage risk from SecurityAuditor.", pulse: true, tokens: 400 },
        { phase: "Phase 2", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "REJECT", conf: 0.89, findings: 1, text: "No test can justify secret exposure.", pulse: true, tokens: 390 },
        { phase: "Phase 2", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "REJECT", conf: 0.91, findings: 1, text: "Secret rotation makes rollback non-trivial.", pulse: true, tokens: 405 },
        { phase: "Phase 3", sender: "Coordinator", text: "Final: BLOCK. Credential rotation required.", pulse: false, final: true }
      ]
    },
    migration: {
      title: "Irreversible migration",
      expected: "BLOCK",
      events: [
        { phase: "Phase 1", sender: "Coordinator", text: "Opening independent assessment.", pulse: false },
        { phase: "Phase 1", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "CONCERN", conf: 0.74, findings: 1, text: "Schema change lacks compatibility note.", pulse: true, tokens: 335 },
        { phase: "Phase 1", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "CONCERN", conf: 0.76, findings: 1, text: "Migration tests do not cover rollback.", pulse: true, tokens: 340 },
        { phase: "Phase 1", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "APPROVE", conf: 0.89, findings: 0, text: "No direct security issue.", pulse: true, tokens: 310 },
        { phase: "Phase 1", sender: "ReleaseManager", agent: "ReleaseManager", verdict: "REJECT", conf: 0.95, findings: 1, text: "DROP COLUMN without rollback is irreversible.", pulse: true, tokens: 380 },
        { phase: "Phase 2", sender: "Coordinator", text: "Opening round 1.", pulse: false },
        { phase: "Phase 2", sender: "CodeReviewer", agent: "CodeReviewer", verdict: "REJECT", conf: 0.88, findings: 2, text: "Adopted migration blocker from ReleaseManager.", pulse: true, tokens: 400 },
        { phase: "Phase 2", sender: "TestAnalyst", agent: "TestAnalyst", verdict: "REJECT", conf: 0.90, findings: 2, text: "Rollback absence is a critical release test gap.", pulse: true, tokens: 405 },
        { phase: "Phase 2", sender: "SecurityAuditor", agent: "SecurityAuditor", verdict: "REJECT", conf: 0.87, findings: 1, text: "Data loss risk is a security-adjacent integrity issue.", pulse: true, tokens: 395 },
        { phase: "Phase 3", sender: "Coordinator", text: "Final: BLOCK. Use expand-contract migration.", pulse: false, final: true }
      ]
    }
  };

  let key = scenarios[initialKey] ? initialKey : "secret";
  let events = scenarios[key].events.slice();
  let idx = 0;
  let playing = true;
  let speed = 1;
  let last = 0;
  let pulse = null;
  let tokens = 0;
  let cost = 0;
  let log = [];

  const canvas = document.getElementById("rcCanvas");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("rcStatus");
  const playBtn = document.getElementById("rcPlay");
  const scenarioSel = document.getElementById("rcScenario");
  const speedInput = document.getElementById("rcSpeed");
  const recordBtn = document.getElementById("rcRecord");
  const downloadA = document.getElementById("rcDownload");

  const agents = [
    { name: "CodeReviewer", short: "CR", color: "#60a5fa", x: 160, y: 120, verdict: "—", conf: 0, findings: 0 },
    { name: "TestAnalyst", short: "TA", color: "#34d399", x: 160, y: 420, verdict: "—", conf: 0, findings: 0 },
    { name: "SecurityAuditor", short: "SA", color: "#f87171", x: 490, y: 80, verdict: "—", conf: 0, findings: 0 },
    { name: "ReleaseManager", short: "RM", color: "#fbbf24", x: 490, y: 470, verdict: "—", conf: 0, findings: 0 }
  ];
  const coordinator = { x: 490, y: 270 };

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function reset() {
    idx = 0;
    tokens = 0;
    cost = 0;
    pulse = null;
    log = [];
    agents.forEach(a => {
      a.verdict = "—";
      a.conf = 0;
      a.findings = 0;
    });
  }

  function advance() {
    if (idx >= events.length) {
      playing = false;
      playBtn.textContent = "Play";
      setStatus("Demo complete. Click Generate Live Video to record/download a .webm.");
      if (recording) stopRecording();
      return;
    }

    const e = events[idx++];
    if (e.agent) {
      const a = agents.find(x => x.name === e.agent);
      if (a) {
        a.verdict = e.verdict || a.verdict;
        a.conf = typeof e.conf === "number" ? e.conf : a.conf;
        a.findings = typeof e.findings === "number" ? e.findings : a.findings;
      }
    }
    if (e.pulse) pulse = { from: e.sender, progress: 0 };
    const tok = e.tokens || 320;
    tokens += tok;
    cost += (tok / 1000) * 0.00002;
    log.push(e);
    if (log.length > 7) log.shift();
  }

  function rr(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function verdictColor(v) {
    if (v === "APPROVE") return "#22c55e";
    if (v === "CONCERN") return "#f59e0b";
    if (v === "REJECT") return "#ef4444";
    return "#64748b";
  }

  function wrapText(text, maxWidth) {
    const words = String(text).split(" ");
    const lines = [];
    let line = "";
    for (const w of words) {
      const test = line ? line + " " + w : w;
      if (ctx.measureText(test).width > maxWidth && line) {
        lines.push(line);
        line = w;
      } else {
        line = test;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  function drawNode(a) {
    ctx.save();
    ctx.strokeStyle = a.color;
    ctx.lineWidth = 3;
    ctx.fillStyle = "#111827";
    ctx.beginPath();
    ctx.arc(a.x, a.y, 48, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#e5e7eb";
    ctx.font = "bold 16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(a.short, a.x, a.y - 8);

    ctx.fillStyle = verdictColor(a.verdict);
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(a.verdict, a.x, a.y + 12);

    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px sans-serif";
    ctx.fillText(a.conf ? a.conf.toFixed(2) : "0.00", a.x, a.y + 28);
    ctx.fillText("findings: " + a.findings, a.x, a.y + 42);
    ctx.restore();
  }

  function draw() {
    const g = ctx.createLinearGradient(0, 0, 980, 560);
    g.addColorStop(0, "#0b1220");
    g.addColorStop(1, "#111827");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 980, 560);

    ctx.fillStyle = "#e5e7eb";
    ctx.font = "bold 24px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("ReleaseCouncil live demo", 28, 38);

    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px sans-serif";
    ctx.fillText(scenarios[key].title + "  |  expected: " + scenarios[key].expected, 28, 62);

    // progress
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    rr(28, 76, 560, 10, 5);
    ctx.fill();
    ctx.fillStyle = "#60a5fa";
    rr(28, 76, Math.max(4, 560 * (idx / Math.max(1, events.length))), 10, 5);
    ctx.fill();

    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px sans-serif";
    ctx.fillText("tokens: " + tokens + "   cost: $" + cost.toFixed(6), 28, 104);

    // edges
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 2;
    agents.forEach(a => {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(coordinator.x, coordinator.y);
      ctx.stroke();
    });

    // coordinator
    ctx.fillStyle = "#1f2937";
    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(coordinator.x, coordinator.y, 58, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "bold 15px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Coordinator", coordinator.x, coordinator.y - 4);
    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px sans-serif";
    ctx.fillText("protocol engine", coordinator.x, coordinator.y + 16);

    // pulse
    if (pulse) {
      pulse.progress += 0.035 * speed;
      if (pulse.progress >= 1) pulse = null;
      else {
        const fromAgent = agents.find(a => a.name === pulse.from);
        if (fromAgent) {
          const x = fromAgent.x + (coordinator.x - fromAgent.x) * pulse.progress;
          const y = fromAgent.y + (coordinator.y - fromAgent.y) * pulse.progress;
          ctx.fillStyle = fromAgent.color;
          ctx.beginPath();
          ctx.arc(x, y, 7, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.strokeStyle = "rgba(167,139,250," + (1 - pulse.progress) + ")";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(coordinator.x, coordinator.y, 58 + pulse.progress * 60, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }

    agents.forEach(drawNode);

    // log panel
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    rr(635, 80, 315, 390, 14);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#e5e7eb";
    ctx.font = "bold 15px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Shared message board", 655, 108);

    let y = 135;
    ctx.font = "12px sans-serif";
    log.forEach(e => {
      ctx.fillStyle = "#9ca3af";
      ctx.fillText(e.phase + " · " + e.sender, 655, y);
      y += 16;
      ctx.fillStyle = "#e5e7eb";
      const lines = wrapText(e.text, 275);
      lines.forEach(ln => {
        ctx.fillText(ln, 655, y);
        y += 15;
      });
      y += 10;
    });

    if (idx >= events.length) {
      ctx.fillStyle = "rgba(16,185,129,0.15)";
      rr(28, 490, 560, 48, 12);
      ctx.fill();
      ctx.fillStyle = "#34d399";
      ctx.font = "bold 18px sans-serif";
      ctx.fillText("Final council decision: " + scenarios[key].expected, 46, 520);
    }
  }

  function tick(ts) {
    if (!last) last = ts;
    const elapsed = ts - last;
    if (playing && elapsed > 1250 / speed) {
      last = ts;
      advance();
    }
    draw();
    requestAnimationFrame(tick);
  }

  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
    setStatus(playing ? "Simulation running…" : "Simulation paused.");
  });

  scenarioSel.addEventListener("change", () => {
    key = scenarioSel.value;
    events = scenarios[key].events.slice();
    reset();
    playing = true;
    playBtn.textContent = "Pause";
    setStatus("Scenario switched: " + scenarios[key].title);
  });

  speedInput.addEventListener("input", () => {
    speed = parseFloat(speedInput.value);
  });

  // Recording --------------------------------------------------------------
  let recorder = null;
  let chunks = [];
  let recording = false;

  function supportedMime() {
    const types = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm"
    ];
    for (const t of types) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  }

  function startRecording() {
    if (!window.MediaRecorder || !canvas.captureStream) {
      setStatus("Video recording is not supported in this browser.");
      return;
    }
    reset();
    playing = true;
    playBtn.textContent = "Pause";
    chunks = [];
    const stream = canvas.captureStream(30);
    const mime = supportedMime();
    try {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
    } catch (err) {
      setStatus("Recorder error: " + err.message);
      return;
    }
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size) chunks.push(ev.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime || "video/webm" });
      const url = URL.createObjectURL(blob);
      downloadA.href = url;
      downloadA.download = "releasecouncil-" + key + "-" + Date.now() + ".webm";
      downloadA.textContent = "Download recorded video";
      downloadA.style.display = "inline-block";
      downloadA.click();
      setStatus("Video generated. Download started.");
    };
    recorder.start();
    recording = true;
    recordBtn.textContent = "Stop Recording";
    setStatus("Recording live demo… it will stop and download at the end.");
  }

  function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
    recording = false;
    recordBtn.textContent = "Generate Live Video (.webm)";
  }

  recordBtn.addEventListener("click", () => {
    if (recording) stopRecording();
    else startRecording();
  });

  // init
  scenarioSel.value = key;
  reset();
  requestAnimationFrame(tick);
})();
</script>
"""


def render_live_widget(scenario_key: str):
    html = LIVE_WIDGET_TEMPLATE.replace("__SCENARIO__", scenario_key)
    components.html(html, height=700, scrolling=False)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ReleaseCouncil",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session():
    defaults = {
        "mode": "Live Council",
        "demo_scenario": "Hardcoded AWS secret",
        "selected_demo_scenario": "Hardcoded AWS secret",
        "loaded_demo_scenario": None,
        "loaded_settings": None,
        "max_rounds": 3,
        "budget_cap": 5.0,
        "cost_per_1k": 0.02,
        "evidence_source": "Built-in demo",
        "ablation_agent": "CodeReviewer",
        "benchmark_full_metrics": None,
        "benchmark_full_rows": None,
        "benchmark_ablation_metrics": None,
        "benchmark_ablation_rows": None,
        "benchmark_ablation_agent": "CodeReviewer",
        "benchmark_prs": None,
        "benchmark_records_full": None,
        "benchmark_records_ablation": None,
        "test_results": None,
        "demo_pr": None,
        "demo_record": None,
        "demo_board": None,
        "ablation_record": None,
        "ablation_board": None,
        "ablation_agent_used": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    .rc-card {
      background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(31,41,55,0.92));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 14px;
      margin-bottom: 10px;
    }
    .rc-title {font-size: 1.45rem; font-weight: 800;}
    .rc-sub {color:#9ca3af; font-size: 0.95rem;}
    .rc-badge {
      display:inline-block; padding:4px 10px; border-radius:999px;
      font-weight:700; font-size:0.8rem; margin-left:6px;
    }
    .merge {background:rgba(34,197,94,0.16); color:#22c55e; border:1px solid rgba(34,197,94,0.35);}
    .conditions {background:rgba(245,158,11,0.16); color:#f59e0b; border:1px solid rgba(245,158,11,0.35);}
    .block {background:rgba(239,68,68,0.16); color:#ef4444; border:1px solid rgba(239,68,68,0.35);}
    </style>
    """,
    unsafe_allow_html=True,
)


def verdict_badge(v: FinalVerdict) -> str:
    cls = {
        FinalVerdict.MERGE: "merge",
        FinalVerdict.MERGE_WITH_CONDITIONS: "conditions",
        FinalVerdict.BLOCK: "block",
    }[v]
    return f'<span class="rc-badge {cls}">{v.value}</span>'


def run_demo_with_pr(
    pr: PRInput,
    max_rounds: int,
    budget_cap: float,
    cost_per_1k: float,
    loaded_label: Optional[str] = None,
):
    with st.spinner("Running ReleaseCouncil simulation…"):
        record, board = run_council(
            pr,
            max_rounds=max_rounds,
            budget_cap=budget_cap,
            cost_per_1k=cost_per_1k,
        )
    st.session_state["demo_pr"] = pr
    st.session_state["demo_record"] = record
    st.session_state["demo_board"] = board
    st.session_state["loaded_settings"] = (int(max_rounds), float(budget_cap), float(cost_per_1k))
    if loaded_label is not None:
        st.session_state["loaded_demo_scenario"] = loaded_label


# Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ ReleaseCouncil")
    st.caption("Multi-agent PR merge council. No API key required.")

    mode = st.radio(
        "Workspace",
        ["Live Council", "Benchmark", "Ablation", "Decision Record", "Tests"],
        key="mode",
    )

    st.subheader("Protocol controls")
    max_rounds = st.slider(
        "Max deliberation rounds",
        0,
        3,
        int(st.session_state.max_rounds),
        key="max_rounds",
    )
    budget_cap = st.number_input(
        "Budget cap USD per decision",
        value=float(st.session_state.budget_cap),
        step=0.5,
        key="budget_cap",
    )
    cost_per_1k = st.number_input(
        "Cost per 1K tokens USD",
        value=float(st.session_state.cost_per_1k),
        step=0.005,
        format="%.4f",
        key="cost_per_1k",
    )

    st.subheader("Evidence source")
    evidence_source = st.radio(
        "Source",
        ["Built-in demo", "Paste diff", "Local repo (GitPython)"],
        key="evidence_source",
    )

    default_scenario = st.session_state.get("demo_scenario", "Hardcoded AWS secret")
    if default_scenario not in DEMO_SCENARIOS:
        default_scenario = "Hardcoded AWS secret"

    demo_scenario = default_scenario

    if evidence_source == "Built-in demo":
        demo_scenario = st.selectbox(
            "Demo PR",
            list(DEMO_SCENARIOS.keys()),
            index=list(DEMO_SCENARIOS.keys()).index(default_scenario),
            key="demo_scenario",
        )
        st.session_state["selected_demo_scenario"] = demo_scenario

        if st.button("Run selected demo", key="run_demo_btn"):
            run_demo_with_pr(
                get_demo_pr(demo_scenario),
                max_rounds=max_rounds,
                budget_cap=budget_cap,
                cost_per_1k=cost_per_1k,
                loaded_label=demo_scenario,
            )

    elif evidence_source == "Paste diff":
        pasted = st.text_area(
            "Paste unified diff",
            height=180,
            value="+def ping():\n+    return 'pong'\n",
            key="paste_diff",
        )
        if st.button("Analyze pasted diff", key="analyze_pasted_diff"):
            run_demo_with_pr(
                build_pr_from_diff(pasted, title="Pasted sandbox PR"),
                max_rounds=max_rounds,
                budget_cap=budget_cap,
                cost_per_1k=cost_per_1k,
                loaded_label=None,
            )

    else:
        repo_path = st.text_input("Repo path", value=".", key="repo_path")
        base = st.text_input("Base ref", value="HEAD~1", key="base_ref")
        head = st.text_input("Head ref", value="HEAD", key="head_ref")
        if st.button("Load diff from repo", key="load_repo_diff"):
            if not GIT_AVAILABLE:
                st.error("GitPython is not available.")
            else:
                diff, err = load_git_diff(repo_path, base, head)
                if err:
                    st.error(err)
                else:
                    run_demo_with_pr(
                        build_pr_from_diff(diff, title=f"Local repo {base}..{head}"),
                        max_rounds=max_rounds,
                        budget_cap=budget_cap,
                        cost_per_1k=cost_per_1k,
                        loaded_label=None,
                    )

    st.divider()
    st.info(
        "The live widget below auto-plays and can generate a downloadable `.webm` "
        "video directly in the browser."
    )


# Auto-run initial demo ------------------------------------------------------

settings_sig = (int(max_rounds), float(budget_cap), float(cost_per_1k))

if evidence_source == "Built-in demo":
    if (
        st.session_state.get("loaded_demo_scenario") != demo_scenario
        or st.session_state.get("loaded_settings") != settings_sig
        or st.session_state.get("demo_record") is None
    ):
        run_demo_with_pr(
            get_demo_pr(demo_scenario),
            max_rounds=max_rounds,
            budget_cap=budget_cap,
            cost_per_1k=cost_per_1k,
            loaded_label=demo_scenario,
        )
else:
    if st.session_state.get("demo_record") is None:
        run_demo_with_pr(
            get_demo_pr("Hardcoded AWS secret"),
            max_rounds=max_rounds,
            budget_cap=budget_cap,
            cost_per_1k=cost_per_1k,
            loaded_label="Hardcoded AWS secret",
        )


# Main ----------------------------------------------------------------------

st.markdown('<div class="rc-title">ReleaseCouncil</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rc-sub">Four specialized agents independently assess a pull request, deliberate through a shared message board, and a coordinator issues a merge/block decision.</div>',
    unsafe_allow_html=True,
)
st.write("")

if mode == "Live Council":
    pr: PRInput = st.session_state["demo_pr"]
    record: DecisionRecord = st.session_state["demo_record"]
    board: MessageBoard = st.session_state["demo_board"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final verdict", record.final_verdict.value)
    c2.metric("Rounds used", len(record.deliberation_rounds))
    c3.metric("Tokens", f"{record.token_total:,}")
    c4.metric("Cost USD", f"{record.cost_total:.6f}")

    st.write("")
    scenario_key = live_scenario_key_for_pr(pr)

    st.subheader("Live animated council demo")
    render_live_widget(scenario_key)

    st.divider()

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader("Pull request")
        st.markdown(f"**{pr.id}** — {pr.title}")
        st.caption(f"Category: `{pr.category}` | Author: `{pr.author}` | Expected benchmark label: `{pr.expected_verdict.value}`")
        st.markdown(
            f"""
            <div class="rc-card">
            <b>Description</b><br/>
            {pr.description or 'No description supplied.'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View diff", expanded=False):
            st.code(pr.diff or "(empty diff)", language="diff")
        with st.expander("Changed files and test summary", expanded=True):
            st.json(
                {
                    "changed_files": [f.model_dump(mode="json") for f in pr.changed_files],
                    "tests": pr.tests.model_dump(mode="json"),
                    "flags": {
                        "critical_path": pr.critical_path,
                        "migration": pr.migration,
                        "dependency_change": pr.dependency_change,
                        "secret_pattern": pr.secret_pattern,
                        "bug_pattern": pr.bug_pattern,
                        "missing_tests": pr.missing_tests,
                    },
                }
            )

    with right:
        st.subheader("Final council positions")
        for a in record.final_assessments:
            badge = {
                AgentVerdict.APPROVE: "merge",
                AgentVerdict.CONCERN: "conditions",
                AgentVerdict.REJECT: "block",
            }[a.verdict]
            findings_html = ""
            if a.findings:
                items = "".join(
                    f"<li><code>{f.severity.value}</code> {f.title} — {f.file or 'n/a'}:{f.line or 'n/a'}</li>"
                    for f in a.findings
                )
                findings_html = f"<ul>{items}</ul>"
            st.markdown(
                f"""
                <div class="rc-card">
                  <b>{a.agent}</b>
                  <span class="rc-badge {badge}">{a.verdict.value}</span>
                  <div class="rc-sub">confidence {a.confidence:.2f} · round {a.round}</div>
                  <div>{a.rationale}</div>
                  {findings_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Decision")
    st.markdown(f"### {record.final_verdict.value} {verdict_badge(record.final_verdict)}", unsafe_allow_html=True)
    st.markdown(f"**Rule:** {record.rule}")
    if record.conditions:
        st.markdown("**Conditions:**")
        for c in record.conditions:
            st.markdown(f"- {c}")
    else:
        st.markdown("**Conditions:** none")

    md = render_markdown(pr, record, board)
    st.download_button(
        "Download decision record (Markdown)",
        md,
        file_name=f"{pr.id}-decision.md",
        mime="text/markdown",
        key="download_decision_live",
    )

    with st.expander("Inspectable message board", expanded=False):
        for m in board.messages:
            st.markdown(f"**{m.sender}** · `{m.phase.value}` · round `{m.round}` · `{m.kind}`")
            st.json(m.content, expanded=False)

    with st.expander("Reasoning trail", expanded=True):
        for step in record.reasoning_trail:
            st.markdown(f"- {step}")

elif mode == "Benchmark":
    st.subheader("30-PR synthetic benchmark")
    st.caption("Covers clean changes, subtle bugs, missing tests, hardcoded secrets, and migration risks.")

    abl_agent = st.selectbox(
        "Ablation agent",
        DEFAULT_AGENT_NAMES,
        index=DEFAULT_AGENT_NAMES.index(st.session_state.get("ablation_agent", "CodeReviewer")),
        key="ablation_agent",
    )

    if st.button("Run full benchmark + ablation", type="primary", key="run_benchmark"):
        prs = generate_benchmark()
        with st.spinner("Running full four-agent council…"):
            full_pairs = [
                run_council(
                    p,
                    max_rounds=max_rounds,
                    budget_cap=budget_cap,
                    cost_per_1k=cost_per_1k,
                )
                for p in prs
            ]
        with st.spinner("Running single-agent ablation…"):
            abl_pairs = [
                run_council(
                    p,
                    agent_names=[abl_agent],
                    max_rounds=max_rounds,
                    budget_cap=budget_cap,
                    cost_per_1k=cost_per_1k,
                )
                for p in prs
            ]

        full_records = [r for r, _ in full_pairs]
        abl_records = [r for r, _ in abl_pairs]
        full_metrics, full_rows = compute_metrics(prs, full_records)
        abl_metrics, abl_rows = compute_metrics(prs, abl_records)

        st.session_state["benchmark_prs"] = prs
        st.session_state["benchmark_records_full"] = full_records
        st.session_state["benchmark_records_ablation"] = abl_records
        st.session_state["benchmark_full_metrics"] = full_metrics
        st.session_state["benchmark_full_rows"] = full_rows
        st.session_state["benchmark_ablation_metrics"] = abl_metrics
        st.session_state["benchmark_ablation_rows"] = abl_rows
        st.session_state["benchmark_ablation_agent"] = abl_agent

    if st.session_state.get("benchmark_full_metrics"):
        fm = st.session_state["benchmark_full_metrics"]
        am = st.session_state["benchmark_ablation_metrics"]
        used_abl_agent = st.session_state.get("benchmark_ablation_agent", abl_agent)

        st.markdown("### Full council metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Blocking precision", f"{fm['precision'] * 100:.1f}%")
        m2.metric("Blocking recall", f"{fm['recall'] * 100:.1f}%")
        m3.metric("Blocking F1", f"{fm['f1'] * 100:.1f}%")
        m4.metric("Overall accuracy", f"{fm['accuracy'] * 100:.1f}%")
        m5.metric("Deliberation changed outcome", f"{fm['changed']}/{fm['count']} ({fm['change_rate'] * 100:.0f}%)")
        m6.metric("Avg rounds", f"{fm['avg_rounds']:.2f}")

        st.markdown(f"### Ablation: {used_abl_agent} only")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Blocking precision", f"{am['precision'] * 100:.1f}%")
        a2.metric("Blocking recall", f"{am['recall'] * 100:.1f}%")
        a3.metric("Blocking F1", f"{am['f1'] * 100:.1f}%")
        a4.metric("Overall accuracy", f"{am['accuracy'] * 100:.1f}%")
        a5.metric("Deliberation changed outcome", f"{am['changed']}/{am['count']}")

        st.caption(
            "The ablation runs the same coordinator protocol but with only one specialist. "
            "This shows whether multi-agent perspective-taking adds value."
        )

        st.subheader("Per-PR results")
        tab_full, tab_abl = st.tabs(["Full council", "Ablation"])
        with tab_full:
            st.dataframe(st.session_state["benchmark_full_rows"], use_container_width=True)
        with tab_abl:
            st.dataframe(st.session_state["benchmark_ablation_rows"], use_container_width=True)
    else:
        st.info("Run the benchmark to populate precision/recall and deliberation-change statistics.")

elif mode == "Ablation":
    st.subheader("Single-agent ablation")
    st.caption("Choose one specialist and see how the council protocol behaves without the other perspectives.")

    abl_agent = st.selectbox(
        "Ablation agent",
        DEFAULT_AGENT_NAMES,
        index=DEFAULT_AGENT_NAMES.index(st.session_state.get("ablation_agent", "CodeReviewer")),
        key="ablation_agent",
    )

    if st.session_state.get("demo_pr") is not None:
        pr = st.session_state["demo_pr"]
        if st.button("Run ablation on current demo PR", key="run_ablation"):
            with st.spinner("Running ablation…"):
                rec, brd = run_council(
                    pr,
                    agent_names=[abl_agent],
                    max_rounds=max_rounds,
                    budget_cap=budget_cap,
                    cost_per_1k=cost_per_1k,
                )
            st.session_state["ablation_record"] = rec
            st.session_state["ablation_board"] = brd
            st.session_state["ablation_agent_used"] = abl_agent

        if st.session_state.get("ablation_record") is not None:
            rec = st.session_state["ablation_record"]
            full = st.session_state["demo_record"]
            used_agent = st.session_state.get("ablation_agent_used", abl_agent)
            c1, c2 = st.columns(2)
            c1.metric("Full council verdict", full.final_verdict.value)
            c2.metric(f"{used_agent}-only verdict", rec.final_verdict.value)
            st.markdown("### Ablation reasoning trail")
            for s in rec.reasoning_trail:
                st.markdown(f"- {s}")
            st.download_button(
                "Download ablation decision record",
                render_markdown(pr, rec, st.session_state["ablation_board"]),
                file_name=f"{pr.id}-ablation-{used_agent}.md",
                mime="text/markdown",
                key="download_ablation_record",
            )
    else:
        st.warning("Load a demo PR first from the Live Council tab.")

elif mode == "Decision Record":
    st.subheader("Decision record")
    if st.session_state.get("demo_pr") is not None:
        pr = st.session_state["demo_pr"]
        record = st.session_state["demo_record"]
        board = st.session_state["demo_board"]
        md = render_markdown(pr, record, board)
        st.markdown(md)
        st.download_button(
            "Download Markdown",
            md,
            file_name=f"{pr.id}-decision.md",
            mime="text/markdown",
            key="download_decision_record_page",
        )
    else:
        st.warning("No decision record yet.")

elif mode == "Tests":
    st.subheader("Coordinator state-machine tests")
    st.caption("These tests validate the protocol before relying on agent behavior.")
    if st.button("Run tests", type="primary", key="run_tests"):
        with st.spinner("Running tests…"):
            st.session_state["test_results"] = run_state_machine_tests()

    if st.session_state.get("test_results"):
        passed = sum(1 for r in st.session_state["test_results"] if r["Status"] == "PASS")
        total = len(st.session_state["test_results"])
        st.metric("Passed", f"{passed}/{total}")
        st.dataframe(st.session_state["test_results"], use_container_width=True)
    else:
        st.info("Run the tests to validate the state machine.")