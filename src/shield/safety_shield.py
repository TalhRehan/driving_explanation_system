"""
Safety shield: hard constraint that prevents explanations during
safety-critical moments. Also manages the deferred explanation buffer.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeferredEvent:
    timestamp: float
    action: str
    evidence: list
    context: dict


class SafetyShield:
    """
    Evaluates SafeBlock(x_t) and gates the timing policy's decision.
    Optionally maintains a bounded buffer for deferred explanations.
    """

    def __init__(self, cfg: dict, buffer_capacity: int = 10):
        s = cfg["shield"]
        self.tau_ttc = s["tau_ttc"]
        self.tau_a = s["tau_a"]
        self.tau_omega = s["tau_omega"]
        self.ttc_inf = s["ttc_inf"]
        self._buffer: deque[DeferredEvent] = deque(maxlen=buffer_capacity)

    def is_blocked(
        self,
        ttc: float,
        accel: float,
        omega: float,
        crit_zone: int,
    ) -> bool:
        """Return True when displaying an explanation would be unsafe."""
        return (
            ttc < self.tau_ttc
            or abs(accel) > self.tau_a
            or abs(omega) > self.tau_omega
            or bool(crit_zone)
        )

    def gate(
        self,
        policy_decision: int,
        ttc: float,
        accel: float,
        omega: float,
        crit_zone: int,
        timestamp: float,
        action: str,
        evidence: Optional[list] = None,
        context: Optional[dict] = None,
    ) -> int:
        """
        Apply the shield to the policy decision.

        Returns delta_exec_t (0 or 1).
        If policy wanted to explain but shield blocks, the event is
        buffered for possible deferred delivery.
        """
        blocked = self.is_blocked(ttc, accel, omega, crit_zone)

        if blocked:
            if policy_decision == 1:
                self._buffer.append(
                    DeferredEvent(
                        timestamp=timestamp,
                        action=action,
                        evidence=evidence or [],
                        context=context or {},
                    )
                )
            return 0

        return policy_decision

    def pop_deferred(self, now: float, rho: float = 0.05) -> Optional[DeferredEvent]:
        """
        Pop the most recent buffered event when the shield is clear.
        Returns None if the buffer is empty.
        """
        if not self._buffer:
            return None
        event = self._buffer.pop()
        delay = now - event.timestamp
        # Staleness is tracked externally via reward; just return the event.
        return event

    def buffer_size(self) -> int:
        return len(self._buffer)