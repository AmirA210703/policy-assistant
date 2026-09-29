from dataclasses import asdict, dataclass, field


@dataclass
class Result:
    method: str                      # "rules" | "llm_full" | "llm_rag"
    question: str
    answer: str
    policy_title: str | None         # the policy the method says is relevant
    policy_text: str | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens_est: int = 0    # RAG only: estimated tokens embedded for the query
    covered: str = "yes"             # model's own claim: yes | partial | no
    grounding: str = "grounded"      # automatic check: grounded | abstained | unsupported
    grounding_reason: str = ""
    retrieved: list[str] = field(default_factory=list)  # candidates considered (rules/RAG)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
