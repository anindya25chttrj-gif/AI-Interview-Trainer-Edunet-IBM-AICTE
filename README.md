# AI-Powered Interview Trainer (Native watsonx.ai RAG Pipeline)
# AI Interview Trainer

🚀 **Live Deployment URL:** [Go To Website](https://ai-interview-trainer-edunet-ibm-aic.vercel.app)

An enterprise-grade, high-contrast, minimalist AI Interview Trainer built utilizing a localized **Retrieval-Augmented Generation (RAG)** architecture. The system ingests foundational career frameworks, stores structural text representations inside a multidimensional matrix vector space, and utilizes **Meta-Llama-3.3-70B-Instruct** hosted on **IBM watsonx.ai** to simulate precise technical, behavioral, and resume-grounded mock interviews.

---

## 🛠️ Core System Architecture

The application bypasses heavy enterprise orchestration overhead by implementing a native, low-latency execution flow across two distinct processing layers:

1. **Local Semantic Embedding Engine (`rag_engine.py`)**: Localizes the vectorization pass by caching the neural weights of the `all-MiniLM-L6-v2` model directly on the host machine. It processes knowledge sources using a sliding window chunking mechanism ($500$ words per block, $20\%$ overlap) and executes deterministic matrix similarity checking via local NumPy dot-product calculations.
2. **In-Memory Document Ingestion (`app.py`)**: Features an advanced multipart file stream pipeline. When a user uploads a raw `.pdf` resume from the frontend, the system intercepts the binary data directly within a RAM bytes buffer (`io.BytesIO`), processes its layout structure on the fly using a fast text extractor, and passes it directly into the LLM system window context without ever executing heavy, permanent disk writes.

---

## ⚙️ Deterministic Hyperparameter Tuning

To fulfill academic compliance and eliminate generic AI hallucinations, the inference parameters are strictly constrained:

| Hyperparameter | Configured Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Model ID** | `meta-llama/llama-3-3-70b-instruct` | State-of-the-art 70-billion parameter instruction model for advanced logic. |
| **Temperature** | `0.3` | Low variance threshold; forces deterministic, fact-anchored response generation. |
| **Top-P** | `0.9` | Restricts nucleus sampling to the top 90% cumulative probability mass. |
| **Max New Tokens**| `800` | Limits generation window headroom to optimize latency and manage API resource usage. |

### 🛑 Domain Guardrails & Safety Compliance
The system instruction matrix implements an explicit **Domain Intercept Layer**. If a user submits queries outside the professional scope of career coaching, computer science competencies, or interview simulation (e.g., general science, history, pop culture), the model catches the violation and drops the execution thread with a standardized, polite domain exception message.

---
Matrix & Vector DB Setup
├── requirements.txt        # Frozen Application Dependencies
└── .env                    # Local Environment Configuration Variables (Ignored on Git)
