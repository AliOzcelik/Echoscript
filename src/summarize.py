from pathlib import Path
from llama_cpp import Llama




class SummarizationError(Exception):
    """Raised when the local LLM can't load or can't generate"""



SYSTEM_PROMPT = (
    "You are a careful meeting-notes assistant. Summarize transcripts faithfully. "
    "Always reply in the SAME language as the transcript. Do not invent facts."
)

# What we ask for on a single (short enough) transcript.
SUMMARY_INSTRUCTION = (
    "Summarize the meeting transcript below. Produce:\n"
    "1. A 2-3 sentence overview.\n"
    "2. Key points (bullets).\n"
    "3. Decisions made.\n"
    "4. Action items (who, what)."
)

# What we ask when combining several partial summaries into one.
COMBINE_INSTRUCTION = (
    "Below are partial summaries of consecutive parts of ONE meeting. "
    "Merge them into a single coherent summary with the same four sections "
    "(overview, key points, decisions, action items). Remove duplicates."
)



project_root = Path(__file__).resolve().parent.parent
llm_path = f"{project_root}/models/Qwen3.5-2B-Q5_K_M.gguf"


# Raspberry pi 5 has 4 threads
# max_chars_per_chunk is how much transcript fits in one pass
class Summarize:

    def __init__(self, model_path=llm_path, n_ctx=4096, n_threads=4, max_tokens=600, temperature=0.3, max_chars_per_chunk=8000):
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.max_chars_per_chunk = max_chars_per_chunk

        try:
            self.llm = Llama(model_path = model_path, n_ctx = n_ctx, n_threads = n_threads, verbose = False)
        except Exception as exc:
            raise SummarizationError(f"Could not load model '{model_path}': {exc}")


    def chunk(self, text):
        lines = text.split("\n")
        chunks = []
        current = []
        size = 0

        for line in lines:
            new_size = size + len(line)

            # if adding this line would push the current chunk over the limit
            if current and new_size > self.max_chars_per_chunk:
                chunks.append("\n".join(current))
                current = []
                size = 0

            current.append(line)
            size += len(line) + 1

        # add the last partial chunk that didn't hit the limit
        if current:
            chunks.append("\n".join(current))

        return chunks


    def ask(self, instruction, text):

        try:
            response = self.llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\n\n{text}"}
                ],
                max_tokens = self.max_tokens,
                temperature = self.temperature
            )

        except Exception as exc:
            raise SummarizationError(f"LLM generation failed: {exc}") from exc

        return response["choices"][0]["message"]["content"].strip()


    def summarize(self, transcription_text):
        if not transcription_text.strip():
            return "(empty transcript - nothing to summarize)"

        chunks = self.chunk(transcription_text)

        if len(chunks) == 1:
            response = self.ask(SUMMARY_INSTRUCTION, chunks[0])
        else:
            partials = [self.ask(SUMMARY_INSTRUCTION, c) for c in chunks]
            response = self.ask(COMBINE_INSTRUCTION, "\n\n".join(partials))

        return response
