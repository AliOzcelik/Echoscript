from pathlib import Path
from hailo_platform.genai import LLM




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



llm_path = Path.home() / "hailo-rpi5-examples/llm_models/Qwen2.5-1.5B-Instruct.hef"

# tokens taken by the system prompt, the instruction and the chat template, kept conservative
prompt_overhead_tokens = 300


class Summarize:

    # vdevice is shared with Whisper and created once in main.py
    def __init__(self, vdevice, model_path=llm_path, max_tokens=600, temperature=0.3):
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.llm = LLM(vdevice, str(model_path))
        except Exception as exc:
            raise SummarizationError(f"Could not load model '{model_path}': {exc}")

        # the context size is fixed inside the HEF
        # transcript text gets what is left after the reply and the prompt overhead
        self.max_chunk_tokens = self.llm.max_context_capacity() - max_tokens - prompt_overhead_tokens


    def chunk(self, text):
        lines = text.split("\n")
        chunks = []
        current = []
        size = 0

        for line in lines:
            line_tokens = len(self.llm.tokenize(line))

            # if adding this line would push the current chunk over the limit
            if current and size + line_tokens > self.max_chunk_tokens:
                chunks.append("\n".join(current))
                current = []
                size = 0

            current.append(line)
            size += line_tokens

        # add the last partial chunk that didn't hit the limit
        if current:
            chunks.append("\n".join(current))

        return chunks


    def ask(self, instruction, text):
        # Hailo keeps the conversation between calls, so reset it
        # otherwise the chunks and partial summaries pile up and overflow the context
        self.llm.clear_context()

        try:
            response = self.llm.generate_all(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\n\n{text}"}
                ],
                max_generated_tokens = self.max_tokens,
                temperature = self.temperature
            )

        except Exception as exc:
            raise SummarizationError(f"LLM generation failed: {exc}") from exc

        return response.strip()


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


    # must be called before the VDevice closes
    def close(self):
        self.llm.release()
