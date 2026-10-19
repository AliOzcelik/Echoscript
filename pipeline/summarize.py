"""Summarize: transcript -> summary with a local LLM (stage [6] of PROJECT_PLAN.md).

Two backends, picked with [summary] backend in config.toml:
    "hailo"  Qwen3-1.7B on the AI HAT+2 (Hailo-10H): the CPU stays free. 2048-token context.
    "llama"  llama-server (llama.cpp) on the CPU, e.g. Qwen3.5-2B GGUF. Larger context, uses the CPU.

A transcript longer than the context is summarized in parts, and the parts are merged (map-reduce).
"""
import re

import httpx

from pipeline.config import SummaryConfig


# Above 1.0 it discourages repeating tokens. Without it the small models loop
# ("- Bu kişi, ... zorunda." 30 times); values below 1.0 make that worse.
REPEAT_PENALTY = 1.1
CHAT_OVERHEAD = 64      # tokens the chat template adds around the messages (role tags), kept generous


# Prompts. Tested on Turkish interview recordings with Qwen2.5-1.5B and Qwen3-1.7B (Echoscript/src/summarize.py):
# - asked in English, the small models answer in English even for a Turkish transcript: every language
#   gets prompts in its own language
# - a numbered list of sections made the model invent decisions and action items for an interview,
#   so every section is a heading to fill and the empty ones get "Yok" / "None"
# - the transcript goes before the instructions: the model follows what it read last
# - a longer list of rules, or a shorter prompt, made the model summarize the prompt or copy the transcript back

TR_TEMPLATE = (
    "## Konu\n"
    "Konuşmanın ne hakkında olduğunu 2-3 cümleyle anlat.\n\n"
    "## Önemli noktalar\n"
    "Konuşmada söylenen en önemli 3-5 bilgiyi madde madde yaz. Her madde farklı bir bilgi olsun.\n\n"
    "## Kararlar\n"
    "Açıkça alınan kararları yaz. Karar alınmadıysa sadece \"Yok\" yaz.\n\n"
    "## Yapılacaklar\n"
    "Kimin ne yapacağını yaz. Görev verilmediyse sadece \"Yok\" yaz."
)

EN_TEMPLATE = (
    "## Topic\n"
    "Describe in 2-3 sentences what the conversation is about.\n\n"
    "## Key points\n"
    "List the 3-5 most important things that were said, one per bullet. Each bullet is a different fact.\n\n"
    "## Decisions\n"
    "Write the decisions that were clearly made. If none were made, write only \"None\".\n\n"
    "## Action items\n"
    "Write who will do what. If no tasks were given, write only \"None\"."
)

# language -> (system prompt, summary instruction, combine instruction, word for an empty section)
PROMPTS = {
    "tr": (
        "Sen konuşma dökümlerini özetleyen bir asistansın. Yalnızca Türkçe yazarsın. "
        "Sadece dökümde geçen bilgileri kullanırsın, hiçbir şey uydurmazsın. Aynı bilgiyi iki kez yazmazsın.",

        "Aşağıda bir konuşmanın otomatik ses tanıma ile çıkarılmış dökümü var. Dökümde yazım hataları ve "
        "anlamsız kelimeler olabilir; anlamı belirsiz kısımları atla.\n\nDÖKÜM:\n{text}\n\n"
        "Bu dökümü aşağıdaki dört başlıkla özetle. Başlıkları aynen kullan, açıklamaları yazma.\n\n" + TR_TEMPLATE,

        "Aşağıda TEK bir konuşmanın ardışık bölümlerine ait özetler var.\n\nÖZETLER:\n{text}\n\n"
        "Bunları tek bir özette birleştir. Aynı bilgiyi bir kez yaz. "
        "Aşağıdaki dört başlığı aynen kullan, açıklamaları yazma.\n\n" + TR_TEMPLATE,

        "Yok",
    ),
    "en": (
        "You summarize conversation transcripts. You use only information from the transcript "
        "and never invent anything. You never write the same fact twice.",

        "Below is a conversation transcript made by automatic speech recognition. It may contain "
        "misspelled or meaningless words; skip the parts whose meaning is unclear.\n\nTRANSCRIPT:\n{text}\n\n"
        "Summarize this transcript with the four headings below. Use the headings exactly, "
        "do not write the descriptions.\n\n" + EN_TEMPLATE,

        "Below are summaries of consecutive parts of ONE conversation.\n\nSUMMARIES:\n{text}\n\n"
        "Merge them into a single summary. Write each fact once. "
        "Use the four headings below exactly, do not write the descriptions.\n\n" + EN_TEMPLATE,

        "None",
    ),
}



def summarize(lines, n_words, config, language):
    """lines: transcript lines ("[0:01:23] SPK_0: ..."); n_words: spoken words, without the prefixes."""
    if n_words < config.min_words:
        return "(transcript too short to summarize)"

    prompts = PROMPTS.get(language, PROMPTS["en"])
    llm = HailoLLM(config.hailo_model) if config.backend == "hailo" else LlamaServer(config.llama_url, config.timeout_s)
    try:
        return map_reduce(llm, lines, prompts, config)
    finally:
        llm.close()



# Backends: both have context(), tokens(text), chat(system, user, max_tokens, temperature), close()

class HailoLLM:
    """A Qwen .hef on the Hailo-10H. The context size is fixed inside the HEF (2048 tokens)."""

    def __init__(self, model_path):
        # imported here: the llama backend must work without HailoRT installed
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path}: Hailo model not found")
        try:
            self.vdevice = VDevice()
            self.llm = LLM(self.vdevice, str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Hailo LLM could not load {model_path.name}: {exc}. Is the AI HAT+2 "
                               f"detected (/dev/h1x-0, `hailortcli scan`)? Or set [summary] backend = \"llama\"") from exc

        # Qwen3 first writes its reasoning inside <think>...</think>; /no_think in the system prompt turns it off
        self.no_think = " /no_think" if "Qwen3" in model_path.name else ""

    def context(self):
        return self.llm.max_context_capacity()

    def tokens(self, text):
        return len(self.llm.tokenize(text))

    def chat(self, system, user, max_tokens, temperature):
        # Hailo keeps the conversation between calls: without a reset the parts pile up and overflow the context
        self.llm.clear_context()
        return self.llm.generate_all(
            [{"role": "system", "content": system + self.no_think}, {"role": "user", "content": user}],
            max_generated_tokens = max_tokens,
            temperature = temperature,
            frequency_penalty = REPEAT_PENALTY,     # HailoRT's name for a repeat penalty (> 1.0 = less repetition)
        )

    def close(self):
        self.llm.release()          # the model must be released before the VDevice closes
        self.vdevice.release()


class LlamaServer:
    """llama-server (llama.cpp) over HTTP, like whisper-server in stt.py. Context = the server's -c."""

    def __init__(self, url, timeout_s):
        self.client = httpx.Client(base_url = url, timeout = timeout_s)
        response = self.client.get("/props")
        response.raise_for_status()
        self.n_ctx = response.json()["default_generation_settings"]["n_ctx"]   # per slot

    def context(self):
        return self.n_ctx

    def tokens(self, text):
        response = self.client.post("/tokenize", json = {"content": text})
        response.raise_for_status()
        return len(response.json()["tokens"])

    def chat(self, system, user, max_tokens, temperature):
        response = self.client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat_penalty": REPEAT_PENALTY,
                "chat_template_kwargs": {"enable_thinking": False},    # Qwen3 / 3.5: no <think> block
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def close(self):
        self.client.close()



# Map-reduce

def map_reduce(llm, lines, prompts, config):
    system, summary_instruction, combine_instruction, none_word = prompts

    # the transcript gets what the reply, the prompt and the chat template leave of the context
    prompt_tokens = llm.tokens(system) + max(llm.tokens(summary_instruction), llm.tokens(combine_instruction))
    budget = llm.context() - config.max_tokens - prompt_tokens - CHAT_OVERHEAD
    if budget < 200:
        raise RuntimeError(f"LLM context {llm.context()} is too small for max_tokens = {config.max_tokens}")

    def ask(instruction, text, max_tokens):
        reply = llm.chat(system, instruction.replace("{text}", text), max_tokens, config.temperature)
        return clean(reply, instruction, none_word)

    parts = pack(llm, lines, "\n", budget)
    if len(parts) == 1:
        return ask(summary_instruction, parts[0], config.max_tokens)

    # A long meeting gives more partial summaries than one prompt can hold, so they are merged in
    # rounds, as many per call as fit, until one is left. Partial summaries get half the reply budget,
    # so at least two always fit together and every round shrinks the list.
    half = config.max_tokens // 2
    partials = []
    for i, part in enumerate(parts):
        print(f"  part {i + 1}/{len(parts)}", flush=True)
        partials.append(ask(summary_instruction, part, half))

    while True:
        groups = pack(llm, partials, "\n\n", budget)
        if len(groups) == 1:
            print("  combining", flush=True)
            return ask(combine_instruction, groups[0], config.max_tokens)
        if len(groups) == len(partials):
            raise RuntimeError("partial summaries are too long to merge within the LLM context")

        print(f"  combining {len(partials)} partial summaries into {len(groups)}", flush=True)
        partials = [ask(combine_instruction, g, half) for g in groups]


def pack(llm, items, separator, budget):
    """Items -> as few groups as fit in the budget each, keeping their order."""
    groups = []
    current = []
    size = 0

    for item in items:
        for piece, n in fit(llm, item, budget):
            if current and (size + n) > budget:
                groups.append(separator.join(current))      # this piece would overflow the group
                current = []
                size = 0

            current.append(piece)
            size += n

    if current:
        groups.append(separator.join(current))

    return groups


def fit(llm, item, budget):
    """An item longer than the budget (a long monologue) is cut in halves at word boundaries."""
    n = llm.tokens(item)
    words = item.split()
    if n <= budget or len(words) < 2:
        return [(item, n)]

    half = len(words) // 2
    return fit(llm, " ".join(words[:half]), budget) + fit(llm, " ".join(words[half:]), budget)


def clean(reply, instruction, none_word):
    """Remove what the models add around the answer, and instruction lines they copy into it."""
    reply = reply.replace("<|im_end|>", "")                         # Hailo leaves the end-of-turn token
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)    # Qwen3: an empty think block

    # The model sometimes copies an instruction line of the template into its answer
    # ('Kimin ne yapacağını yaz. Görev verilmediyse sadece "Yok" yaz.'). Drop those lines, headings stay.
    instruction_lines = {line.strip() for line in instruction.split("\n") if line.strip() and not line.startswith("#")}
    lines = [line for line in reply.split("\n") if line.strip().lstrip("-* ").strip() not in instruction_lines]

    # A section left empty by that means there was nothing for it: it gets "Yok" / "None".
    # The extra "#" closes the last section, it is not part of the result.
    result = []
    for line in lines + ["#"]:
        if line.startswith("#"):
            written = [l for l in result if l.strip()]
            if written and written[-1].startswith("#"):
                result.append(none_word)
        result.append(line)

    return "\n".join(result[:-1]).strip()
