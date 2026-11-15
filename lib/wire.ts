// The client wires and upstream wires: OpenAI Chat Completions, Anthropic Messages, and Google Gemini.
//
// Translates between client wire formats and upstream provider wires deterministically.
// Supports bidirectional streaming and non-streaming responses, tools, vision, and JSON mode.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return content === undefined ? '' : JSON.stringify(content);
  const text = textOf(content);
  return text !== '' ? text : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: { type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof fn.name === 'string' ? fn.name : '' },
            })
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (typeof choice.finish_reason === 'string') pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';
      for (const frame of frames) yield* eventsOfFrame(frame);
    }
    if (buffer.trim() !== '') yield* eventsOfFrame(buffer);
    if (!finished) yield* startEvents(), yield* finishEvents(pendingStop ?? 'end_turn');
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Google Gemini Wire Translation
// ---------------------------------------------------------------------------

/** Convert OpenAI Chat Completions body to Google Gemini generateContent payload. */
export function openAIToGemini(body: Json): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let systemText = '';
  const parts: Json[] = [];

  for (const msg of messages) {
    if (!isRecord(msg)) continue;
    if (msg.role === 'system') {
      systemText = textOf(msg.content);
    } else if (msg.role === 'user') {
      if (typeof msg.content === 'string') {
        parts.push({ text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ text: block.text });
          } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const url = block.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              const meta = url.slice(5, comma);
              const data = url.slice(comma + 1);
              const mime = meta.split(';')[0] || 'image/png';
              parts.push({ inlineData: { mimeType: mime, data } });
            }
          }
        }
      }
    }
  }

  const gemini: Json = {
    contents: [{ role: 'user', parts: parts.length > 0 ? parts : [{ text: '' }] }],
  };
  if (systemText !== '') {
    gemini.systemInstruction = { parts: [{ text: systemText }] };
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const declarations = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function)) return [];
      return [
        {
          name: typeof t.function.name === 'string' ? t.function.name : '',
          description: typeof t.function.description === 'string' ? t.function.description : '',
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) {
      gemini.tools = [{ functionDeclarations: declarations }];
    }
  }
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (Array.isArray(body.stop) && body.stop.length > 0) generationConfig.stopSequences = body.stop;
  if (isRecord(body.response_format) && body.response_format.type === 'json_object') {
    generationConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(generationConfig).length > 0) {
    gemini.generationConfig = generationConfig;
  }
  return gemini;
}

/** Convert a non-streamed Gemini generateContent response into an OpenAI Chat Completions object. */
export function geminiToOpenAI(geminiRes: Json, model: string): Json {
  const candidates = Array.isArray(geminiRes.candidates) ? geminiRes.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const p of parts) {
    if (isRecord(p) && typeof p.text === 'string') text += p.text;
  }
  const meta = isRecord(geminiRes.usageMetadata) ? geminiRes.usageMetadata : {};
  const prompt_tokens = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const completion_tokens = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;

  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: 'stop',
      },
    ],
    usage: { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
  };
}

/** Stream translator: Gemini SSE stream -> OpenAI Chat Completions SSE frames. */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string, includeUsage: boolean): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let promptTokens = 0;
  let completionTokens = 0;

  const startFrame = (): string => {
    if (started) return '';
    started = true;
    return `data: ${JSON.stringify({
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
    })}\n\n`;
  };

  const framesOfChunk = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usageMetadata)) {
        promptTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
        completionTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand !== null) {
        if (!started) {
          const s = startFrame();
          if (s) out.push(s);
        }
        const content = isRecord(cand.content) ? cand.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const p of parts) {
          if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
            out.push(`data: ${JSON.stringify({
              id: 'chatcmpl-stream',
              object: 'chat.completion.chunk',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [{ index: 0, delta: { content: p.text }, finish_reason: null }],
            })}\n\n`);
          }
        }
        if (cand.finishReason === 'STOP' && !finished) {
          finished = true;
          out.push(`data: ${JSON.stringify({
            id: 'chatcmpl-stream',
            object: 'chat.completion.chunk',
            created: FIXED_OPENAI_CREATED,
            model,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          })}\n\n`);
          if (includeUsage) {
            out.push(`data: ${JSON.stringify({
              id: 'chatcmpl-stream',
              object: 'chat.completion.chunk',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [],
              usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
            })}\n\n`);
          }
          out.push('data: [DONE]\n\n');
        }
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';
      for (const frame of frames) {
        for (const f of framesOfChunk(frame)) yield f;
      }
    }
    if (buffer.trim() !== '') {
      for (const f of framesOfChunk(buffer)) yield f;
    }
    if (!finished) {
      if (!started) {
        const s = startFrame();
        if (s) yield s;
      }
      yield `data: ${JSON.stringify({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      })}\n\n`;
      if (includeUsage) {
        yield `data: ${JSON.stringify({
          id: 'chatcmpl-stream',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [],
          usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
        })}\n\n`;
      }
      yield 'data: [DONE]\n\n';
    }
  } finally {
    reader.releaseLock();
  }
}
