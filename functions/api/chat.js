// Cek apakah ada pesan yang mengandung gambar (dipakai buat milih model yang support vision)
function hasImage(messages) {
  return (messages || []).some(m => Array.isArray(m.content) && m.content.some(c => c.type === 'image_url'));
}

// ===== GEMINI =====
async function callGemini(env, system, messages) {
  if (!env.GEMINI_API_KEY) {
    return { ok: false, retryable: true, errorMessage: 'GEMINI_API_KEY belum di-set' };
  }

  const contents = (messages || []).map(m => {
    if (typeof m.content === 'string') {
      return { role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] };
    }
    const parts = m.content.map(c => {
      if (c.type === 'text') return { text: c.text };
      if (c.type === 'image_url') {
        const base64 = c.image_url.url.split(',')[1];
        const mimeType = c.image_url.url.split(';')[0].split(':')[1];
        return { inline_data: { mime_type: mimeType, data: base64 } };
      }
      return { text: '' };
    });
    return { role: m.role === 'assistant' ? 'model' : 'user', parts };
  });

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${env.GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system_instruction: { parts: [{ text: system || '' }] }, contents })
    }
  );

  const raw = await res.text();
  let data;
  try { data = JSON.parse(raw); } catch (_) { data = {}; }

  if (!res.ok) {
    const isQuota = res.status === 429;
    const isServerErr = res.status >= 500;
    return { ok: false, retryable: isQuota || isServerErr, errorMessage: data?.error?.message || `HTTP ${res.status}` };
  }

  const candidate = data?.candidates?.[0];
  let text = candidate?.content?.parts?.map(p => p.text).join('\n') || '';

  if (!text.trim()) {
    const blockReason = data?.promptFeedback?.blockReason;
    const finishReason = candidate?.finishReason;
    if (blockReason) {
      return { ok: false, retryable: true, errorMessage: `Diblokir filter keamanan Gemini: ${blockReason}` };
    }
    if (finishReason && finishReason !== 'STOP') {
      return { ok: false, retryable: true, errorMessage: `Gemini berhenti tanpa hasil (${finishReason})` };
    }
    return { ok: false, retryable: true, errorMessage: 'Balasan Gemini kosong' };
  }

  return { ok: true, text };
}

// ===== Helper generik buat provider OpenAI-compatible (Groq, DeepSeek, OpenRouter, dst) =====
async function callOpenAICompatible({ baseUrl, apiKey, model, system, messages, extraHeaders }) {
  const chatMessages = [
    { role: 'system', content: system || '' },
    ...(messages || []).map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content // sudah format OpenAI-compatible (string atau array text/image_url)
    }))
  ];

  const res = await fetch(baseUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      ...(extraHeaders || {})
    },
    body: JSON.stringify({ model, messages: chatMessages })
  });

  const raw = await res.text();
  let data;
  try { data = JSON.parse(raw); } catch (_) { data = {}; }

  if (!res.ok) {
    const status = res.status;
    const isQuota = status === 429;
    const isServerErr = status >= 500;
    return { ok: false, retryable: isQuota || isServerErr, errorMessage: data?.error?.message || `HTTP ${status}` };
  }

  const text = data?.choices?.[0]?.message?.content || '';
  if (!text.trim()) {
    return { ok: false, retryable: true, errorMessage: 'Balasan kosong' };
  }
  return { ok: true, text };
}

// ===== Daftar provider fallback, dicoba berurutan kalau yang di atasnya gagal =====
// Tiap provider bisa punya flag `visionOnly`/`noVision` supaya provider yang gak support
// gambar otomatis dilewati kalau pesannya ada gambar (bukannya error nyasar).
function buildFallbackProviders(env, hasImg) {
  const providers = [];

  if (env.OPENROUTER_API_KEY) {
    providers.push({
      name: 'OpenRouter',
      call: (system, messages) => callOpenAICompatible({
        baseUrl: 'https://openrouter.ai/api/v1/chat/completions',
        apiKey: env.OPENROUTER_API_KEY,
        model: 'openrouter/free', // auto-router, otomatis pilih model gratis yg cocok (termasuk vision kalau perlu)
        system, messages,
        extraHeaders: { 'HTTP-Referer': 'https://office-suite.pages.dev', 'X-Title': 'Office Suite AI' }
      })
    });
  }

  if (env.DEEPSEEK_API_KEY && !hasImg) {
    // DeepSeek belum support input gambar, jadi cuma dipakai kalau percakapannya teks doang
    providers.push({
      name: 'DeepSeek',
      call: (system, messages) => callOpenAICompatible({
        baseUrl: 'https://api.deepseek.com/v1/chat/completions',
        apiKey: env.DEEPSEEK_API_KEY,
        model: 'deepseek-chat',
        system, messages
      })
    });
  }

  if (env.GROQ_API_KEY) {
    providers.push({
      name: 'Groq',
      call: (system, messages) => callOpenAICompatible({
        baseUrl: 'https://api.groq.com/openai/v1/chat/completions',
        apiKey: env.GROQ_API_KEY,
        model: hasImg ? 'qwen/qwen3.6-27b' : 'openai/gpt-oss-120b',
        system, messages
      })
    });
  }

  return providers;
}

export async function onRequestPost(context) {
  const { request, env } = context;
  const body = await request.json();
  const { system, messages } = body;
  const hasImg = hasImage(messages);

  const attempts = [];
  let result;
  let usedProvider = 'gemini';

  try {
    result = await callGemini(env, system, messages);
  } catch (e) {
    result = { ok: false, retryable: true, errorMessage: e.message };
  }
  if (!result.ok) attempts.push(`Gemini: ${result.errorMessage}`);

  if (!result.ok && result.retryable) {
    const fallbacks = buildFallbackProviders(env, hasImg);
    for (const provider of fallbacks) {
      try {
        const r = await provider.call(system, messages);
        if (r.ok) {
          result = r;
          usedProvider = provider.name;
          break;
        }
        attempts.push(`${provider.name}: ${r.errorMessage}`);
        if (!r.retryable) continue; // tetap lanjut ke provider berikutnya, cuma dicatat aja errornya
      } catch (e) {
        attempts.push(`${provider.name}: ${e.message}`);
      }
    }
  }

  if (!result.ok) {
    const detail = attempts.length ? attempts.join(' | ') : result.errorMessage;
    return new Response(JSON.stringify({ error: { message: detail } }), {
      status: 502,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
    });
  }

  const text = usedProvider === 'gemini'
    ? result.text
    : `${result.text}\n\n_(dijawab oleh ${usedProvider}, karena Gemini lagi kena limit/gangguan)_`;

  return new Response(JSON.stringify({ content: [{ type: 'text', text }], provider: usedProvider }), {
    status: 200,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
  });
}

export async function onRequestOptions() {
  return new Response(null, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  });
}
