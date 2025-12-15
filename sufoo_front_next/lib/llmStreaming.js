/**
 * LLM SSE(POST) 스트리밍 유틸
 * - API_STREAMING_MANUAL.md 규격: event/data 라인 기반 SSE(text/event-stream)
 * - chunk 이벤트의 data는 보통 {"request_0":{"title":"안"}} 같은 JSON 문자열(패치)
 */
export async function postSSE(url, jsonBody, onEvent, options = {}) {
  const { signal } = options;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(jsonBody),
    signal,
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${await res.text()}`);
  }

  if (!res.body) {
    throw new Error("SSE 응답 바디(res.body)가 비어있습니다.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";
  let currentEvent = "message";
  let currentDataLines = [];

  function flushEvent() {
    const event = currentEvent || "message";
    const data = currentDataLines.join("\n");

    // keep-alive(빈 message) 방지
    if (event === "message" && data.length === 0) {
      currentEvent = "message";
      currentDataLines = [];
      return;
    }

    onEvent({ event, data });
    currentEvent = "message";
    currentDataLines = [];
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    // SSE는 빈 줄(\n\n)로 이벤트를 구분
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const rawEventBlock = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      const lines = rawEventBlock.split("\n");
      for (const line of lines) {
        if (!line) continue;
        if (line.startsWith(":")) continue; // comment/keep-alive
        if (line.startsWith("event:")) {
          currentEvent = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          currentDataLines.push(line.slice("data:".length).trimStart());
        }
      }
      flushEvent();
    }
  }

  // 스트림이 \n\n 없이 종료된 경우 마지막 블록 처리(가능하면)
  const remaining = buffer.trim();
  if (remaining.length > 0) {
    const lines = remaining.split("\n");
    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        currentEvent = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        currentDataLines.push(line.slice("data:".length).trimStart());
      }
    }
    flushEvent();
  }
}

export function deepClone(obj) {
  try {
    // 대부분의 최신 브라우저에서 지원
    return structuredClone(obj);
  } catch {
    return JSON.parse(JSON.stringify(obj));
  }
}

export function buildInitialLlmViewData(template) {
  const templateRequest = Array.isArray(template?.request) ? template.request : [];

  return {
    include_images: {
      request: templateRequest.map((reqItem) => ({
        title: typeof reqItem?.title === "string" ? reqItem.title : "",
        description: typeof reqItem?.description === "string" ? reqItem.description : "",
        result: typeof reqItem?.result === "string" ? reqItem.result : "",
        // 이미지 키는 서버가 제공할 수도 있으므로 미리 두되, 없으면 빈 문자열 유지
        image_url: typeof reqItem?.image_url === "string" ? reqItem.image_url : "",
        representative_image_name:
          typeof reqItem?.representative_image_name === "string"
            ? reqItem.representative_image_name
            : "",
        subject: Array.isArray(reqItem?.subject)
          ? reqItem.subject.map((sub) => ({
              sub_title: typeof sub?.sub_title === "string" ? sub.sub_title : "",
              sub_description:
                typeof sub?.sub_description === "string" ? sub.sub_description : "",
              sub_result: typeof sub?.sub_result === "string" ? sub.sub_result : "",
            }))
          : [],
      })),
    },
  };
}

export function normalizeLlmResultForView(result) {
  if (!result || typeof result !== "object") return result;

  if (result.include_images && Array.isArray(result.include_images.request)) {
    return result;
  }

  if (Array.isArray(result.request)) {
    return {
      ...result,
      include_images: {
        ...(result.include_images ?? {}),
        request: result.request,
      },
    };
  }

  return result;
}

function mergeText(prev, next) {
  if (typeof next !== "string") return next;
  if (typeof prev !== "string") return next;
  if (next.length === 0) return prev;
  if (prev.length === 0) return next;

  // 서버가 "현재까지 누적된 문자열"을 주는 경우(스냅샷)도 방어
  if (next.startsWith(prev)) return next;
  if (prev.startsWith(next)) return prev;

  return prev + next;
}

function ensureRequestItem(requestArray, idx) {
  while (requestArray.length <= idx) {
    requestArray.push({ title: "", description: "", result: "", subject: [] });
  }
  const item = requestArray[idx];
  if (!item || typeof item !== "object") {
    requestArray[idx] = { title: "", description: "", result: "", subject: [] };
  }
  if (!Array.isArray(requestArray[idx].subject)) {
    requestArray[idx].subject = [];
  }
}

function ensureSubjectItem(subjectArray, idx) {
  while (subjectArray.length <= idx) {
    subjectArray.push({ sub_title: "", sub_description: "", sub_result: "" });
  }
  const item = subjectArray[idx];
  if (!item || typeof item !== "object") {
    subjectArray[idx] = { sub_title: "", sub_description: "", sub_result: "" };
  }
}

function mergeSubjectPatch(targetSubjectItem, patchObj) {
  if (!patchObj || typeof patchObj !== "object") return;
  for (const [k, v] of Object.entries(patchObj)) {
    if (typeof v === "string") {
      targetSubjectItem[k] = mergeText(targetSubjectItem[k], v);
    } else {
      targetSubjectItem[k] = v;
    }
  }
}

function mergeRequestPatch(targetRequestItem, patchObj) {
  if (!patchObj || typeof patchObj !== "object") return;

  for (const [k, v] of Object.entries(patchObj)) {
    const subMatch = /^subject_(\d+)$/.exec(k);
    if (subMatch) {
      const subIdx = Number(subMatch[1]);
      if (!Array.isArray(targetRequestItem.subject)) targetRequestItem.subject = [];
      ensureSubjectItem(targetRequestItem.subject, subIdx);
      mergeSubjectPatch(targetRequestItem.subject[subIdx], v);
      continue;
    }

    if (typeof v === "string") {
      targetRequestItem[k] = mergeText(targetRequestItem[k], v);
    } else {
      targetRequestItem[k] = v;
    }
  }
}

function applyRequestKeyPatch(target, requestKey, requestPatchObj) {
  const match = /^request_(\d+)$/.exec(requestKey);
  if (!match) return false;

  const reqIdx = Number(match[1]);
  if (!target.include_images || typeof target.include_images !== "object") {
    target.include_images = {};
  }
  if (!Array.isArray(target.include_images.request)) {
    target.include_images.request = [];
  }
  ensureRequestItem(target.include_images.request, reqIdx);
  mergeRequestPatch(target.include_images.request[reqIdx], requestPatchObj);
  return true;
}

/**
 * chunk patch(JSON.parse(data))를 누적 데이터(target)에 반영합니다.
 * - 기본 반영 위치: target.include_images.request
 * - patch 키: request_0, request_1, ... / 내부: title/description/result, subject_0...
 */
export function applyLlmStreamPatch(target, patch) {
  if (!target || typeof target !== "object") return;
  if (!patch || typeof patch !== "object") return;

  for (const [k, v] of Object.entries(patch)) {
    // 가장 흔한 케이스: {"request_0": {...}}
    if (applyRequestKeyPatch(target, k, v)) continue;

    // 혹시 include_images 하위로 들어오는 케이스 방어: {"include_images": {"request_0": {...}}}
    if (k === "include_images" && v && typeof v === "object") {
      for (const [ik, iv] of Object.entries(v)) {
        applyRequestKeyPatch(target, ik, iv);
      }
      continue;
    }

    // 그 외(루트 레벨 문자열 필드 등)도 안전하게 누적
    if (typeof v === "string") {
      target[k] = mergeText(target[k], v);
    } else {
      target[k] = v;
    }
  }
}


