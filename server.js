// server.js
// Minimal RAG server: Qdrant + OpenAI + text/PDF/YouTube (LangChain YoutubeLoader version)

import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs/promises";
import fssync from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

import { QdrantClient } from "@qdrant/js-client-rest";
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube";

// ---------- paths / ESM helpers ----------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env (adjust/remove if you set env in terminal)
dotenv.config({ path: "D:/harshita/genaiJScohort/.env" });

// ---------- config ----------
const COLLECTION_NAME = "ragChat"; // persistent shared collection
const UPLOADS_DIR = "uploads";

// Validate critical envs (warn early; validate on use)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || null;
const YT_LANGUAGE = process.env.YT_LANGUAGE || "en";

if (!OPENAI_API_KEY) {
  console.warn("[WARN] OPENAI_API_KEY not set: embeddings and LLM calls will fail at runtime.");
}

// ---------- app bootstrap ----------
if (!fssync.existsSync(UPLOADS_DIR)) {
  fssync.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Limit files to 25MB to avoid memory spikes
const upload = multer({
  dest: UPLOADS_DIR + "/",
  limits: { fileSize: 25 * 1024 * 1024 },
});

// ---------- shared Qdrant + embeddings (singletons) ----------
const qdrantClient = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY || undefined,
});

const embeddings = new OpenAIEmbeddings({
  apiKey: OPENAI_API_KEY,
  model: "text-embedding-3-small",
  dimensions: 1536,
});

// ---------- helpers ----------
async function loadInline(text) {
  if (!text || !text.trim()) return [];
  return [new Document({ pageContent: text, metadata: { source: "inline" } })];
}

async function loadPdf(tempPath, originalName) {
  if (!tempPath) return [];
  try {
    await fs.access(tempPath);
  } catch {
    throw new Error(`Uploaded file cannot be accessed at ${tempPath}`);
  }

  try {
    // LangChain PDF loader (requires pdf-parse installed in this project)
    const loader = new PDFLoader(tempPath, { splitPages: false });
    const docs = await loader.load(); // Documents[]
    return docs.map((doc) => ({
      ...doc,
      metadata: {
        ...doc.metadata,
        source: originalName || "upload.pdf",
      },
    }));
  } catch (error) {
    console.error("PDF loading error:", error);
    return [
      new Document({
        pageContent: "",
        metadata: {
          source: originalName || "upload.pdf",
          note: `Error processing PDF: ${error?.message || error}`,
        },
      }),
    ];
  }
}

/**
 * Load YouTube transcript using LangChain's YoutubeLoader
 * - Accepts any standard YouTube URL (watch, youtu.be, embed, shorts, m.youtube.com, music.youtube.com)
 * - Attempts with language hint; still returns whatever captions are available if hint fails
 */
async function loadYouTube(url) {
  if (!url || !url.trim()) return [];

  // First attempt: with language hint
  try {
    const loader = YoutubeLoader.createFromUrl(url, {
      language: 'en',      // preferred language (e.g., "en")
      addVideoInfo: true,        // set true if you need title/author in metadata
      translation: 'en',     // could set "en" to fetch auto-translate where supported
    });
    const docs = await loader.load();
    const text = docs.map((d) => d.pageContent || "").join("\n").trim();

    if (text) {
      // Metadata: include url for provenance
      return [
        new Document({
          pageContent: text,
          metadata: { source: url, note: `YouTube transcript loaded via LangChain (lang=${YT_LANGUAGE})` },
        }),
      ];
    }
  } catch (err) {
    console.warn(`YouTube load attempt (lang=${YT_LANGUAGE}) failed:`, err?.message || err);
  }

  // Second attempt: without language hint (let loader decide best available)
  try {
    const loader = YoutubeLoader.createFromUrl(url, {
      addVideoInfo: false,
    });
    const docs = await loader.load();
    const text = docs.map((d) => d.pageContent || "").join("\n").trim();

    if (text) {
      return [
        new Document({
          pageContent: text,
          metadata: { source: url, note: "YouTube transcript loaded via LangChain (no lang hint)" },
        }),
      ];
    }
  } catch (err) {
    console.warn("YouTube load attempt (no lang hint) failed:", err?.message || err);
  }

  // If both attempts fail, return a placeholder with a diagnostic note
  return [
    new Document({
      pageContent: "",
      metadata: {
        source: url,
        note: "No transcript available or failed to fetch via LangChain YoutubeLoader",
      },
    }),
  ];
}

// Speech-friendly defaults: smaller chunk size and larger overlap
async function chunkDocs(docs, chunkSize = 800, chunkOverlap = 160) {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
  return splitter.splitDocuments(docs);
}

// Connect to existing collection (or create on first upsert)
async function makeStore(collectionName) {
  // fromExistingCollection will connect; if collection does not exist yet,
  // we will create it implicitly on first fromDocuments call in upsertChunks
  return QdrantVectorStore.fromExistingCollection(embeddings, {
    client: qdrantClient,
    collectionName,
  });
}

// Upsert using fromDocuments against the same collection/client
async function upsertChunks(chunks) {
  if (!chunks || chunks.length === 0) return 0;
  await QdrantVectorStore.fromDocuments(chunks, embeddings, {
    client: qdrantClient,
    collectionName: COLLECTION_NAME,
  });
  return chunks.length;
}

function makeLLM() {
  if (!OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY");
  }
  return new ChatOpenAI({
    apiKey: OPENAI_API_KEY,
    model: "gpt-4o-mini",
    temperature: 0.2,
  });
}

async function answer(store, query, k = 5) {
  const retriever = store.asRetriever(k);
  const docs = await retriever.getRelevantDocuments(query);

  const context = docs
    .map((d, i) => {
      const body = (d.pageContent || "").slice(0, 1200);
      const src = d.metadata?.source || "unknown";
      return `# Doc ${i + 1} (source: ${src})\n${body}`;
    })
    .join("\n\n");

  const llm = makeLLM();

  // System prompt encouraging careful reasoning and concise answers.
  const messages = [
  {
    role: "system",
    content:
      "👋 Hi! I’m your friendly RAG assistant. I’ll use only the provided context to help you quickly and clearly. " +
      "If the context isn’t enough, I’ll say so and suggest what to add. Keep answers short, practical, and easy to act on. " +
      "Use light emojis when helpful (✅, ⚠️). No guesses.",
  },
  {
    role: "user",
    content: `📚 Context:\n${context}\n\n❓ Question: ${query}\n\n💡 Answer:`,
  },
];


  const res = await llm.invoke(messages);
  const text =
    typeof res?.content === "string"
      ? res.content
      : Array.isArray(res?.content)
      ? res.content.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n")
      : JSON.stringify(res?.content);

  const sources = docs.map((d) => ({
    source: d.metadata?.source || "unknown",
    snippet: (d.pageContent || "").slice(0, 200),
  }));

  return { text, sources };
}

// ---------- sessions ----------
const sessions = new Map();
function newSessionId() {
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

let sharedStorePromise = null;
function getSharedStore() {
  if (!sharedStorePromise) sharedStorePromise = makeStore(COLLECTION_NAME);
  return sharedStorePromise;
}

// ---------- routes ----------
app.post("/process", upload.single("pdf"), async (req, res) => {
  try {
    const { inlineText, youtubeUrl } = req.body || {};
    const file = req.file || null;

    // Logs to help verify uploads (keep minimal)
    console.log("PROCESS fields:", {
      inlineLen: (inlineText || "").length,
      youtubeUrl: youtubeUrl || "",
      file: file ? { path: file.path, originalname: file.originalname, size: file.size } : null,
    });

    const parts = await Promise.all([
      loadInline(inlineText),
      file ? loadPdf(file.path, file.originalname) : Promise.resolve([]),
      loadYouTube(youtubeUrl),
    ]);
    const docs = parts.flat();

    if (file) {
      fs.unlink(file.path).catch(() => {});
    }

    const hasAnyContent = docs.some((d) => (d.pageContent || "").trim().length > 0);
    if (!hasAnyContent) {
      // Provide hints if YouTube provided no transcript, etc.
      const notes = docs.map((d) => d.metadata?.note).filter(Boolean);
      return res.status(400).json({
        error:
          "No extractable content. Tips: add some Inline Text, use a text-based PDF (not scanned), or a YouTube URL with captions.",
        notes,
      });
    }

    const chunks = await chunkDocs(docs);
    const store = await getSharedStore();
    const added = await upsertChunks(chunks);

    const sessionId = newSessionId();
    sessions.set(sessionId, { store });

    // Provide per-source meta for UI
    const sourcesMeta = docs.map((d) => ({
      source: d.metadata?.source || "unknown",
      note: d.metadata?.note || null,
      hasContent: Boolean((d.pageContent || "").trim()),
    }));

    res.json({ sessionId, chunks: chunks.length, added, sources: sourcesMeta });
  } catch (e) {
    console.error("PROCESS error:", e);
    res.status(500).json({ error: e?.message || "Failed to process" });
  }
});

app.post("/ask", async (req, res) => {
  try {
    const { sessionId, query } = req.body || {};
    if (!sessionId || !query) return res.status(400).json({ error: "sessionId and query are required" });

    const sess = sessions.get(sessionId);
    if (!sess) return res.status(404).json({ error: "Invalid or expired sessionId" });

    const result = await answer(sess.store, query);
    res.json(result);
  } catch (e) {
    console.error("ASK error:", e);
    res.status(500).json({ error: e?.message || "Failed to answer" });
  }
});

// Serve index.html from the current folder (since you keep index.html next to server.js)
app.use(express.static(__dirname));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`RAG server running on http://localhost:${PORT}`);
});
