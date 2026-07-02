import { useCallback, useEffect, useRef, useState } from "react";
import JSZip from "jszip";
import { DropZone } from "@/components/DropZone";
import { SplitPreview } from "@/components/SplitPreview";
import { BatchList } from "@/components/BatchList";
import { OutputOptions } from "@/components/OutputOptions";
import { processImage, checkHealth, downloadBlob } from "@/lib/api";
import type { AppMode, BackgroundColor, ImageItem, ProcessOptions } from "@/types";
import { isValidBackground } from "@/types";

declare const __APP_VERSION__: string;

const BACKGROUND_STORAGE_KEY = "pureremove-background";

interface SingleState {
  sourceUrl: string;
  resultUrl: string;
  resultBlob: Blob | null;
  isProcessing: boolean;
}

function toMsg(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  try { return JSON.stringify(e); } catch { return String(e); }
}

function loadStoredBackground(): BackgroundColor {
  try {
    const stored = localStorage.getItem(BACKGROUND_STORAGE_KEY);
    if (!stored) return { type: "Transparent" };
    const parsed: unknown = JSON.parse(stored);
    return isValidBackground(parsed) ? parsed : { type: "Transparent" };
  } catch {
    return { type: "Transparent" };
  }
}

function stemOf(name: string): string {
  return name.replace(/\.[^.]+$/, "") || "output";
}

export default function App() {
  const [mode, setMode] = useState<AppMode>("idle");
  const [single, setSingle] = useState<SingleState | null>(null);
  const [batchItems, setBatchItems] = useState<ImageItem[]>([]);
  const [isSavingBatch, setIsSavingBatch] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [background, setBackground] = useState<BackgroundColor>(loadStoredBackground);

  const singleFileRef = useRef<File | null>(null);
  const singleNameRef = useRef<string>("output");
  const bgInitRef = useRef(true);
  const objectUrlsRef = useRef<string[]>([]);

  // ── Gestion mémoire des object URLs ───────────────────────────────────────
  const trackUrl = useCallback((url: string) => {
    objectUrlsRef.current.push(url);
    return url;
  }, []);

  const revokeAllUrls = useCallback(() => {
    objectUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    objectUrlsRef.current = [];
  }, []);

  useEffect(() => () => revokeAllUrls(), [revokeAllUrls]);

  // ── Santé serveur / modèle ────────────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then((h) => {
        if (!h.model) setServerError("Modèle IA non chargé côté serveur — réessayez plus tard.");
      })
      .catch((e) => setServerError(toMsg(e)));
  }, []);

  const handleBackgroundChange = useCallback((bg: BackgroundColor) => {
    localStorage.setItem(BACKGROUND_STORAGE_KEY, JSON.stringify(bg));
    setBackground(bg);
  }, []);

  const getOptions = useCallback((): ProcessOptions => ({ background }), [background]);

  const showError = useCallback((msg: string) => {
    setGlobalError(msg);
    setTimeout(() => setGlobalError(null), 7000);
  }, []);

  // ── Process single ────────────────────────────────────────────────────────

  const processSingleFile = useCallback(async (file: File) => {
    singleFileRef.current = file;
    singleNameRef.current = stemOf(file.name);
    setMode("single");
    setSingle((prev) => ({
      sourceUrl: prev?.sourceUrl ?? "",
      resultUrl: prev?.resultUrl ?? "",
      resultBlob: prev?.resultBlob ?? null,
      isProcessing: true,
    }));

    const sourceUrl = trackUrl(URL.createObjectURL(file));
    setSingle((prev) => prev ? { ...prev, sourceUrl } : null);

    try {
      const resultBlob = await processImage(file, getOptions());
      const resultUrl = trackUrl(URL.createObjectURL(resultBlob));
      setSingle((prev) => prev
        ? { ...prev, resultUrl, resultBlob, isProcessing: false }
        : null
      );
    } catch (e) {
      showError(`Erreur de traitement : ${toMsg(e)}`);
      setSingle((prev) => prev ? { ...prev, isProcessing: false } : null);
    }
  }, [getOptions, showError, trackUrl]);

  // ── Retraitement auto quand le fond change ────────────────────────────────
  useEffect(() => {
    if (bgInitRef.current) { bgInitRef.current = false; return; }
    const file = singleFileRef.current;
    if (file && mode === "single") {
      processSingleFile(file);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [background]);

  // ── Batch ─────────────────────────────────────────────────────────────────

  const processBatch = useCallback(async (files: File[]) => {
    singleFileRef.current = null;
    // Libère les object URLs du batch précédent avant d'en créer de nouvelles
    revokeAllUrls();
    const items: ImageItem[] = files.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
      file: f,
      status: "pending",
    }));
    setBatchItems(items);
    setMode("batch");

    const options = getOptions();

    // Boucle séquentielle : le serveur sérialise l'inférence de toute façon
    for (let i = 0; i < items.length; i++) {
      setBatchItems((prev) =>
        prev.map((item, idx) => (idx === i ? { ...item, status: "processing" } : item))
      );
      try {
        const blob = await processImage(files[i], options);
        const url = trackUrl(URL.createObjectURL(blob));
        setBatchItems((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, status: "done", resultDataUrl: url, resultBlob: blob } : item
          )
        );
      } catch (e) {
        setBatchItems((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, status: "error", error: toMsg(e) } : item
          )
        );
      }
    }
  }, [getOptions, trackUrl]);

  // ── Dispatch fichiers ─────────────────────────────────────────────────────

  const handleFiles = useCallback(async (files: File[]) => {
    if (files.length === 1) {
      await processSingleFile(files[0]);
    } else {
      await processBatch(files);
    }
  }, [processSingleFile, processBatch]);

  // ── Coller depuis le presse-papier (événement navigateur) ────────────────
  useEffect(() => {
    const handler = (e: ClipboardEvent) => {
      if (serverError) return;
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) =>
        f.type.startsWith("image/")
      );
      if (files.length > 0) {
        e.preventDefault();
        const named = files.map((f, i) =>
          f.name ? f : new File([f], `collage_${i + 1}.png`, { type: f.type })
        );
        void handleFiles(named);
      }
    };
    window.addEventListener("paste", handler);
    return () => window.removeEventListener("paste", handler);
  }, [handleFiles, serverError]);

  // ── Actions single ────────────────────────────────────────────────────────

  const handleCopy = useCallback(async () => {
    if (!single?.resultBlob) return;
    try {
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": single.resultBlob }),
      ]);
    } catch (e) {
      showError(`Copie échouée : ${toMsg(e)}`);
    }
  }, [single, showError]);

  const handleSaveSingle = useCallback(() => {
    if (!single?.resultBlob) return;
    downloadBlob(single.resultBlob, `${singleNameRef.current}_nobg.png`);
  }, [single]);

  const handleReset = useCallback(() => {
    singleFileRef.current = null;
    setMode("idle");
    setSingle(null);
    setBatchItems([]);
    revokeAllUrls();
  }, [revokeAllUrls]);

  // ── Actions batch ─────────────────────────────────────────────────────────

  const handleSaveOne = useCallback((item: ImageItem) => {
    if (!item.resultBlob) return;
    downloadBlob(item.resultBlob, `${stemOf(item.name)}_nobg.png`);
  }, []);

  const handleSaveAll = useCallback(async () => {
    const doneItems = batchItems.filter((i) => i.status === "done" && i.resultBlob);
    if (doneItems.length === 0) return;

    setIsSavingBatch(true);
    try {
      const zip = new JSZip();
      for (const item of doneItems) {
        zip.file(`${stemOf(item.name)}_nobg.png`, item.resultBlob!);
      }
      const blob = await zip.generateAsync({ type: "blob" });
      downloadBlob(blob, "pureremove_batch.zip");
    } catch (e) {
      showError(`Erreur lors de la création du ZIP : ${toMsg(e)}`);
    } finally {
      setIsSavingBatch(false);
    }
  }, [batchItems, showError]);

  // ─── Rendu ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center">
            <img src="/logo.png" alt="PureRemove" className="w-full h-full object-cover" />
          </div>
          <span className="text-foreground font-bold text-lg tracking-tight">PureRemove</span>
          <span className="text-muted-foreground text-xs bg-secondary px-2 py-0.5 rounded-full">v{__APP_VERSION__}</span>
        </div>

        <OutputOptions value={background} onChange={handleBackgroundChange} disabled={single?.isProcessing} />
      </header>

      {/* ── Bandeau serveur indisponible ── */}
      {serverError && (
        <div className="mx-4 mt-4 p-4 rounded-xl bg-destructive/10 border border-destructive/30 flex gap-3 items-start">
          <svg className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
          <div className="text-sm text-destructive">
            <p className="font-semibold mb-0.5">Service indisponible</p>
            <p className="text-destructive/80">{serverError}</p>
          </div>
        </div>
      )}

      {/* ── Toast erreur globale ── */}
      {globalError && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-5 py-3 rounded-xl bg-destructive text-white text-sm shadow-2xl max-w-md text-center">
          {globalError}
        </div>
      )}

      {/* ── Zone principale ── */}
      <main className="flex-1 min-h-0 p-4">
        {mode === "idle" && (
          <DropZone onFiles={handleFiles} disabled={!!serverError} />
        )}
        {mode === "single" && single && (
          <SplitPreview
            originalSrc={single.sourceUrl}
            resultSrc={single.resultUrl || single.sourceUrl}
            onCopy={handleCopy}
            onSave={handleSaveSingle}
            onReset={handleReset}
            isProcessing={single.isProcessing}
          />
        )}
        {mode === "batch" && (
          <BatchList
            items={batchItems}
            onSaveAll={handleSaveAll}
            onSaveOne={handleSaveOne}
            onReset={handleReset}
            isSaving={isSavingBatch}
          />
        )}
      </main>

      <footer className="text-center py-3 text-xs text-neutral-500">
        <a href="https://heiphaistos.org/legal/" target="_blank" rel="noopener noreferrer" className="hover:underline">
          Mentions légales
        </a>
      </footer>
    </div>
  );
}
