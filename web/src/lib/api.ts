import type { ProcessOptions } from "@/types";

/** Envoie une image au serveur et retourne le PNG détouré. */
export async function processImage(file: File | Blob, options: ProcessOptions): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("options", JSON.stringify(options));

  const res = await fetch("/api/process", { method: "POST", body: form });

  if (!res.ok) {
    if (res.status === 429) {
      throw new Error("Trop de requêtes — patientez une minute avant de réessayer.");
    }
    const msg = await res
      .json()
      .then((j: { error?: string }) => j.error)
      .catch(() => undefined);
    throw new Error(msg ?? `Erreur serveur (${res.status})`);
  }

  return res.blob();
}

export interface HealthStatus {
  status: string;
  model: boolean;
  version: string;
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Serveur indisponible (${res.status})`);
  return res.json();
}

/** Déclenche le téléchargement d'un blob côté navigateur. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener noreferrer";
  // Nécessaire sur Firefox : l'ancre doit être dans le DOM pour déclencher le DL
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Laisse le temps au navigateur de démarrer le téléchargement avant revoke
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
