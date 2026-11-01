// Account-backed provider settings; saved secrets are never returned to the browser.

import { useEffect, useRef, useState } from "react";
import { authFetch } from "../session";

const API = `${import.meta.env.VITE_API_URL}/api/account`;

export default function ApiKeySettings({ credits, onChange, onClose, onPrivacy, onTerms, onDeleteAccount }) {
  const dialog = useRef(null);
  const [settings, setSettings] = useState(null);
  const [provider, setProvider] = useState("huggingface");
  const [form, setForm] = useState({ hfToken: "", cfAccountId: "", cfToken: "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!dialog.current.open) dialog.current.showModal();
    let active = true;
    authFetch(`${API}/api-keys`).then(async (response) => {
      if (!response.ok) throw new Error("Could not load provider settings.");
      const data = await response.json();
      if (!active) return;
      setSettings(data);
      setProvider(data.selected === "cloudflare" ? "cloudflare" : "huggingface");
      setForm({ hfToken: "", cfAccountId: data.cloudflare.accountId || "", cfToken: "" });
    }).catch(() => { if (active) setError("Could not load provider settings. Close and try again."); });
    return () => { active = false; };
  }, []);

  const update = async (method, path, body) => {
    setError("");
    setSaving(true);
    try {
      const response = await authFetch(`${API}${path}`, {
        method,
        headers: { "Content-Type": "application/json" },
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.error || "Could not update provider settings.");
        return;
      }
      setSettings(data);
      setForm({ hfToken: "", cfAccountId: data.cloudflare.accountId || "", cfToken: "" });
      await onChange();
    } catch {
      setError("Provider settings are unavailable. Try again later.");
    } finally {
      setSaving(false);
    }
  };

  const save = (event) => {
    event.preventDefault();
    if (saved?.saved && !form.hfToken.trim() && !form.cfToken.trim() &&
        (provider !== "cloudflare" || form.cfAccountId.trim() === settings.cloudflare.accountId)) {
      update("PUT", "/provider", { provider });
      return;
    }
    update("PUT", "/api-keys", { provider, hfToken: form.hfToken.trim(),
      ...(provider === "cloudflare" ? { cfAccountId: form.cfAccountId.trim(), cfToken: form.cfToken.trim() } : {}) });
  };
  const saved = settings?.[provider];
  const disabled = saving || !settings;

  return (
    <dialog ref={dialog} className="api-keys-dialog" onClose={onClose} aria-labelledby="api-keys-title">
      <form onSubmit={save}>
        <div className="api-keys-heading">
          <h2 id="api-keys-title">Settings</h2>
          <button type="button" onClick={() => dialog.current.close()} aria-label="Close settings">×</button>
        </div>
        <h3 className="settings-section-heading">AI provider</h3>
        <p>{!settings ? "Loading settings…" : settings.selected === "free"
          ? credits && !credits.unlimited
            ? `${credits.remaining}/${credits.cap} free questions remaining today.`
            : "Using daily free credits with Curalink’s keys."
          : `Using your ${settings.selected === "cloudflare" ? "Cloudflare" : "Hugging Face"} credentials; daily question credits do not apply.`}</p>
        <p>Keys are saved encrypted on your account.</p>
        <label htmlFor="llm-provider">Answer provider</label>
        <select id="llm-provider" value={provider} onChange={(event) => setProvider(event.target.value)} disabled={disabled}>
          <option value="huggingface">Hugging Face</option>
          <option value="cloudflare">Cloudflare Workers AI</option>
        </select>
        {saved?.saved && <p>Saved credentials ending in …{saved.last4}. <button type="button"
          className="api-keys-remove" disabled={disabled}
          onClick={() => update("DELETE", `/api-keys/${provider}`)}>Remove</button></p>}
        <label htmlFor="hf-key">Hugging Face token</label>
        <input id="hf-key" type="password" autoComplete="off" value={form.hfToken}
          onChange={(event) => setForm({ ...form, hfToken: event.target.value })}
          required={provider === "huggingface" ? !saved?.saved : !settings?.huggingface.saved}
          placeholder={settings?.huggingface.saved ? "Saved — leave blank to keep" : ""}
          maxLength={1024} disabled={disabled} />
        {provider === "cloudflare" && (
          <>
            <label htmlFor="cf-account">Cloudflare account ID</label>
            <input id="cf-account" type="text" autoComplete="off" value={form.cfAccountId}
              onChange={(event) => setForm({ ...form, cfAccountId: event.target.value })} required maxLength={32} disabled={disabled} />
            <label htmlFor="cf-key">Cloudflare Workers AI token</label>
            <input id="cf-key" type="password" autoComplete="off" value={form.cfToken}
              onChange={(event) => setForm({ ...form, cfToken: event.target.value })}
              required={!saved?.saved || !!form.hfToken.trim() || form.cfAccountId.trim() !== settings?.cloudflare.accountId}
              placeholder={saved?.saved ? "Saved — leave blank to keep" : ""}
              maxLength={1024} disabled={disabled} />
            <p>Hugging Face handles research ranking; Cloudflare generates the answer.</p>
          </>
        )}
        {error && <p role="alert" className="api-keys-error">{error}</p>}
        <div className="api-keys-actions">
          <button type="submit" className="api-keys-save" disabled={disabled}>{saving ? "Checking…" : "Save and use"}</button>
        </div>
      </form>
      <section className="settings-account" aria-labelledby="settings-account-title">
        <h3 id="settings-account-title" className="settings-section-heading">Account</h3>
        <div className="settings-account-actions">
          <button type="button" onClick={() => { dialog.current.close(); onPrivacy(); }}>Privacy</button>
          <button type="button" onClick={() => { dialog.current.close(); onTerms(); }}>Terms</button>
          <button type="button" className="delete-account-btn"
            onClick={() => { dialog.current.close(); onDeleteAccount(); }}>Delete account</button>
        </div>
      </section>
    </dialog>
  );
}
